# Copyright 2025 starVLA community. All rights reserved.
# Licensed under the MIT License, Version 1.0 (the "License");
# Implemented by [Junqiu YU / Fudan University] in [2025]. 
# Design and Merged by [Jinhui YE / HKUST University] in [2025].
"""
Qwen-GR00T Framework
A lightweight implementation that Qwen-VL + Flow-matching head to directly predict continuous actions
Flow-matching header is copyright from GR00T N1.5,
"""
from typing import List
from tqdm import tqdm
from typing import List, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from PIL import Image
from transformers import AutoVideoProcessor, AutoModel, AutoTokenizer, VJEPA2VideoProcessor

from starVLA.training.trainer_utils import initialize_overwatch

logger = initialize_overwatch(__name__)

# HuggingFace Default / LLaMa-2 IGNORE_INDEX (for labels)
IGNORE_INDEX = -100

from starVLA.model.framework.base_framework import baseframework
from starVLA.model.modules.vlm import get_vlm_model
from starVLA.model.modules.action_model.GR00T_ActionHeader import get_action_model, FlowmatchingActionHead
from starVLA.model.modules.world_model.vj2_predictor import VisionTransformerPredictorAC
from starVLA.model.modules.fusion.gating import LearnedGatingFusion
from starVLA.model.modules.projector.vj_to_dit import VJtoDiTProjection
from starVLA.model.modules.projector.correction_projector import CorrectionProjector
from starVLA.training.trainer_utils.trainer_tools import resize_images
from starVLA.model.tools import FRAMEWORK_REGISTRY

@FRAMEWORK_REGISTRY.register("VLA_JEPA")
class VLA_JEPA(baseframework):
    """
    Multimodal vision-language-action model.

    Components:
      - Qwen VL interface for fused language/vision token embeddings
      - DiT diffusion head for future action sequence modeling
      - JEPA world model for future frame prediction

    Focus: Predict future continuous actions conditioned on images + instruction.
    """

    def __init__(
        self,
        config: Optional[dict] = None,
        **kwargs,
    ) -> None:
        """
        Construct all submodules and cache key configuration values.

        Args:
            config: Hierarchical configuration (OmegaConf/dict) containing framework + trainer sections.
            **kwargs: Reserved for future overrides (unused).
        """
        super().__init__()
        self.config = config
        self.qwen_vl_interface = get_vlm_model(config=self.config)
        embodied_action_token = self.config.framework.vj2_model.get("embodied_action_token", "<|embodied_action|>")
        action_tokens, self.action_token_ids, self.embodied_action_token_id = self.expand_tokenizer(
            tokenizer=self.qwen_vl_interface.processor.tokenizer,
            special_action_token=self.config.framework.vj2_model.special_action_token,
            max_action_tokens=self.config.framework.action_model.action_horizon * 4,
            embodied_action_token=embodied_action_token
        )

        # TODO speical tokens

        # align dims --> we should put them to config or no?
        self.config.framework.action_model.diffusion_model_cfg.cross_attention_dim = self.qwen_vl_interface.model.config.hidden_size

        self.action_model: FlowmatchingActionHead = get_action_model(config=self.config)  # 修复后续引用

        self.future_action_window_size = config.framework.action_model.future_action_window_size
        self.past_action_window_size = config.framework.action_model.past_action_window_size
        self.chunk_len = self.past_action_window_size + 1 + self.future_action_window_size
        
        self.vj_encoder = AutoModel.from_pretrained(self.config.framework.vj2_model.base_encoder)
        self.vj_processor = AutoVideoProcessor.from_pretrained(self.config.framework.vj2_model.base_encoder)

        tubelet_size = self.vj_encoder.config.tubelet_size
        self.vj_predictor = VisionTransformerPredictorAC(
            num_frames=self.config.framework.vj2_model.num_frames//tubelet_size,
            img_size=((self.vj_encoder.config.image_size, self.vj_encoder.config.image_size)),
            tubelet_size=1,
            depth=self.config.framework.vj2_model.depth,
            num_heads=self.config.framework.vj2_model.num_heads,
            embed_dim=self.vj_encoder.config.hidden_size * 2, # multi view
            action_embed_dim=self.qwen_vl_interface.model.config.hidden_size,
            num_add_tokens=self.config.framework.vj2_model.num_action_tokens_per_timestep,
        )
        self.replace_prompt = "".join(
            [each * self.config.framework.vj2_model.num_action_tokens_per_timestep for each in
             action_tokens[:self.config.framework.vj2_model.num_frames//tubelet_size - 1]]
        )

        self.embodied_replace_prompt = "".join([embodied_action_token * self.config.framework.vj2_model.num_embodied_action_tokens_per_instruction])

        # Stage 3: correction token special tokens + projector (disabled by default)
        self._stage3_enabled     = False
        self.correction_projector = None
        self._correction_token_ids = None  # list of token IDs for <|correction_i|>

        # KF state (populated by load_kf; None = KF disabled)
        self._lds       = None
        self._kf_z      = None   # (latent_dim,) numpy
        self._kf_P      = None   # (latent_dim, latent_dim) numpy
        self._kf_Q      = None
        self._kf_R      = None

        # EMA state (populated by load_ema; None = EMA disabled)
        self._ema_alpha = None
        self._ema_y     = None   # (feat_dim,) numpy per batch item

        # Phase 2: Recurrent JEPA training modules (config-driven)
        recurrent_cfg = getattr(self.config.framework, "recurrent_jepa", None)
        if recurrent_cfg and getattr(recurrent_cfg, "enabled", False):
            vj_dim  = self.vj_encoder.config.hidden_size * 2           # 2816
            dit_dim = self.qwen_vl_interface.model.config.hidden_size   # 2048
            n_cond  = getattr(recurrent_cfg, "n_cond_tokens", 8)
            use_cos = getattr(recurrent_cfg, "use_cosine", True)
            self.fusion    = LearnedGatingFusion(vj_dim, use_cosine=use_cos)
            self.vj_to_dit = VJtoDiTProjection(vj_dim, dit_dim, n_cond)
            self._recurrent_training = True
            logger.info(f"[Recurrent-JEPA] Phase 2 modules initialised  n_cond={n_cond}")
        else:
            self._recurrent_training = False

        # Recurrent JEPA state (enabled by load_recurrent; None = disabled)
        self._recurrent_enabled = False
        self._vj_prev_obs   = None  # (B, spatial_tokens, vj_embed_dim*V)
        self._prev_action_t = None  # (B, (T-1)*num_add_tokens, qwen_H)
        self._last_vj_pred  = None  # (B, spatial_tokens, vj_embed_dim*V); set after each predict_action
        self._last_vj_pred  = None  # (B, spatial_tokens, vj_embed_dim*V) — set after each predict_action

    def load_kf(self, lds_path: str, q_noise: float = 0.1, r_noise: float = 5.0) -> None:
        """Load a trained LearnedLDS and enable KF filtering on embodied_action_tokens."""
        import sys
        sys.path.insert(0, "/home/choi/vjepa2")
        from src.models.kf.learned_lds import LearnedLDS
        self._lds  = LearnedLDS.load(lds_path)
        d          = self._lds.latent_dim
        self._kf_Q = q_noise * np.eye(d)
        self._kf_R = r_noise * np.eye(d)
        self.reset_kf()
        logger.info(f"[KF] Loaded LDS from {lds_path}  latent_dim={d}  q={q_noise}  r={r_noise}")

    def reset_kf(self) -> None:
        """Reset KF state to uninformed prior (call at episode start)."""
        if self._lds is None:
            return
        d          = self._lds.latent_dim
        self._kf_z = np.zeros(d, dtype=np.float32)
        self._kf_P = np.eye(d, dtype=np.float32)

    def load_ema(self, alpha: float) -> None:
        """Enable EMA smoothing on embodied_action_tokens (no offline training needed)."""
        self._ema_alpha = alpha
        self._ema_y     = None
        logger.info(f"[EMA] Enabled  alpha={alpha}")

    def reset_ema(self) -> None:
        """Reset EMA state (call at episode start)."""
        self._ema_y = None

    def load_recurrent(self) -> None:
        """Enable recurrent JEPA inference mode (Phase 1: sanity check)."""
        self._recurrent_enabled = True
        self._vj_prev_obs   = None
        self._prev_action_t = None
        self._last_vj_pred  = None
        logger.info("[Recurrent-JEPA] Enabled — will log pred/obs cosine_sim each step.")

    def reset_recurrent(self) -> None:
        """Reset recurrent JEPA state (call at episode start)."""
        self._vj_prev_obs   = None
        self._prev_action_t = None
        self._last_vj_pred  = None

    # ──────────────────────────────────────────────────────────────────────
    # Stage 3: QwenVL correction-token injection
    # ──────────────────────────────────────────────────────────────────────

    def load_stage3(
        self,
        n_correction_tokens: int = 8,
        use_lora: bool = False,
        lora_r: int = 16,
        lora_alpha: int = 32,
        lora_dropout: float = 0.05,
    ) -> None:
        """
        Enable Stage 3 mode:
          - Registers <|correction_i|> special tokens into QwenVL's tokenizer
          - Instantiates CorrectionProjector (vj_dim → qwen_dim)
          - Optionally wraps QwenVL with LoRA for joint adaptation
          - Builds correction_replace_prompt to be appended after action tokens

        Call once after model construction, before training or inference.
        """
        tokenizer = self.qwen_vl_interface.processor.tokenizer
        corr_tokens = [f"<|correction_{i}|>" for i in range(n_correction_tokens)]

        # Add only tokens not already in vocab
        new_tokens = [t for t in corr_tokens if t not in tokenizer.get_vocab()]
        if new_tokens:
            tokenizer.add_tokens(new_tokens, special_tokens=True)
            self.qwen_vl_interface.model.resize_token_embeddings(len(tokenizer))
            logger.info(f"[Stage3] Added {len(new_tokens)} correction tokens to tokenizer.")

        self._correction_token_ids = torch.tensor(
            [tokenizer.convert_tokens_to_ids(t) for t in corr_tokens], dtype=torch.long
        )

        vj_dim  = self.vj_encoder.config.hidden_size * 2
        qwen_dim = self.qwen_vl_interface.model.config.hidden_size
        self.correction_projector = CorrectionProjector(
            vj_dim=vj_dim, qwen_dim=qwen_dim, n_tokens=n_correction_tokens
        )
        # correction_replace_prompt: appended after self.replace_prompt for Stage 3
        self.correction_replace_prompt = "".join(corr_tokens)
        self._stage3_enabled = True
        self._vj_prev_obs    = None
        self._last_vj_pred   = None

        # Save reference to Qwen2_5_VLModel BEFORE any LoRA wrapping so that
        # _run_qwen_with_correction can access it directly regardless of PeftModel layers.
        self._qwen_inner_model = self.qwen_vl_interface.model.model  # Qwen2_5_VLModel

        if use_lora:
            from peft import LoraConfig, get_peft_model
            lora_config = LoraConfig(
                r=lora_r,
                lora_alpha=lora_alpha,
                target_modules=["q_proj", "v_proj"],
                lora_dropout=lora_dropout,
                bias="none",
            )
            self.qwen_vl_interface.model = get_peft_model(
                self.qwen_vl_interface.model, lora_config
            )
            self._stage3_lora = True
            logger.info(
                f"[Stage3] LoRA applied to QwenVL  r={lora_r}  alpha={lora_alpha}  "
                f"target=[q_proj, v_proj]"
            )
        else:
            self._stage3_lora = False

        logger.info(f"[Stage3] Correction projector ready  n_tokens={n_correction_tokens}  "
                    f"vj_dim={vj_dim}  qwen_dim={qwen_dim}  lora={use_lora}")

    def _run_qwen_with_correction(
        self,
        batch_images: list,
        instructions: list,
        delta_z: torch.Tensor,
    ) -> torch.Tensor:
        """
        Run QwenVL with Δz-projected correction tokens injected between
        action_tokens and embodied_action_tokens.

        Steps:
          1. Build qwen_inputs with extended replace_prompt
             ({actions} → action_tokens + correction_token_placeholders)
          2. Manually get text embeddings + merge image features
          3. Replace correction token positions with CorrectionProjector(Δz)
          4. Compute correct position_ids (preserves 3D RoPE for vision tokens)
          5. Run inner language model; return last hidden states

        Args:
            batch_images:  List[List[PIL.Image]] of shape [B][V]
            instructions:  List[str] of length B
            delta_z:       (B, spatial, vj_dim) prediction error tensor (on model device)

        Returns:
            last_hidden: (B, L, qwen_dim) final hidden states from QwenVL
        """
        # Stage 3 replace prompt = action_tokens + correction_placeholder_tokens
        stage3_replace = self.replace_prompt + self.correction_replace_prompt

        qwen_inputs = self.qwen_vl_interface.build_qwenvl_inputs(
            images=batch_images,
            instructions=instructions,
            prompt_replace_dict={
                "{actions}":   stage3_replace,
                "{e_actions}": self.embodied_replace_prompt,
            },
            prompt_template=self.config.datasets.vla_data.get("CoT_prompt", ""),
        )

        input_ids      = qwen_inputs["input_ids"]
        attention_mask = qwen_inputs.get("attention_mask")
        pixel_values   = qwen_inputs.get("pixel_values")
        image_grid_thw = qwen_inputs.get("image_grid_thw")

        # Qwen2_5_VLModel — use cached reference so PeftModel wrapping doesn't shift the path
        inner_model = getattr(self, "_qwen_inner_model", self.qwen_vl_interface.model.model)

        with torch.autocast("cuda", dtype=torch.bfloat16):
            # 1. Text embeddings (image placeholder positions still have text embedding)
            inputs_embeds = inner_model.get_input_embeddings()(input_ids)

            # 2. Merge image features (call visual directly to avoid split/tuple ambiguity)
            if pixel_values is not None:
                pv = pixel_values.type(inner_model.visual.dtype)
                image_embeds = inner_model.visual(pv, grid_thw=image_grid_thw)
                if isinstance(image_embeds, tuple):
                    image_embeds = image_embeds[0]
                image_embeds = image_embeds.to(inputs_embeds.device, inputs_embeds.dtype)
                image_mask, _ = inner_model.get_placeholder_mask(
                    input_ids, inputs_embeds=inputs_embeds, image_features=image_embeds
                )
                inputs_embeds = inputs_embeds.masked_scatter(image_mask, image_embeds)

            # 3. Inject correction embeddings at <|correction_i|> positions
            corr_ids = self._correction_token_ids.to(input_ids.device)
            corr_mask = torch.isin(input_ids, corr_ids)  # (B, L)
            corr_embeds = self.correction_projector(delta_z.to(inputs_embeds.dtype))  # (B, N, qwen_dim)
            inputs_embeds[corr_mask] = corr_embeds.reshape(-1, inputs_embeds.shape[-1])

            # 4. Compute position_ids preserving 3D RoPE for vision tokens
            position_ids, rope_deltas = inner_model.get_rope_index(
                input_ids, image_grid_thw, None, attention_mask=attention_mask
            )
            inner_model.rope_deltas = rope_deltas

            # 5. Run language model directly (skip vision processing in forward)
            outputs = inner_model.language_model(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                position_ids=position_ids,
                output_hidden_states=True,
                return_dict=True,
            )

        return outputs.hidden_states[-1]  # (B, L, qwen_dim)

    @torch.inference_mode()
    def _predict_action_stage3(
        self,
        batch_images: list,
        instructions: list,
        state=None,
        **kwargs,
    ) -> dict:
        """
        Stage 3 inference:
          1. V-JEPA encode current frame → vj_obs_t
          2. Δz = vj_obs_t - _last_vj_pred  (zeros at t=0)
          3. Run QwenVL with correction injection → last_hidden
          4. Extract action_tokens, embodied_action_tokens
          5. V-JEPA predictor → vj_pred for next step
          6. DiT → action
        """
        # Step 1: V-JEPA encode (needed before QwenVL to compute Δz)
        vj_obs_t = self._encode_vj_from_images(batch_images)  # (B, spatial, D)
        B = vj_obs_t.shape[0]

        # Step 2: Δz
        if self._last_vj_pred is not None:
            delta_z = (vj_obs_t - self._last_vj_pred.to(vj_obs_t.device)).float()
        else:
            delta_z = torch.zeros_like(vj_obs_t).float()

        # Step 3: QwenVL with correction injection
        last_hidden = self._run_qwen_with_correction(batch_images, instructions, delta_z)
        # last_hidden: (B, L, qwen_dim)

        # Step 4a: embodied_action_tokens
        emb_id_tensor = torch.tensor([self.embodied_action_token_id], device=last_hidden.device)
        # We need input_ids to find positions — rebuild without autocast side effects
        qwen_inputs_tmp = self.qwen_vl_interface.build_qwenvl_inputs(
            images=batch_images,
            instructions=instructions,
            prompt_replace_dict={
                "{actions}":   self.replace_prompt + self.correction_replace_prompt,
                "{e_actions}": self.embodied_replace_prompt,
            },
            prompt_template=self.config.datasets.vla_data.get("CoT_prompt", ""),
        )
        input_ids_tmp = qwen_inputs_tmp["input_ids"]

        emb_mask = torch.isin(input_ids_tmp, emb_id_tensor).nonzero(as_tuple=True)
        H = last_hidden.shape[-1]
        embodied_action_tokens = last_hidden[emb_mask[0], emb_mask[1], :].view(B, -1, H)

        # Step 4b: action_tokens for vj_predictor
        act_id_tensor = torch.tensor(self.action_token_ids, device=last_hidden.device)
        act_mask = torch.isin(input_ids_tmp, act_id_tensor).nonzero(as_tuple=True)
        action_tokens_for_pred = last_hidden[act_mask[0], act_mask[1], :].view(B, -1, H)

        # Step 5: V-JEPA predictor → vj_pred for next step
        tubelet  = self.vj_encoder.config.tubelet_size
        T_pred   = self.config.framework.vj2_model.num_frames // tubelet - 1
        num_add  = self.config.framework.vj2_model.num_action_tokens_per_timestep
        prev_exp = vj_obs_t.repeat(1, T_pred, 1)
        prev_act = action_tokens_for_pred[:, : T_pred * num_add, :]
        with torch.autocast("cuda", dtype=torch.bfloat16):
            vj_pred_raw = self.vj_predictor(prev_exp, prev_act)
        spatial = vj_obs_t.shape[1]
        vj_pred_t = vj_pred_raw[:, -spatial:, :].float()

        # Update recurrent state
        self._vj_prev_obs   = vj_obs_t.detach()
        self._last_vj_pred  = vj_pred_t.detach()
        self._prev_action_t = action_tokens_for_pred.detach()

        # Step 6: DiT → action
        state_tensor = (
            torch.from_numpy(np.array(state)).to(last_hidden.device, dtype=last_hidden.dtype)
            if state is not None else None
        )
        with torch.autocast("cuda", dtype=torch.float32):
            pred_actions = self.action_model.predict_action(embodied_action_tokens, state_tensor)

        return {
            "normalized_actions":    pred_actions.detach().cpu().numpy(),
            "embodied_action_tokens": embodied_action_tokens.float().detach().cpu().numpy(),
        }

    def _encode_vj_from_images(self, batch_images: list) -> torch.Tensor:
        """Encode current-frame PIL images through V-JEPA encoder.

        Replicates each PIL image to fill the encoder's temporal dimension,
        runs vj_encoder, and returns the LAST frame's spatial tokens so that
        the shape matches one timestep of training embeddings.

        Returns: (B, spatial_tokens, vj_hidden * V)  on vj_encoder's device
        """
        num_frames = self.config.framework.vj2_model.num_frames  # e.g. 8
        B = len(batch_images)
        V = len(batch_images[0])

        input_videos = []
        for b in range(B):
            for v in range(V):
                img_np = np.array(batch_images[b][v])          # (H, W, 3) uint8
                frame  = torch.from_numpy(img_np).permute(2, 0, 1)  # (3, H, W)
                frames = frame.unsqueeze(0).repeat(num_frames, 1, 1, 1)  # (T, 3, H, W)
                processed = self.vj_processor(
                    videos=frames, return_tensors="pt"
                )["pixel_values_videos"].to(self.vj_encoder.device)  # (1, T, 3, H, W)
                input_videos.append(processed)

        input_videos = torch.cat(input_videos, dim=0)  # (B*V, T, 3, H, W)

        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            embeddings = self.vj_encoder.get_vision_features(
                pixel_values_videos=input_videos
            )  # (B*V, T//tubelet * spatial, hidden)

        embeddings = torch.cat(torch.chunk(embeddings, chunks=V, dim=0), dim=2)
        # (B, T//tubelet * spatial, V*hidden)

        tubelet  = self.vj_encoder.config.tubelet_size
        T_enc    = num_frames // tubelet
        spatial  = embeddings.shape[1] // T_enc
        return embeddings[:, -spatial:, :].float()  # (B, spatial, V*hidden)

    def _ema_step(self, y_obs: np.ndarray) -> np.ndarray:
        """One EMA step.  y_obs: (feat_dim,) → returns smoothed (feat_dim,)."""
        if self._ema_y is None:
            self._ema_y = y_obs.copy()
        else:
            self._ema_y = self._ema_alpha * y_obs + (1 - self._ema_alpha) * self._ema_y
        return self._ema_y.copy()

    def _kf_step(self, y_obs: np.ndarray) -> np.ndarray:
        """One KF predict+update step.  y_obs: (feat_dim,) → returns filtered (feat_dim,)."""
        z_obs  = y_obs @ self._lds.E.T                              # encode to latent
        z_pred = self._lds.A @ self._kf_z                           # predict
        P_pred = self._lds.A @ self._kf_P @ self._lds.A.T + self._kf_Q
        S      = P_pred + self._kf_R
        K      = P_pred @ np.linalg.solve(S.T, np.eye(S.shape[0])).T
        self._kf_z = z_pred + K @ (z_obs - z_pred)                 # update state
        self._kf_P = (np.eye(len(self._kf_z)) - K) @ P_pred
        return self._kf_z @ self._lds.E                             # decode

    def expand_tokenizer(self, 
                         tokenizer: AutoTokenizer,
                         special_action_token: str = "<|action_{}|>",
                         max_action_tokens: int = 32,
                         embodied_action_token: str = "<|embodied_action|>"):
        action_tokens, action_token_ids = [], []
        for i in range(0, max_action_tokens):
            action_token_i = special_action_token.format(i)
            action_tokens.append(action_token_i)
            if action_token_i not in tokenizer.get_vocab():
                added = tokenizer.add_tokens([action_token_i], special_tokens=True)
                if added == 0:
                    logger.warning(f"Warning: 0 tokens added (they may already exist) action_token_i: {action_token_i}.")
            action_token_id = tokenizer.convert_tokens_to_ids(action_token_i)    
            action_token_ids.append(action_token_id)
        
        if embodied_action_token not in tokenizer.get_vocab():
            added = tokenizer.add_tokens([embodied_action_token], special_tokens=True)
            if added == 0:
                logger.warning(f"Warning: 0 tokens added (they may already exist) embodied_action_token: {embodied_action_token}.")
        embodied_action_token_id = tokenizer.convert_tokens_to_ids(embodied_action_token)

        vla_embedding_size = self.qwen_vl_interface.model.get_input_embeddings().weight.size(0)
        if vla_embedding_size < len(tokenizer):
            # 2) resize embeddings of vla
            self.qwen_vl_interface.model.resize_token_embeddings(len(tokenizer))
        logger.info(f"Model embedding size: {vla_embedding_size} ;tokenizer.vocab_size: {len(tokenizer)}")
        return action_tokens, action_token_ids, embodied_action_token_id

    def forward(
        self,
        examples: List[dict] = None,
        **kwargs,
    ) -> Tuple:
        """

        """
        batch_images = [example["image"] for example in examples]  # [B, [PIL.Image]]
        batch_videos = [example["video"] for example in examples]  #  [B, V, T, H, W, 3]
        instructions = [example["lang"] for example in examples]  # [B, str]
        actions = [example["action"]for example in examples] if "action" in examples[0] else None # label [B， len, 7]
        
        state = [example["state"] for example in examples] if "state" in examples[0] else None  # [B, 1, state_dim]

        """
        if self.action_model.device == torch.device("cuda:0") and "action" in examples[0]:
            print(batch_videos[0].shape) #[V, T, H, W, 3]
            print(instructions[0])
            print(actions[0].shape) # [T-1, action_dim]
            print(state[0].shape) if state is not None else print("No state") #[state_dim]
            print(len(batch_videos), len(instructions), len(actions), len(state) if state is not None else "No state")
            from diffusers.utils import export_to_video
            export_to_video(batch_videos[0][0]/255.0, "data_view_0.mp4")
            export_to_video(batch_videos[0][1]/255.0, "data_view_1.mp4")
            batch_images[0][0].save("data_image_view_0.png")
            batch_images[0][1].save("data_image_view_1.png")
            #print(self.action_tokens)
            print(self.replace_prompt)
            print(self.action_token_ids)
        elif self.action_model.device == torch.device("cuda:0") and "action" not in examples[0]:
            print(batch_videos[0].shape) #[V, T, H, W, 3]
            print(instructions[0])
            print(len(batch_videos), len(instructions))
            from diffusers.utils import export_to_video
            export_to_video(batch_videos[0][0]/255.0, "video_view_0.mp4")
            export_to_video(batch_videos[0][1]/255.0, "video_view_1.mp4")
            batch_images[0][0].save("video_image_view_0.png")
        exit()
        """
        
        

        #[print(each.shape, end=";") for each in batch_videos]
        batch_videos = np.stack(batch_videos)  #  [B, V, T, H, W, 3]
        batch_videos = batch_videos.transpose(0,1,2,5,3,4)  # [B, V, T, 3, H, W]

        # Step 1: QWenVL input format
        if actions is not None:
            qwen_inputs = self.qwen_vl_interface.build_qwenvl_inputs(
                images=batch_images, 
                instructions=instructions,
                prompt_replace_dict={"{actions}":self.replace_prompt, "{e_actions}":self.embodied_replace_prompt},
                prompt_template=self.config.datasets.vla_data.get("CoT_prompt", "")) 
        else:
            qwen_inputs = self.qwen_vl_interface.build_qwenvl_inputs(
                images=batch_images, 
                instructions=instructions,
                prompt_replace_dict={"{actions}":self.replace_prompt},
                prompt_template=self.config.datasets.video_data.get("CoT_prompt", ""))
        
        action_indices = torch.isin(qwen_inputs['input_ids'], torch.tensor(self.action_token_ids, device=qwen_inputs['input_ids'].device))
        action_indices = action_indices.nonzero(as_tuple=True)

        # TODO action condition tokens
        #embodied_action_indices = torch.isin(qwen_inputs['input_ids'], torch.tensor([self.embodied_action_token_id], device=qwen_inputs['input_ids'].device))
        embodied_action_indices = torch.isin(qwen_inputs['input_ids'], torch.tensor([self.embodied_action_token_id], device=qwen_inputs['input_ids'].device))
        embodied_action_indices = embodied_action_indices.nonzero(as_tuple=True)
        
        with torch.autocast("cuda", dtype=torch.bfloat16):
            qwenvl_outputs = self.qwen_vl_interface(
                **qwen_inputs,
                output_attentions=False,
                output_hidden_states=True,
                return_dict=True,
            )
            # last_hidden_state: [B, seq_len, H]
            last_hidden = qwenvl_outputs.hidden_states[-1]   # [B, L, H]
            B, _, H = last_hidden.shape
            action_tokens = last_hidden[action_indices[0], action_indices[1], :].view(B, -1, H)  # [B, action_len, H]
            embodied_action_tokens = last_hidden[embodied_action_indices[0], embodied_action_indices[1], :].view(B, -1, H)  # [B, action_len, H]
            #print(action_tokens.shape, last_hidden.shape, embodied_action_tokens.shape)
            #exit()
        
            # Step 2: JEPA Encoder
            B, V, T, C, H, W = batch_videos.shape
            batch_videos = batch_videos.reshape(B*V, T, C, H, W)  # [B*V, T, C, H, W]
            input_videos = []
            for i in range(B*V):
                input_videos.append(self.vj_processor(
                    videos=batch_videos[i], return_tensors="pt"
                )["pixel_values_videos"].to(self.vj_encoder.device))
            input_videos = torch.cat(input_videos, dim=0)  # [B*V, T, C, H, W]
            with torch.no_grad():
                video_embeddings = self.vj_encoder.get_vision_features(pixel_values_videos=input_videos)
                video_embeddings = torch.cat(torch.chunk(video_embeddings, chunks=V, dim=0), dim=2)
            #print(video_embeddings.shape) # [B, T//tubelet_size * dim_per_frame, V*embed_dim]
        
            # Step 3: VJ Predictor
            T = T // self.vj_encoder.config.tubelet_size
            input_states = video_embeddings[:, :video_embeddings.shape[1] // T * (T-1),:]  # [B, (T-1)*dim_per_frame, V*embed_dim]
            gt_states = video_embeddings[:, video_embeddings.shape[1] // T:, :]
            #print(input_states.shape, action_tokens.shape)
            #exit()
            predicted_states = self.vj_predictor(
                input_states,
                action_tokens
            )

            teacher_forcing_wm_loss = F.l1_loss(
                predicted_states,
                gt_states,
                reduction="mean"
            )

            # Phase 2: fuse last obs + last prediction → conditioning for DiT
            cond_t = None
            if self._recurrent_training:
                spatial      = video_embeddings.shape[1] // T
                vj_obs_last  = video_embeddings[:, -spatial:, :].detach()   # (B, 256, 2816)
                vj_pred_last = predicted_states[:, -spatial:, :].detach()   # (B, 256, 2816)
                fused_t = self.fusion(vj_obs_last.float(), vj_pred_last.float())
                cond_t  = self.vj_to_dit(fused_t)                           # (B, n_cond, 2048)

        if "action" not in examples[0]:
            return {"wm_loss": teacher_forcing_wm_loss}

        # Step 4: Action Expert Forward and Loss
        with torch.autocast("cuda", dtype=torch.float32):
            actions = torch.tensor(
                np.array(actions), device=last_hidden.device, dtype=last_hidden.dtype
            )  # [B, T_full, action_dim]
            actions_target = actions[:, -(self.future_action_window_size+1):, :]  # (B, chunk_len, action_dim)

            repeated_diffusion_steps = (
                self.config.trainer.get("repeated_diffusion_steps", 4) if self.config and self.config.trainer else 4
            )
            actions_target_repeated = actions_target.repeat(repeated_diffusion_steps, 1, 1)
            embodied_action_repeated = embodied_action_tokens.repeat(repeated_diffusion_steps, 1, 1)

            # Concat Cond_t (from world model) with embodied_action tokens
            if cond_t is not None:
                cond_t_repeated  = cond_t.repeat(repeated_diffusion_steps, 1, 1)
                dit_input = torch.cat([embodied_action_repeated, cond_t_repeated], dim=1)
            else:
                dit_input = embodied_action_repeated

            state_repeated = None
            if state is not None:
                state = torch.tensor(
                    np.array(state), device=last_hidden.device, dtype=last_hidden.dtype
                )
                state_repeated = state.repeat(repeated_diffusion_steps, 1, 1)

            action_loss = self.action_model(dit_input, actions_target_repeated, state_repeated)

        return {"action_loss": action_loss, "wm_loss": teacher_forcing_wm_loss * 0.1}

    @torch.inference_mode()
    def predict_action(
        self,
        batch_images: List[List[Image.Image]],  # Batch of PIL Image list as [view1, view2]
        instructions: List[str],
        state: Optional[np.ndarray] = None,
        **kwargs: str,
    ) -> np.ndarray:
        """
        推理：单次前向直接回归未来动作（无扩散采样）。

        Steps:
          1. Resize images to training resolution (if specified)
          2. Encode with QwenVL (hidden states retained)
          6. Return normalized action trajectory

        Args:
            batch_images: List of samples; each sample is List[PIL.Image] (multi-view).
            instructions: List[str] natural language task instructions.
            cfg_scale: >1 enables classifier-free guidance (scales conditional vs unconditional).
            use_ddim: Whether to use DDIM deterministic sampling.
            num_ddim_steps: Number of DDIM steps if enabled.
            **kwargs: Reserved.

        Returns:
            dict:
                normalized_actions (np.ndarray): Shape [B, T, action_dim], diffusion-sampled normalized actions.
        """
        train_obs_image_size = getattr(self.config.datasets.vla_data, "image_size", None)
        if train_obs_image_size:
            batch_images = resize_images(batch_images, target_size=train_obs_image_size)

        # ── Stage 3: correction-token injection path ──────────────────────
        if self._stage3_enabled:
            return self._predict_action_stage3(batch_images, instructions, state, **kwargs)

        # Step 1: QWenVL input format
        qwen_inputs = self.qwen_vl_interface.build_qwenvl_inputs(
            images=batch_images, 
            instructions=instructions,
            prompt_replace_dict={"{actions}":self.replace_prompt, "{e_actions}":self.embodied_replace_prompt})
        
        embodied_action_indices = torch.isin(qwen_inputs['input_ids'], torch.tensor([self.embodied_action_token_id], device=qwen_inputs['input_ids'].device))
        #embodied_action_indices = ~torch.isin(qwen_inputs['input_ids'], torch.tensor(self.action_token_ids, device=qwen_inputs['input_ids'].device))
        embodied_action_indices = embodied_action_indices.nonzero(as_tuple=True)

        with torch.autocast("cuda", dtype=torch.bfloat16):
            qwenvl_outputs = self.qwen_vl_interface(
                **qwen_inputs,
                output_attentions=False,
                output_hidden_states=True,
                return_dict=True,
            )
            # last_hidden_state: [B, seq_len, H]
            last_hidden = qwenvl_outputs.hidden_states[-1]   # [B, L, H]
            B, _, H = last_hidden.shape
            embodied_action_tokens = last_hidden[embodied_action_indices[0], embodied_action_indices[1], :].view(B, -1, H)

            # Extract <|action_i|> tokens for vj_predictor conditioning
            action_indices_for_pred = torch.isin(
                qwen_inputs['input_ids'],
                torch.tensor(self.action_token_ids, device=qwen_inputs['input_ids'].device)
            ).nonzero(as_tuple=True)
            action_tokens_for_pred = last_hidden[
                action_indices_for_pred[0], action_indices_for_pred[1], :
            ].view(B, -1, H)  # (B, (T-1)*num_add_tokens, H)

        # Recurrent JEPA inference loop (Phase 1: logging / Phase 2: fusion)
        cond_t_inf = None
        if self._recurrent_enabled or self._recurrent_training:
            if kwargs.get("reset_kf", False):
                self.reset_recurrent()

            vj_obs_t = self._encode_vj_from_images(batch_images)  # (B, spatial, D)

            if self._vj_prev_obs is not None:
                tubelet  = self.vj_encoder.config.tubelet_size
                T_pred   = self.config.framework.vj2_model.num_frames // tubelet - 1  # T-1
                prev_exp = self._vj_prev_obs.repeat(1, T_pred, 1)  # (B, T_pred*spatial, D)
                num_add  = self.config.framework.vj2_model.num_action_tokens_per_timestep
                prev_act = self._prev_action_t[:, :T_pred * num_add, :]

                with torch.autocast("cuda", dtype=torch.bfloat16):
                    vj_pred_raw = self.vj_predictor(prev_exp, prev_act)
                spatial      = vj_obs_t.shape[1]
                vj_pred_last = vj_pred_raw[:, -spatial:, :].float()  # (B, spatial, D)

                cos_sim = F.cosine_similarity(vj_obs_t, vj_pred_last, dim=-1).mean().item()
                logger.info(f"[Recurrent-JEPA] pred/obs cosine_sim = {cos_sim:.4f}")

                self._last_vj_pred = vj_pred_last.detach()

                if self._recurrent_training:
                    fused_inf  = self.fusion(vj_obs_t, vj_pred_last)
                    cond_t_inf = self.vj_to_dit(fused_inf)          # (B, n_cond, dit_dim)
            else:
                # cold start: no prediction yet — use obs only for conditioning
                self._last_vj_pred = vj_obs_t.detach()
                if self._recurrent_training:
                    cond_t_inf = self.vj_to_dit(vj_obs_t)

            # Update recurrent state
            self._vj_prev_obs   = vj_obs_t.detach()
            self._prev_action_t = action_tokens_for_pred.detach()

        # KF filtering on embodied_action_tokens (if LDS loaded)
        if self._lds is not None:
            if kwargs.get("reset_kf", False):
                self.reset_kf()
            tokens_np  = embodied_action_tokens.float().cpu().numpy()  # (B, n_tok, H)
            y_raw      = tokens_np.mean(axis=1)                        # (B, H) mean-pool
            y_filtered = np.stack([self._kf_step(y_raw[b]) for b in range(tokens_np.shape[0])], axis=0)
            correction = torch.from_numpy(y_filtered - y_raw).to(embodied_action_tokens)
            embodied_action_tokens = embodied_action_tokens + correction.unsqueeze(1)

        # EMA smoothing on embodied_action_tokens (if EMA enabled)
        elif self._ema_alpha is not None:
            if kwargs.get("reset_kf", False):
                self.reset_ema()
            tokens_np  = embodied_action_tokens.float().cpu().numpy()  # (B, n_tok, H)
            y_raw      = tokens_np.mean(axis=1)                        # (B, H) mean-pool
            y_filtered = np.stack([self._ema_step(y_raw[b]) for b in range(tokens_np.shape[0])], axis=0)
            correction = torch.from_numpy(y_filtered - y_raw).to(embodied_action_tokens)
            embodied_action_tokens = embodied_action_tokens + correction.unsqueeze(1)

        state = torch.from_numpy(np.array(state)).to(last_hidden.device, dtype=last_hidden.dtype) if state is not None else None
        with torch.autocast("cuda", dtype=torch.float32):
            if cond_t_inf is not None:
                dit_input = torch.cat([embodied_action_tokens, cond_t_inf], dim=1)
            else:
                dit_input = embodied_action_tokens
            pred_actions = self.action_model.predict_action(dit_input, state)

        normalized_actions = pred_actions.detach().cpu().numpy()
        return {"normalized_actions": normalized_actions, "embodied_action_tokens": embodied_action_tokens.to(dtype=torch.float32).detach().cpu().numpy()}



if __name__ == "__main__":
    from omegaconf import OmegaConf
    import debugpy
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_yaml", type=str, default="./starVLA/config/training/starvla_cotrain_oxe.yaml", help="Path to YAML config")
    args, clipargs = parser.parse_known_args()

    debugpy.listen(("0.0.0.0", 10092))
    print("🔍 Rank 0 waiting for debugger attach on port 10092...")
    debugpy.wait_for_client()

    cfg = OmegaConf.load(args.config_yaml)
    # try get model
    cfg.framework.qwenvl.base_vlm = "./playground/Pretrained_models/Qwen3-VL-4B-Instruct"
     
    model: Qwen_GR00T = Qwen_GR00T(cfg)
    print(model)



    # fake sample 
    image = Image.fromarray(np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8))
    # Create a sample
    sample = {
        "action": np.random.uniform(-1, 1, size=(16, 7)).astype(np.float16), # action_chunk, action_dim
        "image": [image, image], # two views
        "lang": "This is a fake for testing.",
        "state" : np.random.uniform(-1, 1, size=(1, 7)).astype(np.float16), # chunk, state_dim
    }

    batch  = [sample, sample]  # batch size 2
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    forward_output = model(batch)
    action_loss = forward_output['action_loss']
    print(f"Action Loss: {action_loss.item()}")

    # test predict action
    predict_output = model.predict_action(batch_images=[batch[0]["image"]], instructions=[batch[0]["lang"]], state=[batch[0]["state"]])
    normalized_actions = predict_output['normalized_actions']
    print(f"Unnormalized Action: {normalized_actions}")

    # # Advance: try forward model with dataloader
    # # can be fake sample， but here get from dataloader for simpler
    # from starVLA.dataloader.lerobot_datasets import get_vla_dataset, collate_fn

    # vla_dataset_cfg = cfg.datasets.vla_data
    # dataset = get_vla_dataset(data_cfg=vla_dataset_cfg)

    # from torch.utils.data import DataLoader

    # train_dataloader = DataLoader(
    #     dataset,
    #     batch_size=2,
    #     num_workers=1,  # For Debug
    #     collate_fn=collate_fn,
    # )
    # # 
    # for batch in tqdm(train_dataloader, desc="Processing Batches"):
    #     batch
    #     break

    # # try get model
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # model = model.to(device)
    # model(batch)

    # action = model.predict_action(batch_images=[batch[0]["image"]], instructions=[batch[0]["lang"]])

    # # fake state
    # for ba in batch:
    #     ba["state"] = ba["action"][0][None]

    # model(batch)
    # action = model.predict_action(batch_images=[batch[0]["image"]], instructions=[batch[0]["lang"]], state=[batch[0]["state"]])
