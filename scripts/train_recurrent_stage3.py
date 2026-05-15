"""
Stage 3 training: QwenVL correction-token injection.

Architecture:
  - correction_projector:  vj_dim → qwen_dim  (TRAINABLE)
  - QwenVL:                frozen
  - V-JEPA encoder/pred:   frozen
  - DiT:                   frozen

For each sample:
  1. Δz = vj_obs - vj_pred  (from pre-extracted .pt files)
  2. CorrectionProjector(Δz) → correction_embeds  (B, N, qwen_dim)
  3. Inject correction_embeds into QwenVL input between action_tokens and embodied_action_tokens
  4. Run QwenVL (frozen) → embodied_action_tokens that attended to correction
  5. Run DiT (frozen) → action loss
  6. Backprop through frozen QwenVL → CorrectionProjector

Training data requires the updated extraction (with instruction + hdf5_path metadata).
Re-run scripts/extract_recurrent_tokens.py if your .pt files lack these fields.

Usage:
  cd /home/choi/Recurrent-VLA-JEPA
  python scripts/train_recurrent_stage3.py [--epochs 30] [--batch_size 4] [--lr 1e-4]
"""

import argparse
import os
import sys

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.insert(0, "/home/choi/Recurrent-VLA-JEPA")

from starVLA.dataloader.stage3_dataset import Stage3Dataset, stage3_collate_fn

CKPT_PATH  = "/media/choi/8AA890DCA890C859/vjepa2_baseline/checkpoints/VLA-JEPA/LIBERO/checkpoints/VLA-JEPA-LIBERO.pt"
TOKEN_ROOT = "/media/choi/8AA890DCA890C859/vjepa2_baseline/datasets/recurrent_jepa_tokens"
SAVE_DIR   = "checkpoints/recurrent_stage3"

DATA_DIRS = [
    os.path.join(TOKEN_ROOT, "libero_spatial"),
    os.path.join(TOKEN_ROOT, "libero_10"),
    os.path.join(TOKEN_ROOT, "libero_goal"),
    os.path.join(TOKEN_ROOT, "libero_object"),
]

N_CORRECTION_TOKENS = 8


def load_model(ckpt_path, device, use_lora=True, lora_r=16, lora_alpha=32):
    from starVLA.model.framework.base_framework import baseframework
    model = baseframework.from_pretrained(ckpt_path)
    model = model.to(device)

    # Enable Stage 3 (registers correction tokens + builds CorrectionProjector + optional LoRA)
    model.load_stage3(
        n_correction_tokens=N_CORRECTION_TOKENS,
        use_lora=use_lora,
        lora_r=lora_r,
        lora_alpha=lora_alpha,
    )
    model.correction_projector = model.correction_projector.to(device)

    # Freeze everything; unfreeze correction_projector and LoRA params
    for name, param in model.named_parameters():
        param.requires_grad = (
            name.startswith("correction_projector.") or
            "lora_" in name
        )

    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_total = sum(p.numel() for p in model.parameters())
    print(f"  Trainable: {n_train:,} / {n_total:,} params  (lora={use_lora})")
    return model


def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    model = load_model(
        args.ckpt_path, device,
        use_lora=not args.no_lora,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
    )

    data_dirs = [d for d in DATA_DIRS if os.path.isdir(d)]
    if not data_dirs:
        sys.exit(f"No token directories found under {TOKEN_ROOT}.")

    train_ds = Stage3Dataset(data_dirs, split="train", seed=42)
    val_ds   = Stage3Dataset(data_dirs, split="val",   seed=42)

    if len(train_ds) == 0:
        sys.exit(
            "No samples with Stage 3 metadata found.\n"
            "Re-run scripts/extract_recurrent_tokens.py to regenerate .pt files with metadata."
        )

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=2, pin_memory=False, collate_fn=stage3_collate_fn,
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=1, pin_memory=False, collate_fn=stage3_collate_fn,
    )

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(
        trainable_params,
        lr=args.lr, betas=(0.9, 0.95), eps=1e-8, weight_decay=1e-8,
    )

    os.makedirs(args.save_dir, exist_ok=True)
    repeated    = model.config.framework.action_model.repeated_diffusion_steps
    global_step = 0
    best_val    = float("inf")

    for epoch in range(args.epochs):
        model.train()
        model.correction_projector.train()
        total_loss = n_batches = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}", dynamic_ncols=True)
        for batch in pbar:
            vj_obs     = batch["vj_obs"].to(device)     # (B, 256, 2816)
            vj_pred    = batch["vj_pred"].to(device)    # (B, 256, 2816)
            actions    = batch["actions"].to(device)    # (B, horizon, 7)
            states     = batch["states"].to(device)     # (B, 8)
            images     = batch["images"]                # list of [PIL, PIL] per sample
            instructions = batch["instructions"]        # list of str

            with torch.autocast("cuda", dtype=torch.bfloat16):
                delta_z = (vj_obs - vj_pred).float()  # (B, 256, 2816)

                # Run QwenVL with correction injection → last_hidden
                last_hidden = model._run_qwen_with_correction(images, instructions, delta_z)
                # (B, L, qwen_dim)

                # Extract embodied_action_tokens
                qwen_inputs_tmp = model.qwen_vl_interface.build_qwenvl_inputs(
                    images=images,
                    instructions=instructions,
                    prompt_replace_dict={
                        "{actions}":   model.replace_prompt + model.correction_replace_prompt,
                        "{e_actions}": model.embodied_replace_prompt,
                    },
                    prompt_template=model.config.datasets.vla_data.get("CoT_prompt", ""),
                )
                input_ids_tmp = qwen_inputs_tmp["input_ids"]
                emb_id_t = torch.tensor([model.embodied_action_token_id], device=device)
                emb_mask = torch.isin(input_ids_tmp, emb_id_t).nonzero(as_tuple=True)
                B, _, H  = last_hidden.shape
                emb_tokens = last_hidden[emb_mask[0], emb_mask[1], :].view(B, -1, H)

                # DiT loss (frozen DiT, just compute loss)
                emb_rep     = emb_tokens.repeat(repeated, 1, 1)
                actions_rep = actions.repeat(repeated, 1, 1)
                states_rep  = states.repeat(repeated, 1)

                loss = model.action_model(emb_rep, actions_rep, states_rep)

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(trainable_params, 1.0)
            optimizer.step()

            total_loss  += loss.item()
            n_batches   += 1
            global_step += 1
            pbar.set_postfix(loss=f"{loss.item():.4f}", step=global_step)

            if global_step % args.log_freq == 0:
                tqdm.write(f"  step={global_step:5d}  loss={loss.item():.4f}")

            if global_step >= args.max_steps:
                break

        avg_loss = total_loss / max(n_batches, 1)
        print(f"Epoch {epoch+1:3d}  train_loss={avg_loss:.4f}")

        # Validation
        model.eval()
        model.correction_projector.eval()
        val_loss = val_n = 0
        with torch.no_grad():
            for batch in val_loader:
                vj_obs     = batch["vj_obs"].to(device)
                vj_pred    = batch["vj_pred"].to(device)
                actions    = batch["actions"].to(device)
                states     = batch["states"].to(device)
                images     = batch["images"]
                instructions = batch["instructions"]
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    delta_z = (vj_obs - vj_pred).float()
                    last_hidden = model._run_qwen_with_correction(images, instructions, delta_z)
                    qwen_inputs_tmp = model.qwen_vl_interface.build_qwenvl_inputs(
                        images=images, instructions=instructions,
                        prompt_replace_dict={
                            "{actions}":   model.replace_prompt + model.correction_replace_prompt,
                            "{e_actions}": model.embodied_replace_prompt,
                        },
                        prompt_template=model.config.datasets.vla_data.get("CoT_prompt", ""),
                    )
                    input_ids_tmp = qwen_inputs_tmp["input_ids"]
                    emb_id_t = torch.tensor([model.embodied_action_token_id], device=device)
                    emb_mask = torch.isin(input_ids_tmp, emb_id_t).nonzero(as_tuple=True)
                    B, _, H  = last_hidden.shape
                    emb_tokens = last_hidden[emb_mask[0], emb_mask[1], :].view(B, -1, H)
                    loss = model.action_model(emb_tokens, actions, states)
                val_loss += loss.item()
                val_n    += 1

        avg_val = val_loss / max(val_n, 1)
        print(f"           val_loss={avg_val:.4f}")

        # Save projector + LoRA weights
        ckpt = {
            "correction_projector": model.correction_projector.state_dict(),
            "n_correction_tokens":  N_CORRECTION_TOKENS,
            "epoch":                epoch + 1,
            "val_loss":             avg_val,
            "use_lora":             not args.no_lora,
            "lora_r":               args.lora_r,
            "lora_alpha":           args.lora_alpha,
        }
        if not args.no_lora and getattr(model, "_stage3_lora", False):
            from peft import get_peft_model_state_dict
            ckpt["lora_state_dict"] = get_peft_model_state_dict(model.qwen_vl_interface.model)
        torch.save(ckpt, os.path.join(args.save_dir, f"epoch{epoch+1:03d}.pt"))
        if avg_val < best_val:
            best_val = avg_val
            torch.save(ckpt, os.path.join(args.save_dir, "best.pt"))
            print(f"           *** best val={best_val:.4f} → best.pt")

        if global_step >= args.max_steps:
            break

    print(f"\nDone. Checkpoints → {args.save_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt_path",  default=CKPT_PATH)
    parser.add_argument("--save_dir",   default=SAVE_DIR)
    parser.add_argument("--epochs",     type=int,   default=30)
    parser.add_argument("--max_steps",  type=int,   default=10000)
    parser.add_argument("--batch_size", type=int,   default=4)
    parser.add_argument("--lr",         type=float, default=1e-4)
    parser.add_argument("--log_freq",   type=int,   default=10)
    parser.add_argument("--no_lora",    action="store_true", help="Disable LoRA (train CorrectionProjector only)")
    parser.add_argument("--lora_r",     type=int,   default=16)
    parser.add_argument("--lora_alpha", type=int,   default=32)
    args = parser.parse_args()
    train(args)
