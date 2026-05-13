import torch
import torch.nn as nn
import torch.nn.functional as F


class LearnedGatingFusion(nn.Module):
    """
    Fuse V-JEPA observed tokens (vj_obs) and predicted tokens (vj_pred) via
    learned per-dimension gating.

    Gate input:  [obs ; pred ; cos_sim]  (optionally with cosine similarity)
    Gate output: alpha per dimension in (0, 1)
    Fused:       alpha * obs + (1 - alpha) * pred

    Cosine similarity acts as an explicit reliability signal: when prediction
    diverges from observation (low cos_sim), the gate learns to weight obs more.

    Initialization: zero bias -> sigmoid(0) = 0.5 -> equal mix at episode start.
    """

    def __init__(self, embed_dim: int, use_cosine: bool = True):
        super().__init__()
        self.use_cosine = use_cosine
        gate_in_dim = embed_dim * 2 + (1 if use_cosine else 0)
        self.gate = nn.Linear(gate_in_dim, embed_dim)
        nn.init.zeros_(self.gate.bias)
        nn.init.normal_(self.gate.weight, std=0.02)

    def forward(self, obs: torch.Tensor, pred: torch.Tensor) -> torch.Tensor:
        """
        Args:
            obs:  (B, n_tokens, D) — vj_encoder output for current frame
            pred: (B, n_tokens, D) — vj_predictor output from previous step
        Returns:
            fused: (B, n_tokens, D)
        """
        if self.use_cosine:
            cos = F.cosine_similarity(obs, pred, dim=-1).unsqueeze(-1)  # (B, n_tokens, 1)
            gate_in = torch.cat([obs, pred, cos], dim=-1)
        else:
            gate_in = torch.cat([obs, pred], dim=-1)
        alpha = torch.sigmoid(self.gate(gate_in))  # (B, n_tokens, D)
        return alpha * obs + (1.0 - alpha) * pred
