import torch
import torch.nn as nn


class CorrectionProjector(nn.Module):
    """
    Projects Δz = vj_obs - vj_pred into QwenVL embedding space.

    Δz encodes how wrong the world model's prediction was at the previous step.
    These projected embeddings are injected as correction tokens in QwenVL's
    input sequence (between action_tokens and embodied_action_tokens), so that
    embodied_action_tokens attend to this error signal via QwenVL's attention.

    Architecture:
        256 spatial tokens → group mean-pool → n_tokens
        vj_dim (2816) → LayerNorm → Linear → qwen_dim (2048)
        scalar gate (sigmoid(-4) ≈ 0.018 at init) keeps residual near-zero initially
    """

    def __init__(self, vj_dim: int = 2816, qwen_dim: int = 2048, n_tokens: int = 8):
        super().__init__()
        self.n_tokens = n_tokens
        self.proj = nn.Sequential(
            nn.LayerNorm(vj_dim),
            nn.Linear(vj_dim, qwen_dim),
        )
        self.gate = nn.Parameter(torch.tensor(-4.0))
        nn.init.normal_(self.proj[1].weight, std=0.01)
        nn.init.zeros_(self.proj[1].bias)

    def forward(self, delta_z: torch.Tensor) -> torch.Tensor:
        """
        Args:
            delta_z: (B, N_spatial, vj_dim)  e.g. (B, 256, 2816)
        Returns:
            corr_embeds: (B, n_tokens, qwen_dim)  e.g. (B, 8, 2048)
        """
        B, N, D = delta_z.shape
        group = N // self.n_tokens
        delta_z = delta_z[:, : group * self.n_tokens, :]
        delta_z = delta_z.view(B, self.n_tokens, group, D).mean(dim=2)  # (B, n_tokens, D)
        return torch.sigmoid(self.gate) * self.proj(delta_z)             # (B, n_tokens, qwen_dim)
