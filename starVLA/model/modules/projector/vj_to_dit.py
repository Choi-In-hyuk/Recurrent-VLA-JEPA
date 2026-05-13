import torch
import torch.nn as nn


class VJtoDiTProjection(nn.Module):
    """
    Project fused V-JEPA tokens into the DiT conditioning space.

    Reduces 256 spatial tokens → n_cond_tokens via group mean-pooling,
    then maps from vj_dim (2816) to dit_dim (2048) with a Linear + LayerNorm.

    The group pooling preserves spatial structure at a coarser resolution,
    giving the DiT a compact but spatially-aware conditioning signal.
    """

    def __init__(self, vj_dim: int = 2816, dit_dim: int = 2048, n_cond_tokens: int = 8):
        super().__init__()
        self.n_cond = n_cond_tokens
        self.proj = nn.Sequential(
            nn.Linear(vj_dim, dit_dim),
            nn.LayerNorm(dit_dim),
        )

    def forward(self, fused: torch.Tensor) -> torch.Tensor:
        """
        Args:
            fused: (B, N, vj_dim)  e.g. (B, 256, 2816)
        Returns:
            cond:  (B, n_cond_tokens, dit_dim)  e.g. (B, 8, 2048)
        """
        B, N, D = fused.shape
        group = N // self.n_cond
        # Trim to exact multiple, then group mean-pool
        fused = fused[:, : group * self.n_cond, :]              # (B, n_cond*group, D)
        fused = fused.view(B, self.n_cond, group, D).mean(dim=2)  # (B, n_cond, D)
        return self.proj(fused)                                   # (B, n_cond, dit_dim)
