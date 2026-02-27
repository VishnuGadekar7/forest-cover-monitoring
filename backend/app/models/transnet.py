"""
TransNet — Transformer-based Segmentation Architecture
=======================================================
Implements a lightweight Vision Transformer (ViT) encoder fused with
a convolutional decoder, designed for binary forest segmentation from
satellite imagery.

Architecture overview:
  Image → Patch Embedding → Transformer Encoder → Convolutional Decoder → Mask

Using a hybrid approach (CNN stem + Transformer body) for better spatial
preservation at the cost of full attention-everywhere ViT.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ── Patch Embedding ───────────────────────────────────────────────────────────

class PatchEmbedding(nn.Module):
    """
    Splits image into non-overlapping patches and projects to embedding dim.
    Uses a conv layer as an efficient patch extractor.
    """

    def __init__(
        self,
        img_size: int = 512,
        patch_size: int = 16,
        in_channels: int = 3,
        embed_dim: int = 256,
    ):
        super().__init__()
        self.num_patches = (img_size // patch_size) ** 2
        self.patch_size = patch_size
        self.proj = nn.Conv2d(
            in_channels, embed_dim, kernel_size=patch_size, stride=patch_size
        )
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, H, W) → (B, N, D)
        x = self.proj(x)                           # (B, D, H/P, W/P)
        B, D, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)           # (B, N, D)
        x = self.norm(x)
        return x, H, W


# ── Transformer Encoder ───────────────────────────────────────────────────────

class TransformerEncoderBlock(nn.Module):
    """Standard pre-norm transformer block with multi-head self-attention."""

    def __init__(self, embed_dim: int, num_heads: int, mlp_ratio: float = 4.0, dropout: float = 0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(embed_dim)
        mlp_hidden = int(embed_dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, mlp_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden, embed_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.norm1(x)
        h, _ = self.attn(h, h, h)
        x = x + h
        x = x + self.mlp(self.norm2(x))
        return x


# ── Convolutional Decoder ─────────────────────────────────────────────────────

class ConvDecoder(nn.Module):
    """Progressive upsampling decoder that recovers spatial resolution."""

    def __init__(self, embed_dim: int, out_channels: int):
        super().__init__()
        self.layers = nn.Sequential(
            nn.ConvTranspose2d(embed_dim, 128, kernel_size=2, stride=2),
            nn.BatchNorm2d(128), nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128, 64,  kernel_size=2, stride=2),
            nn.BatchNorm2d(64),  nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64,  32,  kernel_size=2, stride=2),
            nn.BatchNorm2d(32),  nn.ReLU(inplace=True),
            nn.ConvTranspose2d(32,  16,  kernel_size=2, stride=2),
            nn.BatchNorm2d(16),  nn.ReLU(inplace=True),
            nn.Conv2d(16, out_channels, kernel_size=1),
        )

    def forward(self, x: torch.Tensor, H: int, W: int) -> torch.Tensor:
        # (B, N, D) → (B, D, H, W) for convolutional decoder
        B, N, D = x.shape
        x = x.transpose(1, 2).reshape(B, D, H, W)
        return self.layers(x)


# ── TransNet ──────────────────────────────────────────────────────────────────

class TransNet(nn.Module):
    """
    Transformer-based forest segmentation model.

    Args:
        img_size:    Input image size (assumes square).
        patch_size:  Patch size for tokenisation (16 gives 32×32 tokens for 512px).
        in_channels: Number of input bands.
        out_channels: 1 for binary mask.
        embed_dim:   Transformer embedding dimension.
        depth:       Number of transformer encoder blocks.
        num_heads:   Attention heads.
    """

    def __init__(
        self,
        img_size: int = 512,
        patch_size: int = 16,
        in_channels: int = 3,
        out_channels: int = 1,
        embed_dim: int = 256,
        depth: int = 6,
        num_heads: int = 8,
    ):
        super().__init__()
        self.patch_embed = PatchEmbedding(img_size, patch_size, in_channels, embed_dim)
        self.pos_embed = nn.Parameter(
            torch.zeros(1, (img_size // patch_size) ** 2, embed_dim)
        )
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

        self.encoder_blocks = nn.ModuleList([
            TransformerEncoderBlock(embed_dim, num_heads) for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(embed_dim)
        self.decoder = ConvDecoder(embed_dim, out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        tokens, H, W = self.patch_embed(x)
        tokens = tokens + self.pos_embed[:, :tokens.size(1), :]

        for block in self.encoder_blocks:
            tokens = block(tokens)

        tokens = self.norm(tokens)
        return self.decoder(tokens, H, W)


def build_model(
    in_channels: int = 3,
    out_channels: int = 1,
    img_size: int = 512,
) -> TransNet:
    """Factory function — returns a TransNet instance."""
    return TransNet(
        img_size=img_size,
        in_channels=in_channels,
        out_channels=out_channels,
    )
