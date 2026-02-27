"""
Attention U-Net Architecture
==============================
Standard U-Net augmented with Attention Gates before each skip connection.
Attention gates suppress irrelevant background activations, which substantially
improves segmentation accuracy for small objects (e.g. forest patches at the
boundary of deforestation fronts).

Reference: Oktay et al. 2018 — "Attention U-Net: Learning Where to Look for
           the Pancreas" (https://arxiv.org/abs/1804.03999)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ── Building Blocks ───────────────────────────────────────────────────────────

class ConvBlock(nn.Module):
    """Two consecutive (Conv → BN → ReLU) layers."""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class AttentionGate(nn.Module):
    """
    Soft attention gate that learns to focus on salient regions.
    Takes gating signal g (from decoder) and skip connection x (from encoder).
    """

    def __init__(self, F_g: int, F_l: int, F_int: int):
        super().__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1, bias=True),
            nn.BatchNorm2d(F_int),
        )
        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1, bias=True),
            nn.BatchNorm2d(F_int),
        )
        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid(),
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, g: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        return x * psi


class UpConv(nn.Module):
    """Transposed convolution for upsampling."""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.up(x)


# ── Attention U-Net ───────────────────────────────────────────────────────────

class AttentionUNet(nn.Module):
    """
    Attention U-Net for binary forest segmentation.

    Args:
        in_channels:  Number of input image channels (3 for RGB).
        out_channels: Number of output channels (1 for binary mask).
        features:     Channel sizes at each encoder level.
    """

    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 1,
        features: list[int] = [64, 128, 256, 512],
    ):
        super().__init__()
        self.encoders = nn.ModuleList()
        self.pools = nn.ModuleList()
        self.decoders = nn.ModuleList()
        self.ups = nn.ModuleList()
        self.attentions = nn.ModuleList()

        # Encoder path
        ch = in_channels
        for feat in features:
            self.encoders.append(ConvBlock(ch, feat))
            self.pools.append(nn.MaxPool2d(2))
            ch = feat

        # Bottleneck
        self.bottleneck = ConvBlock(features[-1], features[-1] * 2)

        # Decoder path
        for feat in reversed(features):
            self.ups.append(UpConv(feat * 2, feat))
            self.attentions.append(
                AttentionGate(F_g=feat, F_l=feat, F_int=feat // 2)
            )
            self.decoders.append(ConvBlock(feat * 2, feat))

        # Final 1×1 conv → binary logit
        self.final = nn.Conv2d(features[0], out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        skip_connections = []

        # Encoder
        for enc, pool in zip(self.encoders, self.pools):
            x = enc(x)
            skip_connections.append(x)
            x = pool(x)

        x = self.bottleneck(x)
        skip_connections = skip_connections[::-1]

        # Decoder
        for up, att, dec, skip in zip(
            self.ups, self.attentions, self.decoders, skip_connections
        ):
            x = up(x)
            skip = att(g=x, x=skip)
            # Handle odd spatial dims
            if x.shape != skip.shape:
                x = F.interpolate(x, size=skip.shape[2:])
            x = torch.cat([skip, x], dim=1)
            x = dec(x)

        return self.final(x)


def build_model(in_channels: int = 3, out_channels: int = 1) -> AttentionUNet:
    """Factory function — returns an Attention U-Net instance."""
    return AttentionUNet(in_channels=in_channels, out_channels=out_channels)
