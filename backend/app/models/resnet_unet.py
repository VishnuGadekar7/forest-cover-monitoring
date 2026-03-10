"""
ResNet U-Net Architecture
===========================
Constructs the skeleton of a ResNet-34 U-Net. 
Pretrained weights are injected by the ModelLoader, so we initialize 
this skeleton without ImageNet weights to save bandwidth and startup time.
"""

try:
    import segmentation_models_pytorch as smp
except ImportError as e:
    raise ImportError(
        "segmentation-models-pytorch is required for ResNetUNet. "
        "Install it with: pip install segmentation-models-pytorch"
    ) from e

import torch.nn as nn


def build_model(
    in_channels: int = 4,        # FIXED: Set to 4 for Sentinel-2 (RGB + NIR)
    out_channels: int = 1,
    encoder_name: str = "resnet34",
    encoder_weights: str = None, # FIXED: Prevent downloading redundant ImageNet weights
) -> nn.Module:
    """
    Build an empty U-Net with a ResNet encoder skeleton.

    Args:
        in_channels:      Number of input image bands (4 for RGB + NIR).
        out_channels:     1 for binary forest / non-forest mask.
        encoder_name:     The ResNet variant used during training (resnet34).
        encoder_weights:  Set to None. The ModelLoader will inject the trained .pth weights.

    Returns:
        nn.Module ready to receive the trained state_dict.
    """
    model = smp.Unet(
        encoder_name=encoder_name,
        encoder_weights=encoder_weights, 
        in_channels=in_channels,
        classes=out_channels,
        activation=None,   # Raw logits — sigmoid applied during inference
    )
    return model