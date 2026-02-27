"""
ResNet U-Net Architecture
===========================
U-Net with a pretrained ResNet-34 encoder via segmentation-models-pytorch.
The pretrained ImageNet encoder provides strong feature extraction out of the
box, reducing the amount of labelled satellite data needed for fine-tuning.

This model is instantiated with the same public API (build_model) as the
other architectures so the model_loader can swap them without code changes.
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
    in_channels: int = 3,
    out_channels: int = 1,
    encoder_name: str = "resnet34",
    encoder_weights: str = "imagenet",
) -> nn.Module:
    """
    Build a U-Net with a ResNet encoder.

    Args:
        in_channels:      Number of input image bands (3 for RGB).
        out_channels:     1 for binary forest / non-forest mask.
        encoder_name:     Any ResNet variant supported by smp (resnet18/34/50…).
        encoder_weights:  'imagenet' for transfer learning, None for random init.

    Returns:
        nn.Module ready for training or inference.
    """
    model = smp.Unet(
        encoder_name=encoder_name,
        encoder_weights=encoder_weights,
        in_channels=in_channels,
        classes=out_channels,
        activation=None,   # Raw logits — sigmoid applied during inference
    )
    return model
