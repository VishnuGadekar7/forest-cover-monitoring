"""
Trans U-Net Architecture
===========================
U-Net with a Mix Vision Transformer (MiT) encoder.
This acts as a hybrid Transformer-CNN architecture for semantic segmentation.
"""

try:
    import segmentation_models_pytorch as smp
except ImportError as e:
    raise ImportError(
        "segmentation-models-pytorch is required. "
        "Install it with: pip install segmentation-models-pytorch"
    ) from e

import torch.nn as nn

def build_model(
    in_channels: int = 4,        # Your ModelLoader passes 4 here for Sentinel-2
    out_channels: int = 1,
    encoder_name: str = "mit_b1",
    encoder_weights: str = None,
) -> nn.Module:
    """
    Builds the Trans U-Net skeleton with a 4-channel bypass.
    """
    
    # Hardcode in_channels=3 right here to satisfy the library
    model = smp.Unet(
        encoder_name=encoder_name,
        encoder_weights=encoder_weights,
        in_channels=3,
        classes=out_channels,
        activation=None,
    )

    # Now we rip out the 3-channel layer and replace it with 4
    if in_channels != 3:
        old_conv = model.encoder.patch_embed1.proj
        
        new_conv = nn.Conv2d(
            in_channels=in_channels,	# 4 channels
            out_channels=old_conv.out_channels,
            kernel_size=old_conv.kernel_size,
            stride=old_conv.stride,
            padding=old_conv.padding,
            bias=(old_conv.bias is not None)
        )
        
        model.encoder.patch_embed1.proj = new_conv

    return model