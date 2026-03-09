"""
Image Preprocessing Utilities
================================
Handles loading, validation, normalisation, and tensor conversion for
satellite images before they enter the model inference pipeline.

Design decisions:
  - Target size 512×512 matches model training resolution.
  - ImageNet mean/std normalisation works well for RGB satellite imagery
    (RGB bands have similar statistical distributions to natural images).
  - Original size is preserved so masks can be rescaled back for display.
"""

import io
import numpy as np
from PIL import Image
import torch
import torchvision.transforms.functional as TF
from torchvision import transforms
from typing import Tuple

# ── Constants ─────────────────────────────────────────────────────────────────

TARGET_SIZE = (512, 512)

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

_VALID_EXTENSIONS = {".png", ".jpg", ".jpeg", ".tif", ".tiff"}

_preprocess_transform = transforms.Compose([
    transforms.Resize(TARGET_SIZE, interpolation=Image.BILINEAR),
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
])


# ── Public API ────────────────────────────────────────────────────────────────

def validate_image_bytes(data: bytes, filename: str) -> None:
    """
    Raise ValueError if the uploaded file is not a supported image format.
    """
    suffix = "." + filename.split(".")[-1].lower() if "." in filename else ""
    if suffix not in _VALID_EXTENSIONS:
        raise ValueError(
            f"Unsupported file type '{suffix}'. "
            f"Accepted formats: {sorted(_VALID_EXTENSIONS)}"
        )
    try:
        img = Image.open(io.BytesIO(data))
        img.verify()          # Check file integrity
    except Exception as exc:
        try:
            import tifffile
            _ = tifffile.imread(io.BytesIO(data))
        except Exception:
            raise ValueError(f"Cannot open image '{filename}': {exc}") from exc


def load_image(data: bytes) -> np.ndarray:
    """Load raw bytes → Numpy Array (H, W, C). Supports RGB and multi-band TIFFs."""
    try:
        # Standard PIL loading
        img = Image.open(io.BytesIO(data))
        # Support RGB (3) and RGBA or similar 4-band modes
        arr = np.array(img)
        if arr.ndim == 3 and arr.shape[-1] in [3, 4]:
            return arr
        return np.array(img.convert("RGB"))
    except Exception:
        try:
            import tifffile
            arr = tifffile.imread(io.BytesIO(data))
            
            # Normalize common Earth Observation bit depths to uint8
            if arr.dtype == np.uint16:
                arr = (np.clip(arr / 4000.0, 0, 1) * 255.0).astype(np.uint8)
            elif arr.dtype in [np.float32, np.float64]:
                if arr.max() <= 1.0:
                    arr = (arr * 255.0).astype(np.uint8)
                else:
                    arr = np.clip(arr / 10000.0, 0, 1.0)
                    arr = (arr * 255.0).astype(np.uint8)
            
            # Standardize channels directly
            if arr.ndim == 2:
                arr = np.stack([arr]*3, axis=-1)
            elif arr.ndim == 3 and arr.shape[0] in [3, 4, 12, 13]: # Bands first
                arr = np.transpose(arr, (1, 2, 0))
                
            if arr.ndim == 3 and arr.shape[-1] > 4:
                arr = arr[:, :, :4] # Take at most 4 bands (RGB + NIR)
            
            return arr
        except Exception as e2:
            raise ValueError(f"Image could not be parsed: {e2}")


def preprocess(image_data: np.ndarray) -> Tuple[torch.Tensor, Tuple[int, int]]:
    """
    Resize + normalise a numpy array (H, W, C) into a model-ready float32 tensor.
    If 3 channels are provided, it pads with a 4th zero channel to match the model.

    Returns:
        tensor        : shape (1, 4, 512, 512)
        original_size : (width, height)
    """
    h, w = image_data.shape[:2]
    original_size = (w, h)
    
    # Convert to PIL for easy resizing
    img = Image.fromarray(image_data)
    img = img.resize(TARGET_SIZE, Image.BILINEAR)
    arr = np.array(img).astype(np.float32) / 255.0
    
    # Handle padding for 3 -> 4 channels
    if arr.shape[-1] == 3:
        padding = np.zeros((TARGET_SIZE[0], TARGET_SIZE[1], 1), dtype=np.float32)
        arr = np.concatenate([arr, padding], axis=-1)
    
    # Normalise first 3 channels with ImageNet mean/std
    # (NIR normalization is usually content-dependent, here we keep it as-is [0,1])
    arr[:, :, :3] = (arr[:, :, :3] - IMAGENET_MEAN) / IMAGENET_STD
    
    # To Tensor: (H, W, 4) -> (4, H, W) -> (1, 4, H, W)
    tensor = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)
    return tensor, original_size


def postprocess_mask(
    raw_logits: torch.Tensor,
    original_size: Tuple[int, int],
    threshold: float = 0.5,
) -> np.ndarray:
    """
    Convert raw model output logits to a binary uint8 mask, resized to
    the original image dimensions.

    Args:
        raw_logits   : tensor shape (1, 1, H, W) of unbounded logits
        original_size: (width, height) to resize mask back to
        threshold    : sigmoid probability cut-off for forest class

    Returns:
        Binary numpy array of shape (H, W) with values {0, 1}.
    """
    with torch.no_grad():
        probs = torch.sigmoid(raw_logits)           # (1, 1, H, W)
        binary = (probs > threshold).float()        # (1, 1, H, W)
        # Resize back to original (W, H)
        binary = TF.resize(
            binary[0],                              # (1, H, W)
            size=[original_size[1], original_size[0]],
            interpolation=Image.NEAREST,
        )
        mask = binary[0].cpu().numpy().astype(np.uint8)  # (H, W)
    return mask
