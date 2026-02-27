"""
Change Detection Engine
========================
Performs pixel-wise comparison of two binary forest masks and generates:
  1. A coloured change map PNG (red=loss, green=gain, white=stable, black=non-forest)
  2. Statistical summary via the metrics module

Colour convention follows standard remote sensing change map visualisation:
  Red   (255,   0,   0) — Forest Loss    (T1=1, T2=0)
  Green (  0, 180,   0) — Forest Gain    (T1=0, T2=1)
  White (255, 255, 255) — Stable Forest  (T1=1, T2=1)
  Black (  0,   0,   0) — Non-forest both epochs
"""

import numpy as np
from PIL import Image
from typing import Tuple, Dict


# ── Colour palette ────────────────────────────────────────────────────────────

COLOUR_LOSS          = np.array([220,  50,  47], dtype=np.uint8)   # Red
COLOUR_GAIN          = np.array([ 40, 167,  69], dtype=np.uint8)   # Green
COLOUR_STABLE_FOREST = np.array([255, 255, 255], dtype=np.uint8)   # White
COLOUR_NON_FOREST    = np.array([ 20,  20,  30], dtype=np.uint8)   # Near-black


# ── Public API ────────────────────────────────────────────────────────────────

def generate_change_map(
    mask_t1: np.ndarray,
    mask_t2: np.ndarray,
) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
    """
    Build an RGB change-map array from two binary masks.

    Args:
        mask_t1: Binary (H, W) uint8 array — T1 forest mask.
        mask_t2: Binary (H, W) uint8 array — T2 forest mask.

    Returns:
        change_rgb : (H, W, 3) uint8 array — coloured change map.
        class_masks: dict of boolean masks for each class.
    """
    if mask_t1.shape != mask_t2.shape:
        raise ValueError(
            f"Mask shapes must match: {mask_t1.shape} vs {mask_t2.shape}"
        )

    H, W = mask_t1.shape
    change_rgb = np.zeros((H, W, 3), dtype=np.uint8)

    # Compute boolean class masks
    loss_mask          = (mask_t1 == 1) & (mask_t2 == 0)
    gain_mask          = (mask_t1 == 0) & (mask_t2 == 1)
    stable_forest_mask = (mask_t1 == 1) & (mask_t2 == 1)
    # non_forest_mask  — leave as black (zeros)

    change_rgb[loss_mask]          = COLOUR_LOSS
    change_rgb[gain_mask]          = COLOUR_GAIN
    change_rgb[stable_forest_mask] = COLOUR_STABLE_FOREST

    class_masks = {
        "loss":         loss_mask,
        "gain":         gain_mask,
        "stable_forest": stable_forest_mask,
    }
    return change_rgb, class_masks


def change_map_to_pil(change_rgb: np.ndarray) -> Image.Image:
    """Convert the (H, W, 3) uint8 change map array to a PIL Image."""
    return Image.fromarray(change_rgb, mode="RGB")
