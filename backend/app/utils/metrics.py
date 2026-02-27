"""
Statistical Metrics — Forest Area & Change Calculations
=========================================================
All area calculations assume a fixed pixel resolution (Sentinel-2 default
is 10 m/pixel, giving 100 m²/pixel = 0.01 ha/pixel).

These functions are pure (no side effects) and easily unit-tested.
"""

import numpy as np
from typing import Dict


# ── Constants ─────────────────────────────────────────────────────────────────

# Default Sentinel-2 ground sampling distance in metres
DEFAULT_PIXEL_RESOLUTION_M: float = 10.0

# Area of one pixel in hectares: (10m × 10m) = 100 m² = 0.01 ha
_PIXEL_AREA_HA: float = (DEFAULT_PIXEL_RESOLUTION_M ** 2) / 10_000


# ── Public API ────────────────────────────────────────────────────────────────

def compute_area(mask: np.ndarray, pixel_resolution_m: float = DEFAULT_PIXEL_RESOLUTION_M) -> float:
    """
    Compute the total area covered by forest pixels (value == 1) in hectares.

    Args:
        mask              : Binary numpy array (H, W) with values {0, 1}.
        pixel_resolution_m: Ground sampling distance in metres/pixel.

    Returns:
        Area in hectares (float, rounded to 4 decimal places).
    """
    pixel_area_ha = (pixel_resolution_m ** 2) / 10_000
    forest_pixels = int(np.sum(mask == 1))
    return round(forest_pixels * pixel_area_ha, 4)


def compute_statistics(
    mask_t1: np.ndarray,
    mask_t2: np.ndarray,
    pixel_resolution_m: float = DEFAULT_PIXEL_RESOLUTION_M,
) -> Dict[str, float]:
    """
    Compute comprehensive forest change statistics between two time periods.

    Args:
        mask_t1           : Binary mask for Time 1.
        mask_t2           : Binary mask for Time 2 (must be same shape as mask_t1).
        pixel_resolution_m: Ground sampling distance in metres/pixel.

    Returns:
        Dictionary with keys:
            forest_area_t1    : ha
            forest_area_t2    : ha
            forest_loss       : ha (area that was forest in T1, non-forest in T2)
            forest_gain       : ha (area that was non-forest in T1, forest in T2)
            percentage_change : % net change relative to T1 (negative = net loss)
    """
    if mask_t1.shape != mask_t2.shape:
        raise ValueError(
            f"Mask shapes must match: got {mask_t1.shape} vs {mask_t2.shape}"
        )

    pixel_area_ha = (pixel_resolution_m ** 2) / 10_000

    # Pixel-wise change maps
    loss_pixels = int(np.sum((mask_t1 == 1) & (mask_t2 == 0)))  # Forest → Non-forest
    gain_pixels = int(np.sum((mask_t1 == 0) & (mask_t2 == 1)))  # Non-forest → Forest

    area_t1   = compute_area(mask_t1, pixel_resolution_m)
    area_t2   = compute_area(mask_t2, pixel_resolution_m)
    loss_ha   = round(loss_pixels * pixel_area_ha, 4)
    gain_ha   = round(gain_pixels * pixel_area_ha, 4)

    if area_t1 > 0:
        pct_change = round(((area_t2 - area_t1) / area_t1) * 100, 2)
    else:
        pct_change = 0.0

    return {
        "forest_area_t1":    area_t1,
        "forest_area_t2":    area_t2,
        "forest_loss":       loss_ha,
        "forest_gain":       gain_ha,
        "percentage_change": pct_change,
    }
