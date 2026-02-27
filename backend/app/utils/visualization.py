"""
Visualization Utilities
=========================
Saves forest masks and change maps to disk and generates overlay images
for display in the frontend.
"""

import uuid
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw, ImageFont

# ── Paths ─────────────────────────────────────────────────────────────────────

STATIC_DIR = Path(__file__).resolve().parents[2] / "static" / "change_maps"
STATIC_DIR.mkdir(parents=True, exist_ok=True)


# ── Image Saving ──────────────────────────────────────────────────────────────

def save_image(img: Image.Image, suffix: str) -> tuple[str, str]:
    """
    Save a PIL image to the static/change_maps directory.

    Returns:
        (filename, relative_url) — e.g. ("abc123_change.png", "/static/change_maps/abc123_change.png")
    """
    base_id = uuid.uuid4().hex[:12]
    filename = f"{base_id}_{suffix}.png"
    path = STATIC_DIR / filename
    img.save(str(path), format="PNG", optimize=True)
    return filename, f"/static/change_maps/{filename}"


def save_mask_as_image(mask: np.ndarray, suffix: str) -> tuple[str, str]:
    """
    Convert a binary (0/1) uint8 mask into a greyscale PIL image and save it.
    Forest pixels become white (255), non-forest black (0).
    """
    visual = (mask * 255).astype(np.uint8)
    img = Image.fromarray(visual, mode="L").convert("RGB")
    return save_image(img, suffix)


def generate_mask_overlay(
    original: Image.Image,
    mask: np.ndarray,
    alpha: float = 0.45,
    color: tuple[int, int, int] = (40, 167, 69),
) -> Image.Image:
    """
    Blend a coloured semi-transparent mask overlay on top of the original image.

    Args:
        original : RGB PIL Image.
        mask     : Binary (H, W) numpy array — same dims as original.
        alpha    : Overlay opacity (0=transparent, 1=opaque).
        color    : RGB colour for forest pixels.

    Returns:
        Composited PIL Image.
    """
    orig_rgba = original.convert("RGBA")
    H, W = mask.shape
    overlay_arr = np.zeros((H, W, 4), dtype=np.uint8)
    forest_pixels = mask == 1
    overlay_arr[forest_pixels, :3] = color
    overlay_arr[forest_pixels, 3]  = int(255 * alpha)
    overlay = Image.fromarray(overlay_arr, mode="RGBA")
    # Resize overlay to match original if shapes differ
    if overlay.size != orig_rgba.size:
        overlay = overlay.resize(orig_rgba.size, Image.NEAREST)
    composited = Image.alpha_composite(orig_rgba, overlay).convert("RGB")
    return composited
