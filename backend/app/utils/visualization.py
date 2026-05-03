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

def save_image(img, suffix: str, task_id: str) -> tuple[str, str]:
    """
    Save a PIL image (or numpy array) to the static/change_maps/<task_id>/<suffix>.png

    Accepts either a PIL.Image.Image or a numpy ndarray (H, W, C).
    4-channel arrays (RGBA / RGB+NIR) are converted to RGB before saving.

    Returns:
        (filename, relative_url) — e.g. ("change.png", "/static/change_maps/abc123/change.png")
    """
    # Coerce numpy arrays to PIL
    if isinstance(img, np.ndarray):
        if img.ndim == 3 and img.shape[2] == 4:
            img = Image.fromarray(img[:, :, :3])   # drop 4th channel (NIR)
        else:
            img = Image.fromarray(img)
    # Ensure RGB mode (no alpha) for clean PNG output
    if img.mode not in ("RGB", "L"):
        img = img.convert("RGB")
    # Create the unique folder for this specific prediction
    task_dir = STATIC_DIR / task_id
    task_dir.mkdir(parents=True, exist_ok=True)
    
    filename = f"{suffix}.png"
    path = task_dir / filename
    img.save(str(path), format="PNG", optimize=True)
    return filename, f"/static/change_maps/{task_id}/{filename}"


def save_mask_as_image(mask: np.ndarray, suffix: str, task_id: str) -> tuple[str, str]:
    """
    Convert a binary (0/1) uint8 mask into a greyscale PIL image and save it.
    Forest pixels become white (255), non-forest black (0).
    """
    visual = (mask * 255).astype(np.uint8)
    img = Image.fromarray(visual, mode="L").convert("RGB")
    return save_image(img, suffix, task_id)


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
