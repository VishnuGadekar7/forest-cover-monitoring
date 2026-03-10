"""
Detection Route — POST /api/v1/detect-change
==============================================
Orchestrates the complete forest change detection pipeline:
  1. Accept & validate two uploaded satellite images (JPG/PNG/TIFF)
  2. Load images and run inference on both
  3. Generate change map and statistics
  4. Save output images to static directory
  5. Return structured JSON response
"""

import io
import logging
import numpy as np
from rasterio.io import MemoryFile
from fastapi import APIRouter, File, UploadFile, HTTPException, status
from PIL import Image

from app.schemas.detection import ChangeDetectionResponse
from app.schemas.stac_query import STACQueryRequest
from app.services.inference_service import InferenceService
from app.services.change_detection import generate_change_map, change_map_to_pil
from app.utils.image_preprocessing import validate_image_bytes
from app.utils.metrics import compute_statistics
from app.utils.visualization import save_image, save_mask_as_image

logger = logging.getLogger(__name__)
router = APIRouter()

_inference = InferenceService()

def load_image_to_numpy(file_bytes: bytes, filename: str) -> np.ndarray:
    """
    Reads image bytes and returns an (H, W, C) numpy array.
    Dynamically routes to Rasterio for GeoTIFFs and PIL for standard images.
    """
    filename_lower = filename.lower()
    
    # --- Route 1: Geospatial TIFFs ---
    if filename_lower.endswith(('.tif', '.tiff')):
        try:
            with MemoryFile(file_bytes) as memfile:
                with memfile.open() as src:
                    # Read all bands; Rasterio outputs (Channels, Height, Width)
                    arr = src.read()
                    # Transpose to (Height, Width, Channels) 
                    return np.transpose(arr, (1, 2, 0))
        except Exception as e:
            raise ValueError(f"Failed to read GeoTIFF: {e}")
    # --- Route 2: Standard Images (JPG, PNG) ---
    else:
        try:
            # Load with PIL and ensure it has 3 channels (RGB)
            image = Image.open(io.BytesIO(file_bytes)).convert("RGB")
            arr = np.array(image)
            
            # Pad with a 4th dummy channel (zeros) so the 4-channel model doesn't crash
            dummy_nir = np.zeros((arr.shape[0], arr.shape[1], 1), dtype=arr.dtype)
            arr = np.concatenate([arr, dummy_nir], axis=-1)
            
            return arr
        except Exception as e:
            raise ValueError(f"Failed to read standard image: {e}")


@router.post(
    "/detect-change",
    response_model=ChangeDetectionResponse,
)
async def detect_change(
    image_t1: UploadFile = File(...),
    image_t2: UploadFile = File(...),
):
    # ── 1. Read Bytes ─────────────────────────────────────────────────────────
    t1_bytes = await image_t1.read()
    t2_bytes = await image_t2.read()

    # ── 2. Load Arrays (Handles both TIF and JPG/PNG dynamically) ─────────────
    try:
        arr_t1 = load_image_to_numpy(t1_bytes, image_t1.filename or "unknown.jpg")
        arr_t2 = load_image_to_numpy(t2_bytes, image_t2.filename or "unknown.jpg")
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(exc),
        )

    # ── 3. Inference — pass numpy arrays ──────────────────────────────────────
    try:
        logger.info("Running inference on T1...")
        mask_t1 = _inference.predict(arr_t1)
        logger.info("Running inference on T2...")
        mask_t2 = _inference.predict(arr_t2)
    except Exception as exc:
        logger.exception("Inference failed")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Inference error using {model_name}: {exc}",
        )

    # ── 4. Change Detection & Stats ───────────────────────────────────────────
    change_rgb, _ = generate_change_map(mask_t1, mask_t2)
    change_pil    = change_map_to_pil(change_rgb)
    stats = compute_statistics(mask_t1, mask_t2)

    # ── 5. Save Output Images ─────────────────────────────────────────────────
    _, change_map_url = save_image(change_pil, "change")
    _, mask_t1_url    = save_mask_as_image(mask_t1, "mask_t1")
    _, mask_t2_url    = save_mask_as_image(mask_t2, "mask_t2")
    _, image_t1_url   = save_image(pil_t1, "input_t1")
    _, image_t2_url   = save_image(pil_t2, "input_t2")

    return ChangeDetectionResponse(
        **stats,
        change_map_url=change_map_url,
        mask_t1_url=mask_t1_url,
        mask_t2_url=mask_t2_url,
    )
