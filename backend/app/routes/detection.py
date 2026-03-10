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
import os
import logging
import numpy as np
from rasterio.io import MemoryFile
from fastapi import APIRouter, File, UploadFile, HTTPException, status
from PIL import Image

from app.schemas.detection import ChangeDetectionResponse
from app.schemas.stac_query import STACQueryRequest
from app.services.inference_service import InferenceService
from app.services.change_detection import generate_change_map, change_map_to_pil
from app.services.stac_service import STACService
from app.utils.image_preprocessing import validate_image_bytes, load_image
from app.utils.metrics import compute_statistics
from app.utils.visualization import save_image, save_mask_as_image, generate_mask_overlay

logger = logging.getLogger(__name__)
router = APIRouter()

# Singleton for STAC service
_stac_service = STACService()

model_name = os.getenv("MODEL_NAME")
if model_name:
	_inference = InferenceService(model_name=model_name)
else:
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
            # detail=f"Inference error using {model_name}: {exc}",
			detail=f"Inference error: {exc}"
        )

    # ── 4. Change Detection & Stats ───────────────────────────────────────────
    change_rgb, _ = generate_change_map(mask_t1, mask_t2)
    change_pil    = change_map_to_pil(change_rgb)
    stats = compute_statistics(mask_t1, mask_t2)

    # ── Prepare arrays for PIL (Convert 16-bit to 8-bit strictly for saving) ──
    def prep_for_pil(arr: np.ndarray) -> np.ndarray:
        """
        Prepares a 16-bit or float array for 8-bit PIL Image rendering.
        Applies a 98th percentile stretch and gamma correction to brighten the image.
        """
        if arr.dtype != np.uint8:
            # 1. The 98th Percentile Stretch
            # Ignore the blindingly bright outliers (clouds, glare)
            v_max = np.percentile(arr, 98)
            
            # Fallback to absolute max if the image is incredibly uniform/dark
            if v_max == 0:
                v_max = arr.max() if arr.max() > 0 else 1
                
            # Normalize strictly between 0.0 and 1.0 based on that 98th percentile cap
            normalized = np.clip(arr / v_max, 0.0, 1.0)
            
            # 2. Gamma Correction (Mid-tone Boost)
            # Gamma > 1.0 brightens the darks and mid-tones without blowing out the whites.
            # 1.2 is a solid baseline for Sentinel-2 data; increase to 1.5 if still too dark.
            gamma = 1.2 
            brightened = np.power(normalized, 1.0 / gamma)
            
            # 3. Scale to 255 and cast
            return (brightened * 255.0).astype(np.uint8)
            
        return arr
        
    pil_t1 = prep_for_pil(arr_t1)
    pil_t2 = prep_for_pil(arr_t2)

    # ── 5. Save Output Images ─────────────────────────────────────────────────
    _, change_map_url = save_image(change_pil, "change")
    _, mask_t1_url    = save_mask_as_image(mask_t1, "mask_t1")
    _, mask_t2_url    = save_mask_as_image(mask_t2, "mask_t2")
    _, image_t1_url   = save_image(pil_t1, "input_t1")
    _, image_t2_url   = save_image(pil_t2, "input_t2")

    logger.info(f"Change detection complete. Stats: {stats}")

    return ChangeDetectionResponse(
        **stats,
        change_map_url=change_map_url,
        mask_t1_url=mask_t1_url,
        mask_t2_url=mask_t2_url,
        image_t1_url=image_t1_url,
        image_t2_url=image_t2_url,
    )

@router.post(
    "/detect-change-automated",
    response_model=ChangeDetectionResponse,
    summary="Automated STAC Ingestion Pipeline",
    description="Fetches cloud-free Sentinel-2 L2A tiles via AWS STAC API given a Bounding Box."
)
async def detect_change_automated(query: STACQueryRequest) -> ChangeDetectionResponse:
    logger.info(f"Received automated STAC request for BBox: {query.bbox}, Model: {query.model_name}")

    # Initialize InferenceService with requested model
    inference_service = InferenceService(model_name=query.model_name)

    try:
        # 1. Fetch T1 and T2 numpy arrays from AWS Open Data
        # Ensure we use the SAME MGRS tile for both to avoid geographic displacement
        arr_t1, mgrs_id = _stac_service.fetch_tile_array(query.bbox, query.date_t1, query.max_cloud_cover)
        arr_t2, _       = _stac_service.fetch_tile_array(query.bbox, query.date_t2, query.max_cloud_cover, preferred_mgrs=mgrs_id)

        logger.info(f"Fetched arrays using MGRS {mgrs_id}: T1 {arr_t1.shape}, T2 {arr_t2.shape}")

        # 2. Convert raw arrays to PIL Images for the InferenceService
        from app.utils.image_preprocessing import TARGET_SIZE
        # Use RGBA mode to preserve 4 bands (RGB + NIR) during resize
        img_t1 = Image.fromarray(arr_t1).resize(TARGET_SIZE, Image.BILINEAR)
        img_t2 = Image.fromarray(arr_t2).resize(TARGET_SIZE, Image.BILINEAR)

        # 3. Predict masks
        mask_t1 = inference_service.predict(img_t1)
        mask_t2 = inference_service.predict(img_t2)

        # 4. Change Detection
        change_rgb, _ = generate_change_map(mask_t1, mask_t2)
        change_pil = change_map_to_pil(change_rgb)

        # 5. Compute Statistics
        stats = compute_statistics(mask_t1, mask_t2)

        # 6. Save Output Images
        _, change_map_url = save_image(change_pil, "change")
        _, mask_t1_url    = save_mask_as_image(mask_t1, "mask_t1")
        _, mask_t2_url    = save_mask_as_image(mask_t2, "mask_t2")
        _, image_t1_url   = save_image(img_t1.convert("RGB"), "input_t1")
        _, image_t2_url   = save_image(img_t2.convert("RGB"), "input_t2")

        return ChangeDetectionResponse(
            **stats,
            change_map_url=change_map_url,
            mask_t1_url=mask_t1_url,
            mask_t2_url=mask_t2_url,
            image_t1_url=image_t1_url,
            image_t2_url=image_t2_url,
        )
    except Exception as e:
        logger.exception("Error during automated STAC change detection: %s", str(e))
        raise HTTPException(status_code=500, detail=str(e))
