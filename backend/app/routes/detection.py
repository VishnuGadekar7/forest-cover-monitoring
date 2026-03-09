"""
Detection Route — POST /api/v1/detect-change
==============================================
Orchestrates the complete forest change detection pipeline:
  1. Accept & validate two uploaded satellite images
  2. Load images and run inference on both
  3. Generate change map and statistics
  4. Save output images to static directory
  5. Return structured JSON response
"""

import logging
import numpy as np
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


@router.post(
    "/detect-change",
    response_model=ChangeDetectionResponse,
    summary="Run Forest Change Detection",
    description=(
        "Accepts two satellite images captured at different times (T1, T2). "
        "Runs semantic segmentation on both, computes pixel-wise forest change, "
        "and returns statistics plus URLs to the generated change map and masks."
    ),
    responses={
        400: {"description": "Invalid file format or corrupted image"},
        500: {"description": "Internal inference error"},
    },
)
async def detect_change(
    image_t1: UploadFile = File(..., description="Satellite image at Time 1 (PNG/JPG/TIFF)"),
    image_t2: UploadFile = File(..., description="Satellite image at Time 2 (PNG/JPG/TIFF)"),
    model_name: str = "attention_unet",
) -> ChangeDetectionResponse:
    """Full pipeline: upload → inference → change detection → response."""
    
    # Initialize InferenceService with requested model
    inference_service = InferenceService(model_name=model_name)

    # ── 1. Read & Validate ────────────────────────────────────────────────────
    try:
        t1_bytes = await image_t1.read()
        t2_bytes = await image_t2.read()
        validate_image_bytes(t1_bytes, image_t1.filename or "image_t1")
        validate_image_bytes(t2_bytes, image_t2.filename or "image_t2")
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc))

    # ── 2. Load PIL Images ────────────────────────────────────────────────────
    try:
        pil_t1 = load_image(t1_bytes)
        pil_t2 = load_image(t2_bytes)
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Failed to decode image: {exc}",
        )

    # ── 3. Inference — generate binary forest masks ───────────────────────────
    try:
        logger.info(f"Running inference using {model_name} on T1...")
        mask_t1 = inference_service.predict(pil_t1)
        logger.info(f"Running inference using {model_name} on T2...")
        mask_t2 = inference_service.predict(pil_t2)
    except Exception as exc:
        logger.exception("Inference failed")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Inference error using {model_name}: {exc}",
        )

    # ── 4. Change Detection ───────────────────────────────────────────────────
    change_rgb, _ = generate_change_map(mask_t1, mask_t2)
    change_pil    = change_map_to_pil(change_rgb)

    # ── 5. Compute Statistics ─────────────────────────────────────────────────
    stats = compute_statistics(mask_t1, mask_t2)

    # ── 6. Save Output Images ─────────────────────────────────────────────────
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
