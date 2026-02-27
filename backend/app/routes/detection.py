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
from fastapi import APIRouter, File, UploadFile, HTTPException, status
from PIL import Image

from app.schemas.detection import ChangeDetectionResponse
from app.services.inference_service import InferenceService
from app.services.change_detection import generate_change_map, change_map_to_pil
from app.utils.image_preprocessing import validate_image_bytes, load_image
from app.utils.metrics import compute_statistics
from app.utils.visualization import save_image, save_mask_as_image, generate_mask_overlay

logger = logging.getLogger(__name__)
router = APIRouter()

# Singleton — InferenceService holds a reference to the loaded singleton model
_inference = InferenceService()


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
) -> ChangeDetectionResponse:
    """Full pipeline: upload → inference → change detection → response."""

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
        logger.info("Running inference on T1...")
        mask_t1 = _inference.predict(pil_t1)
        logger.info("Running inference on T2...")
        mask_t2 = _inference.predict(pil_t2)
    except Exception as exc:
        logger.exception("Inference failed")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Inference error: {exc}",
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

    logger.info(f"Change detection complete. Stats: {stats}")

    return ChangeDetectionResponse(
        **stats,
        change_map_url=change_map_url,
        mask_t1_url=mask_t1_url,
        mask_t2_url=mask_t2_url,
    )
