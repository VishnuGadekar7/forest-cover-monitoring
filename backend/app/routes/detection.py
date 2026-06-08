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
import uuid
import json
import shutil, tempfile
import logging
import numpy as np
import rasterio
from rasterio.io import MemoryFile
import rasterio.errors
import rasterio.transform
from fastapi import APIRouter, File, UploadFile, HTTPException, status, Form
from PIL import Image

from app.schemas.detection import ChangeDetectionResponse
from app.schemas.stac_query import STACQueryRequest
from app.services.tiling import split_array_into_tiles, stitch_mask_tiles, stream_tiles_from_disk
from app.services.inference_service import InferenceService
from app.services.change_detection import generate_change_map, change_map_to_pil
from app.services.stac_service import STACService
from app.utils.image_preprocessing import validate_image_bytes, load_image
from app.utils.metrics import compute_statistics
from app.utils.visualization import save_image, save_mask_as_image, generate_mask_overlay

logger = logging.getLogger(__name__)
router = APIRouter()

# Maximum allowable pixels for real-time CPU processing
# 50,000,000 pixels is roughly a 7000x7000 boundary (~350 patches total for T1 + T2)
MAX_PIXEL_LIMIT = 50_000_000

# Singleton for STAC service
_stac_service = STACService()

def validate_image_dimensions_safe(image_t1: UploadFile, image_t2: UploadFile, task_id: str) -> None:
    """
    Unified pre-flight validation gate for manual uploads:
      1. Inspects image headers within a bounded 4MB RAM footprint.
      2. Enforces system-wide processing pixel limits.
      3. Verifies that T1 and T2 share identical spatial pixel shapes.
      4. Captures spatial referencing coordinate matrices (preferring T1's CRS profile)
         and saves it as a lightweight JSON sidecar tracking asset.
    """
    dimensions = []
    spatial_meta = {"crs_wkt": None, "transform": []}

    # Iterate over both files sequentially to run structural checks
    for idx, upload_file in enumerate([image_t1, image_t2]):
        filename = (upload_file.filename or "unknown.jpg").lower()
        
        if filename.endswith((".tif", ".tiff")):
            h, w = None, None
            # Try reading 4MB first, fail-over to 16MB if headers sit deeper
            for chunk_size in [1024 * 1024 * 4, 1024 * 1024 * 16]:
                try:
                    upload_file.file.seek(0)
                    file_bytes = upload_file.file.read(chunk_size)
                    
                    with rasterio.open(io.BytesIO(file_bytes)) as src:
                        h, w = src.height, src.width
                        
                        # Extract spatial profiles natively once open succeeds
                        if idx == 0:
                            spatial_meta["crs_wkt"] = src.crs.to_wkt() if src.crs else None
                            spatial_meta["transform"] = list(src.transform)[:6]
                        elif idx == 1 and not spatial_meta["crs_wkt"] and src.crs:
                            spatial_meta["crs_wkt"] = src.crs.to_wkt()
                            spatial_meta["transform"] = list(src.transform)[:6]
                    
                    # If rasterio opened the memory block successfully without crashing, break the chunk loop
                    break
                except (rasterio.errors.RasterioIOError, Exception) as e:
                    # If we already tried the maximum 16MB window and still failed, raise the error
                    if chunk_size == 1024 * 1024 * 16:
                        raise HTTPException(
                            status_code=status.HTTP_400_BAD_REQUEST,
                            detail=f"Invalid GeoTIFF structural metadata in {upload_file.filename}: {e}"
                        )
                    # Otherwise, log a warning and let it fall through to try the larger 16MB chunk
                    logger.warning(f"IFD offset missing in initial 4MB chunk for {upload_file.filename}. Retrying with 16MB...")
                    continue
        else:
            try:
                upload_file.file.seek(0)
                with Image.open(upload_file.file) as img:
                    w, h = img.size
            except Exception as e:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"Invalid standard image layout header in {upload_file.filename}: {e}"
                )
                
        # Enforce maximum compute thresholds individually
        total_pixels = h * w
        if total_pixels > MAX_PIXEL_LIMIT:
            raise HTTPException(
                status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                detail="Image is too large! Please crop your image or upload a smaller file (under 7,000 x 7,000 pixels) to process it quickly."
            )
            
        dimensions.append((w, h))
        upload_file.file.seek(0)  # Always reset stream pointer safely

    # Strict shape alignment validation
    (w1, h1), (w2, h2) = dimensions[0], dimensions[1]
    if w1 != w2 or h1 != h2:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Dimension mismatch! Baseline image T1 is {w1}x{h1}, but analysis image T2 is {w2}x{h2}. Images must share identical pixel dimensions."
        )

    # If neither file had geospatial tags, initialize an identity matrix default profile
    if not spatial_meta["transform"]:
        spatial_meta["transform"] = list(rasterio.transform.Affine.identity())[:6]

    spatial_meta["width"] = w1
    spatial_meta["height"] = h1

    try:
        meta_folder = "static/metadata"
        os.makedirs(meta_folder, exist_ok=True)
        meta_path = os.path.join(meta_folder, f"{task_id}.json")
        
        # Write CRS metadata to JSON file
        with open(meta_path, "w") as f:
            json.dump(spatial_meta, f, indent=2)
        logger.info(f"Successfully cached spatial coordinate anchors to {meta_path}")
    except Exception as e:
        logger.error(f"Failed to write spatial metadata tracking asset: {e}")

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

def prep_for_pil(arr: np.ndarray) -> np.ndarray:
    """
    Prepares a 16-bit or float array for 8-bit PIL Image rendering, optimized for memory.
    """
    if arr.dtype != np.uint8:
        # Drop the 4th channel (NIR) if it exists. PIL only needs RGB (3 channels).
        if arr.shape[-1] >= 4:
            arr_rgb = arr[..., :3]
        else:
            arr_rgb = arr

        # Force the array to float32 BEFORE doing math to prevent 64-bit RAM spikes.
        arr_f32 = arr_rgb.astype(np.float32)

        v_max = np.percentile(arr_f32, 98)
        if v_max == 0:
            v_max = arr_f32.max() if arr_f32.max() > 0 else 1.0

        normalized = np.clip(arr_f32 / v_max, 0.0, 1.0)
        gamma = 1.2 
        brightened = np.power(normalized, 1.0 / gamma)
        return (brightened * 255.0).astype(np.uint8)

    # If already uint8, just ensure it is RGB
    if arr.shape[-1] >= 4:
        return arr[..., :3]
    return arr

def compute_ndwi_mask(arr: np.ndarray) -> np.ndarray:
    """Calculates NDWI from a 4-band array and returns a boolean mask (> 0.0)."""
    arr_f = arr.astype(np.float32)
    # Ensure array has at least 4 bands before extracting Green(1) and NIR(3)
    if arr_f.shape[-1] < 4:
        return np.zeros((arr_f.shape[0], arr_f.shape[1]), dtype=bool)
    
    green = arr_f[..., 1]
    nir = arr_f[..., 3]
    epsilon = 1e-8
    ndwi = (green - nir) / (green + nir + epsilon)
    return ndwi > 0.0

def create_hybrid_rgb_map(forest_mask: np.ndarray, snow_mask: np.ndarray) -> np.ndarray:
    """
    Creates an RGB image mapping the user's specific plotting criteria:
    - Only Forest: Forest Green [34, 139, 34]
    - Only Snow/Water: Royal Blue [65, 105, 225]
    - Forest + Snow/Water: Cyan [0, 255, 255]
    """
    h, w = forest_mask.shape
    rgb = np.zeros((h, w, 3), dtype=np.uint8)
    
    f_bool = forest_mask > 0
    s_bool = snow_mask > 0
    
    rgb[f_bool & ~s_bool] = [34, 139, 34]   # Only Forest
    rgb[~f_bool & s_bool] = [65, 105, 225]  # Only Snow/Water
    rgb[f_bool & s_bool]  = [0, 255, 255]   # Both
    
    return rgb

def process_upload(upload_file: UploadFile, inference: InferenceService, **inference_kwargs) -> tuple[np.ndarray, np.ndarray]:
    """
    Handles memory-safe sliding window inference for an uploaded file.
    Returns: (predicted_mask, original_array)
    """
    filename = (upload_file.filename or "unknown.jpg").lower()
    
    if filename.endswith((".tif", ".tiff")):
        # Disk Streaming for massive GeoTIFFs
        with tempfile.NamedTemporaryFile(delete=False, suffix=".tif") as tmp:
            shutil.copyfileobj(upload_file.file, tmp)
            tmp_path = tmp.name

        try:
            # Stream inference directly from disk
            tile_stream = stream_tiles_from_disk(tmp_path, tile_size=512)
            predicted_tiles = []
            for spec, tile in tile_stream:
                predicted_tiles.append((spec, inference.predict(tile, **inference_kwargs)))

            # Read the full array into RAM once for visualization/NDWI (uint16 ~900MB max)
            with rasterio.open(tmp_path) as src:
                arr = np.transpose(src.read(), (1, 2, 0))
            
            h, w = arr.shape[:2]
            mask = stitch_mask_tiles(predicted_tiles, (h, w))
            return mask, arr
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
    else:
        # Standard in-memory processing for JPG/PNG
        upload_file.file.seek(0)
        file_bytes = upload_file.file.read()
        arr = load_image_to_numpy(file_bytes, filename)
        
        # Use in-memory generator
        tiles_gen = split_array_into_tiles(arr, tile_size=512)
        predicted_tiles = []
        for spec, tile in tiles_gen:
            predicted_tiles.append((spec, inference.predict(tile, **inference_kwargs)))

        h, w = arr.shape[:2]
        mask = stitch_mask_tiles(predicted_tiles, (h, w))
        return mask, arr

@router.post(
    "/detect-change",
    response_model=ChangeDetectionResponse,
)
async def detect_change(
    image_t1: UploadFile = File(...),
    image_t2: UploadFile = File(...),
    model_name: str = Form("attention_unet"),
    contrast_stretch: bool = Form(True),
    percentile_2_98: bool = Form(True),
    esa_offset_fix: bool = Form(False),
    enable_ndvi_veto: bool = Form(True),
    ndvi_threshold: float = Form(0.25),
    band_order: str = Form("RGBN")
):
    task_id = uuid.uuid4().hex[:12]

    validate_image_dimensions_safe(image_t1, image_t2, task_id)

    # Bundle inference options
    inf_kwargs = {
        "contrast_stretch": contrast_stretch,
        "percentile_2_98": percentile_2_98,
        "esa_offset_fix": esa_offset_fix,
        "enable_ndvi_veto": enable_ndvi_veto,
        "ndvi_threshold": ndvi_threshold,
        "band_order": band_order
    }

    try:
        inference = InferenceService(model_name=model_name)
        
        logger.info(f"Processing T1 with {model_name}...")
        mask_t1, arr_t1 = process_upload(image_t1, inference, **inf_kwargs)
        
        logger.info(f"Processing T2 with {model_name}...")
        mask_t2, arr_t2 = process_upload(image_t2, inference, **inf_kwargs)
        
    except Exception as exc:
        logger.exception("Inference failed")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Inference error using {model_name}: {exc}"
        )

    # --- Change Detection & Stats ---
    change_rgb, _ = generate_change_map(mask_t1, mask_t2)
    change_pil    = change_map_to_pil(change_rgb)
    stats = compute_statistics(mask_t1, mask_t2)
        
    pil_t1 = prep_for_pil(arr_t1)
    pil_t2 = prep_for_pil(arr_t2)

    # --- Save Output Images ---
    _, change_map_url = save_image(change_pil, "change", task_id)
    _, mask_t1_url    = save_mask_as_image(mask_t1, "mask_t1", task_id)
    _, mask_t2_url    = save_mask_as_image(mask_t2, "mask_t2", task_id)
    _, image_t1_url   = save_image(pil_t1, "input_t1", task_id)
    _, image_t2_url   = save_image(pil_t2, "input_t2", task_id)

    logger.info(f"Change detection complete. Stats: {stats}")

    return ChangeDetectionResponse(
        **stats,
        id=task_id,
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
    inference = InferenceService(model_name=query.model_name)

    task_id = uuid.uuid4().hex[:12]

    inf_kwargs = {
        "contrast_stretch": getattr(query, 'contrast_stretch', True),
        "percentile_2_98": getattr(query, 'percentile_2_98', True),
        "esa_offset_fix": getattr(query, 'esa_offset_fix', False),
        "enable_ndvi_veto": getattr(query, 'enable_ndvi_veto', True),
        "ndvi_threshold": getattr(query, 'ndvi_threshold', 0.25),
        "band_order": getattr(query, 'band_order', 'RGBN')
    }

    try:
        # Fetch T1 and T2 numpy arrays from AWS Open Data
        # Ensure we use the SAME MGRS tile for both to avoid geographic displacement
        arr_t1, mgrs_id, crs_identity, utm_bounds = _stac_service.fetch_tile_array(query.bbox, query.date_t1, query.max_cloud_cover)
        arr_t2, _, _, _ = _stac_service.fetch_tile_array(query.bbox, query.date_t2, query.max_cloud_cover, preferred_mgrs=mgrs_id)

        logger.info(f"Fetched arrays using MGRS {mgrs_id}: T1 {arr_t1.shape}, T2 {arr_t2.shape}")

        h_t1, w_t1 = arr_t1.shape[:2]
        h_t2, w_t2 = arr_t2.shape[:2]
        
        if (h_t1 * w_t1) > MAX_PIXEL_LIMIT or (h_t2 * w_t2) > MAX_PIXEL_LIMIT:
            raise HTTPException(
                status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                detail="Selected area is too large! Please select a smaller bounding box to process the satellite data quickly."
            )

        # CAPTURE & SAVE STAC GEOSPATIAL METADATA SIDE-CAR DOCUMENT
        try:
            # Use the UTM bounding boxes to compute a precise affine pixel mapping scale
            u_west, u_south, u_east, u_north = utm_bounds
            transform = rasterio.transform.from_bounds(u_west, u_south, u_east, u_north, w_t1, h_t1)
            
            spatial_meta = {
                "crs_wkt": crs_identity,     # Maps to the true native target projection (EPSG:32643)
                "transform": list(transform)[:6],
                "width": w_t1,
                "height": h_t1
            }
            
            meta_folder = "static/metadata"
            os.makedirs(meta_folder, exist_ok=True)
            meta_path = os.path.join(meta_folder, f"{task_id}.json")
            
            with open(meta_path, "w") as f:
                json.dump(spatial_meta, f, indent=2)
            logger.info(f"Successfully cached automated planar metadata matrix to {meta_path}")
        except Exception as meta_err:
            logger.error(f"Failed to record automated tracking file info: {meta_err}")
        
        # Predict masks using the memory generator
        tiles_t1 = split_array_into_tiles(arr_t1, tile_size=512)
        pred_t1 = [(spec, inference.predict(tile, **inf_kwargs)) for spec, tile in tiles_t1]
        mask_t1 = stitch_mask_tiles(pred_t1, (h_t1, w_t1))

        tiles_t2 = split_array_into_tiles(arr_t2, tile_size=512)
        pred_t2 = [(spec, inference.predict(tile, **inf_kwargs)) for spec, tile in tiles_t2]
        mask_t2 = stitch_mask_tiles(pred_t2, (h_t2, w_t2))

        # Change Detection
        change_rgb, _ = generate_change_map(mask_t1, mask_t2)
        change_pil = change_map_to_pil(change_rgb)

        # 5. Compute Statistics
        stats = compute_statistics(mask_t1, mask_t2)

        pil_t1 = prep_for_pil(arr_t1)
        pil_t2 = prep_for_pil(arr_t2)

        # 6. Save Output Images
        _, change_map_url = save_image(change_pil, "change", task_id)
        _, mask_t1_url    = save_mask_as_image(mask_t1, "mask_t1", task_id)
        _, mask_t2_url    = save_mask_as_image(mask_t2, "mask_t2", task_id)
        _, image_t1_url   = save_image(Image.fromarray(pil_t1), "input_t1", task_id)
        _, image_t2_url   = save_image(Image.fromarray(pil_t2), "input_t2", task_id)

        return ChangeDetectionResponse(
            **stats,
            id=task_id,
            change_map_url=change_map_url,
            mask_t1_url=mask_t1_url,
            mask_t2_url=mask_t2_url,
            image_t1_url=image_t1_url,
            image_t2_url=image_t2_url,
        )
    except HTTPException as http_exc:
        raise http_exc # Re-raise explicit HTTP exceptions directly
    except Exception as e:
        logger.exception("Error during automated STAC change detection: %s", str(e))
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))

@router.post(
    "/detect-forest-snow",
    response_model=ChangeDetectionResponse,
    summary="Hybrid Forest and Snow/Water Detection"
)
async def detect_forest_snow(
    image_t1: UploadFile = File(...),
    image_t2: UploadFile = File(...),
    model_name: str = Form("attention_unet"),
    contrast_stretch: bool = Form(True),
    percentile_2_98: bool = Form(True),
    esa_offset_fix: bool = Form(False),
    enable_ndvi_veto: bool = Form(True),
    ndvi_threshold: float = Form(0.25),
    band_order: str = Form("RGBN")
):
    """
    Detects forest using ML and snow/water using NDWI physics.
    Plots explicit hybrid masks and enforces strict persistent-snow overlap on the change map.
    """
    task_id = uuid.uuid4().hex[:12]

    validate_image_dimensions_safe(image_t1, image_t2, task_id)

    inf_kwargs = {
        "contrast_stretch": contrast_stretch,
        "percentile_2_98": percentile_2_98,
        "esa_offset_fix": esa_offset_fix,
        "enable_ndvi_veto": enable_ndvi_veto,
        "ndvi_threshold": ndvi_threshold,
        "band_order": band_order
    }

    # Read & Load Data
    try:
        inference = InferenceService(model_name=model_name)
        
        logger.info(f"Processing T1 with {model_name}...")
        mask_forest_t1, arr_t1 = process_upload(image_t1, inference, **inf_kwargs)
        
        logger.info(f"Processing T2 with {model_name}...")
        mask_forest_t2, arr_t2 = process_upload(image_t2, inference, **inf_kwargs)
    except Exception as exc:
        logger.exception("Inference failed")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Inference error using {model_name}: {exc}"
        )

    mask_snow_t1 = compute_ndwi_mask(arr_t1)
    mask_snow_t2 = compute_ndwi_mask(arr_t2)

    # Generate the Hybrid T1 and T2 Classification Maps
    rgb_hybrid_t1 = create_hybrid_rgb_map(mask_forest_t1, mask_snow_t1)
    rgb_hybrid_t2 = create_hybrid_rgb_map(mask_forest_t2, mask_snow_t2)

    # Generate the Advanced Change Map
    base_change_rgb, _ = generate_change_map(mask_forest_t1, mask_forest_t2)
    hybrid_change_rgb = base_change_rgb.copy()
    
    # Show snow/water ONLY at pixels where it is present in BOTH images
    persistent_snow = mask_snow_t1 & mask_snow_t2
    persistent_forest = (mask_forest_t1 > 0) & (mask_forest_t2 > 0)
    
    # Overlay the persistent snow logic onto the base change map
    hybrid_change_rgb[persistent_snow] = [65, 105, 225]                     # Only Snow
    hybrid_change_rgb[persistent_snow & persistent_forest] = [0, 255, 255]  # Both
    
    hybrid_change_pil = Image.fromarray(hybrid_change_rgb)
    
    # Compute base forest statistics
    stats = compute_statistics(mask_forest_t1, mask_forest_t2)

    # Save Images 
    # Note: Using `save_image` for masks instead of `save_mask_as_image` 
    # because our arrays are already fully color-coded RGB representations, not raw classes.
    pil_t1 = prep_for_pil(arr_t1)
    pil_t2 = prep_for_pil(arr_t2)

    _, change_map_url = save_image(hybrid_change_pil, "hybrid_change", task_id)
    _, mask_t1_url    = save_image(Image.fromarray(rgb_hybrid_t1), "hybrid_mask_t1", task_id)
    _, mask_t2_url    = save_image(Image.fromarray(rgb_hybrid_t2), "hybrid_mask_t2", task_id)
    _, image_t1_url   = save_image(Image.fromarray(pil_t1), "input_t1", task_id)
    _, image_t2_url   = save_image(Image.fromarray(pil_t2), "input_t2", task_id)

    return ChangeDetectionResponse(
        **stats,
        id=task_id,
        change_map_url=change_map_url,
        mask_t1_url=mask_t1_url,
        mask_t2_url=mask_t2_url,
        image_t1_url=image_t1_url,
        image_t2_url=image_t2_url,
    )
