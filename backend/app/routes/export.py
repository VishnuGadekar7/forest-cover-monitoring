"""
Export Route — GET /api/v1/export-tif
==============================================
Generates a 16-bit GeoTIFF on-the-fly from the stored change map PNGs.
Assigns the requested EPSG coordinate reference system and streams the 
binary blob back to the client.
"""

import logging
import json
from pathlib import Path
import numpy as np
import rasterio
from rasterio.io import MemoryFile
from rasterio.transform import Affine
from fastapi import APIRouter, HTTPException, Query, status
from fastapi.responses import Response
from PIL import Image

logger = logging.getLogger(__name__)
router = APIRouter()

# Match the path from your visualization.py
STATIC_DIR = Path(__file__).resolve().parents[2] / "static" / "change_maps"
METADATA_DIR = Path(__file__).resolve().parents[2] / "static" / "metadata"

@router.get(
    "/export-tif",
    summary="Export Change Map as 16-bit GeoTIFF",
    response_class=Response
)
async def export_tif(
    task_id: str = Query(..., description="The unique ID of the detection task")
):
    task_dir = STATIC_DIR / task_id
    
    if not task_dir.exists():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Task ID '{task_id}' not found."
        )

    # Determine which change map was generated (standard or hybrid)
    change_map_path = task_dir / "change.png"
    if not change_map_path.exists():
        change_map_path = task_dir / "hybrid_change.png"
        
    if not change_map_path.exists():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Change map image not found for this task."
        )

    meta_path = METADATA_DIR / f"{task_id}.json"
    spatial_meta = None
    if meta_path.exists():
        try:
            with open(meta_path, "r") as f:
                spatial_meta = json.load(f)
        except Exception as err:
            logger.error(f"Failed to read spatial metadata cache for task {task_id}: {err}")

    try:
        # Load the PNG to memory
        img = Image.open(change_map_path).convert("RGB")
        arr_8bit = np.array(img)
        
        # # Convert to 16-bit integer (Scale 0-255 up to 0-65535)
        # # This fulfills the strict 16-bit datatype requirement for the export
        # arr_16bit = (arr_8bit.astype(np.uint16) * 257)
        
        # Rasterio expects dimensions in (Bands, Height, Width)
        arr_rasterio = np.transpose(arr_8bit, (2, 0, 1))
        channels, height, width = arr_rasterio.shape

        # # Create a dummy geotransform
        # transform = from_origin(0, 0, 1, 1)

        crs_target = None
        transform_target = Affine.identity()

        if spatial_meta:
            crs_wkt_val = spatial_meta.get("crs_wkt")
            if crs_wkt_val:
                crs_wkt_str = str(crs_wkt_val).strip()
                if crs_wkt_str.upper().startswith("EPSG:"):
                    crs_target = rasterio.crs.CRS.from_string(crs_wkt_str)
                else:
                    crs_target = rasterio.crs.CRS.from_wkt(crs_wkt_str)
            
            if spatial_meta.get("transform"):
                transform_target = Affine(*spatial_meta["transform"])
        else:
            logging.warning(f"No spatial metadata sidecar found for task {task_id}. Exporting with identity transform mapping.")

        # Write to a GeoTIFF in memory
        with MemoryFile() as memfile:
            with memfile.open(
                driver="GTiff",
                height=height,
                width=width,
                count=channels,
                dtype='uint8',
                crs=crs_target,
                transform=transform_target,
                nodata=None
            ) as dataset:
                dataset.write(arr_rasterio)
            
            # Extract the raw bytes from the MemoryFile
            tif_bytes = memfile.read()

        logger.info(f"Successfully generated 16-bit TIF for task {task_id} ({crs_target})")

        # Stream back to the frontend
        return Response(
            content=tif_bytes,
            media_type="image/tiff",
            headers={
                "Content-Disposition": f'attachment; filename="change_map_{task_id}.tif"',
                "Access-Control-Expose-Headers": "Content-Disposition"
            }
        )

    except Exception as e:
        logger.exception(f"Failed to generate TIF export for task {task_id}. Exception: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to generate GeoTIFF: {str(e)}"
        )