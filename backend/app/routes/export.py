"""
Export Route — GET /api/v1/export-tif
==============================================
Generates a 16-bit GeoTIFF on-the-fly from the stored change map PNGs.
Assigns the requested EPSG coordinate reference system and streams the 
binary blob back to the client.
"""

import logging
from pathlib import Path
import numpy as np
from rasterio.io import MemoryFile
from rasterio.transform import from_origin
from fastapi import APIRouter, HTTPException, Query, status
from fastapi.responses import Response
from PIL import Image

logger = logging.getLogger(__name__)
router = APIRouter()

# Match the path from your visualization.py
STATIC_DIR = Path(__file__).resolve().parents[2] / "static" / "change_maps"

@router.get(
    "/export-tif",
    summary="Export Change Map as 16-bit GeoTIFF",
    response_class=Response
)
async def export_tif(
    task_id: str = Query(..., description="The unique ID of the detection task"),
    epsg: int = Query(4326, description="EPSG Coordinate Reference System code")
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

    try:
        # 1. Load the PNG
        img = Image.open(change_map_path).convert("RGB")
        arr_8bit = np.array(img)
        
        # 2. Convert to 16-bit integer (Scale 0-255 up to 0-65535)
        # This fulfills the strict 16-bit datatype requirement for the export
        arr_16bit = (arr_8bit.astype(np.uint16) * 257)
        
        # 3. Rasterio expects dimensions in (Bands, Height, Width)
        arr_rasterio = np.transpose(arr_16bit, (2, 0, 1))
        channels, height, width = arr_rasterio.shape

        # 4. Create a dummy geotransform
        # Since the pure geographic metadata was lost when converting to PNG, 
        # we provide a generic spatial transform so GIS software can open it.
        # (Top left corner at 0,0 with 1 unit pixel size)
        transform = from_origin(0, 0, 1, 1)

        # 5. Write to a GeoTIFF in memory
        with MemoryFile() as memfile:
            with memfile.open(
                driver="GTiff",
                height=height,
                width=width,
                count=channels,
                dtype=str(arr_rasterio.dtype),
                crs=f"EPSG:{epsg}",
                transform=transform,
                nodata=0
            ) as dataset:
                dataset.write(arr_rasterio)
            
            # Extract the raw bytes from the MemoryFile
            tif_bytes = memfile.read()

        logger.info(f"Successfully generated 16-bit TIF for task {task_id} (EPSG:{epsg})")

        # 6. Stream back to the frontend
        return Response(
            content=tif_bytes,
            media_type="image/tiff",
            headers={
                "Content-Disposition": f'attachment; filename="change_map_{task_id}.tif"'
            }
        )

    except Exception as e:
        logger.exception("Failed to generate TIF export")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to generate GeoTIFF: {str(e)}"
        )