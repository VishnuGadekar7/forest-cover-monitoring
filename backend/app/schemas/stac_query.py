from pydantic import BaseModel, Field
from typing import List

class STACQueryRequest(BaseModel):
    bbox: List[float] = Field(..., description="[min_lon, min_lat, max_lon, max_lat] in WGS84")
    date_t1: str = Field(..., description="Start date/time for T1 (e.g. '2023-01-01T00:00:00Z' or '2023-01-01/2023-01-31')")
    date_t2: str = Field(..., description="Start date/time for T2 (e.g. '2023-12-01T00:00:00Z' or '2023-12-01/2023-12-31')")
    max_cloud_cover: int = Field(10, description="Maximum cloud cover percentage")
    model_name: str = Field("attention_unet", description="Model architecture to use (attention_unet, resnet_unet, transnet)")

    contrast_stretch: bool = Field(
        default=True, 
        description="Enable image-wide radiometric contrast stretching."
    )
    percentile_2_98: bool = Field(
        default=True, 
        description="Use 2%-98% boundary indices for stretching to ignore outliers."
    )
    esa_offset_fix: bool = Field(
        default=False, 
        description="Apply the -0.1 baseline offset subtraction for post-2022 ESA Sentinel-2 data."
    )
    enable_ndvi_veto: bool = Field(
        default=True, 
        description="Toggle the NDVI vegetation signature safety filter."
    )
    ndvi_threshold: float = Field(
        default=0.25, 
        description="Dynamic threshold for the NDVI filter (e.g., 0.25 for standard, 0.45 for strict)."
    )
    band_order: str = Field(
        default="RGBN", 
        description="The multi-spectral layer sequence of the target sensor (e.g., 'RGBN', 'NRGB')."
    )

    model_config = {
        "protected_namespaces": ()
    }
