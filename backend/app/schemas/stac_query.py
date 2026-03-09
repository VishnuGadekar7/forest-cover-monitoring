from pydantic import BaseModel, Field
from typing import List

class STACQueryRequest(BaseModel):
    bbox: List[float] = Field(..., description="[min_lon, min_lat, max_lon, max_lat] in WGS84")
    date_t1: str = Field(..., description="Start date/time for T1 (e.g. '2023-01-01T00:00:00Z' or '2023-01-01/2023-01-31')")
    date_t2: str = Field(..., description="Start date/time for T2 (e.g. '2023-12-01T00:00:00Z' or '2023-12-01/2023-12-31')")
    max_cloud_cover: int = Field(10, description="Maximum cloud cover percentage")
    model_name: str = Field("attention_unet", description="Model architecture to use (attention_unet, resnet_unet, transnet)")
