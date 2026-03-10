"""
Pydantic Schemas — Detection API
==================================
Defines request/response contracts for the change detection endpoint.
Using Pydantic v2 syntax.
"""

from pydantic import BaseModel, Field


class ChangeDetectionResponse(BaseModel):
    """
    JSON response returned by POST /api/v1/detect-change.

    All area values are in hectares (ha), assuming a 10 m/pixel
    ground sampling distance (Sentinel-2 standard).
    """
    forest_area_t1: float = Field(
        ..., description="Forest area in T1 image (hectares)"
    )
    forest_area_t2: float = Field(
        ..., description="Forest area in T2 image (hectares)"
    )
    forest_loss: float = Field(
        ..., description="Area of forest lost between T1 and T2 (hectares)"
    )
    forest_gain: float = Field(
        ..., description="Area of forest gained between T1 and T2 (hectares)"
    )
    percentage_change: float = Field(
        ..., description="Net percentage change in forest cover relative to T1"
    )
    change_map_url: str = Field(
        ..., description="Relative URL to the coloured change map PNG"
    )
    mask_t1_url: str = Field(
        ..., description="Relative URL to the predicted binary mask for T1"
    )
    mask_t2_url: str = Field(
        ..., description="Relative URL to the predicted binary mask for T2"
    )
    image_t1_url: str = Field(
        ..., description="Relative URL to the original input image for T1"
    )
    image_t2_url: str = Field(
        ..., description="Relative URL to the original input image for T2"
    )

    model_config = {
        "json_schema_extra": {
            "example": {
                "forest_area_t1": 1245.6,
                "forest_area_t2": 1102.3,
                "forest_loss": 183.4,
                "forest_gain": 40.1,
                "percentage_change": -11.5,
                "change_map_url": "/static/change_maps/abc123_change.png",
                "mask_t1_url": "/static/change_maps/abc123_mask_t1.png",
                "mask_t2_url": "/static/change_maps/abc123_mask_t2.png",
            }
        }
    }


class ErrorResponse(BaseModel):
    detail: str
