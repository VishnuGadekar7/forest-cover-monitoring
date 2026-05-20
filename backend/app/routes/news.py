import json
import os

from fastapi import APIRouter

from app.services.news_pipeline import (
    generate_real_time_incidents
)

from app.services.historic_pipeline import (
    generate_historic_incidents
)

router = APIRouter()

# =====================================================
# BASE PATHS
# =====================================================

BASE_DIR = os.path.dirname(
    os.path.dirname(
        os.path.dirname(__file__)
    )
)

LIVE_JSON = os.path.join(
    BASE_DIR,
    "static",
    "news_data",
    "incidents.json"
)

HISTORIC_JSON = os.path.join(
    BASE_DIR,
    "static",
    "news_data",
    "historic_incidents.json"
)

# =====================================================
# LIVE NEWS ROUTE
# =====================================================

@router.get("/news")
async def get_news():

    print("Fetching live forest news...")

    generate_real_time_incidents()

    if not os.path.exists(LIVE_JSON):
        return []

    with open(
        LIVE_JSON,
        "r",
        encoding="utf-8"
    ) as f:

        data = json.load(f)

    return data[:10]

# =====================================================
# HISTORIC NEWS ROUTE
# =====================================================

@router.get("/historic-news")
async def get_historic_news():

    print("Generating historic incidents...")

    generate_historic_incidents()

    if not os.path.exists(HISTORIC_JSON):
        return []

    with open(
        HISTORIC_JSON,
        "r",
        encoding="utf-8"
    ) as f:

        data = json.load(f)

    return data[:10]