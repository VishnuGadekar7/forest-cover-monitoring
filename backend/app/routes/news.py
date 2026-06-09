import json
import os
import time

from fastapi import APIRouter, Query

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
# CACHE SETTINGS
# =====================================================

CACHE_DURATION = 60 * 60 * 6
# 6 hours

# =====================================================
# LIVE NEWS ROUTE
# =====================================================

@router.get("/news")
async def get_news(limit: int = Query(10, ge=1, le=50), offset: int = Query(0, ge=0)):

    print("Checking live forest news cache...")

    should_refresh = True

    if os.path.exists(LIVE_JSON):

        modified_time = os.path.getmtime(LIVE_JSON)

        current_time = time.time()

        age = current_time - modified_time

        if age < CACHE_DURATION:
            should_refresh = False

    if should_refresh:

        print("Refreshing live news...")

        generate_real_time_incidents()

    if not os.path.exists(LIVE_JSON):
        return []

    with open(
        LIVE_JSON,
        "r",
        encoding="utf-8"
    ) as f:

        data = json.load(f)

    # Return paginated results
    total = len(data)
    results = data[offset:offset+limit]
    
    return {
        "data": results,
        "total": total,
        "offset": offset,
        "limit": limit,
        "has_more": (offset + limit) < total
    }

# =====================================================
# HISTORIC NEWS ROUTE
# =====================================================

@router.get("/historic-news")
async def get_historic_news():

    print("Generating historic incidents...")

    if not os.path.exists(HISTORIC_JSON):

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