import json
import os
import time
from typing import Optional

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
async def get_news(
    limit: int = Query(10, ge=1, le=100),
    offset: int = Query(0, ge=0),
    keyword: Optional[str] = Query(None, description="Filter by keyword tag (case-insensitive)"),
    search: Optional[str] = Query(None, description="Free-text search across title and keywords"),
    sort: Optional[str] = Query("date_desc", description="Sort order: date_desc | date_asc | relevance"),
):
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
        return {"data": [], "total": 0, "offset": 0, "limit": limit, "has_more": False}

    with open(LIVE_JSON, "r", encoding="utf-8") as f:
        data = json.load(f)

    # ─── Keyword filter ───────────────────────────────────────────
    if keyword:
        kw_lower = keyword.strip().lower()
        data = [
            item for item in data
            if any(kw_lower in k.lower() for k in item.get("keywords", []))
        ]

    # ─── Free-text search ─────────────────────────────────────────
    if search:
        search_lower = search.strip().lower()
        data = [
            item for item in data
            if (
                search_lower in (item.get("title") or "").lower()
                or search_lower in (item.get("source") or "").lower()
                or search_lower in (item.get("matched_name") or "").lower()
                or any(search_lower in k.lower() for k in item.get("keywords", []))
            )
        ]

    # ─── Sorting ──────────────────────────────────────────────────
    if sort == "date_asc":
        data = sorted(data, key=lambda x: x.get("date") or "", reverse=False)
    elif sort == "relevance":
        data = sorted(data, key=lambda x: x.get("relevance_score") or 0, reverse=True)
    else:  # date_desc (default)
        data = sorted(data, key=lambda x: x.get("date") or "", reverse=True)

    # ─── All unique keywords across filtered set (for chip pills) ─
    all_keywords: list = []
    seen_kw: set = set()
    for item in data:
        for kw in item.get("keywords", []):
            if kw.lower() not in seen_kw:
                seen_kw.add(kw.lower())
                all_keywords.append(kw)

    # ─── Paginate ─────────────────────────────────────────────────
    total = len(data)
    results = data[offset:offset + limit]

    return {
        "data": results,
        "total": total,
        "offset": offset,
        "limit": limit,
        "has_more": (offset + limit) < total,
        "available_keywords": all_keywords,
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