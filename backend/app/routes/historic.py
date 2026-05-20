from fastapi import APIRouter
import json
import os

router = APIRouter()

BASE_DIR = os.path.dirname(
    os.path.dirname(
        os.path.dirname(__file__)
    )
)

JSON_PATH = os.path.join(
    BASE_DIR,
    "static",
    "news_data",
    "historic_incidents.json"
)

@router.get("/historic-news")
async def get_historic_news():

    if not os.path.exists(JSON_PATH):
        return []

    with open(JSON_PATH, "r", encoding="utf-8") as f:

        data = json.load(f)

    return data