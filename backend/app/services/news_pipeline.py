import os
import re
import json
import math
import time
import warnings
import requests
import feedparser
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from urllib.parse import quote

import spacy
from dotenv import load_dotenv

load_dotenv()
from geopy.geocoders import Nominatim
from geopy.exc import (
    GeocoderTimedOut,
    GeocoderServiceError
)

warnings.filterwarnings("ignore")

# =========================================================
# API KEYS
# =========================================================

GNEWS_KEY = os.getenv("GNEWS_KEY")

# =========================================================
# CONFIG
# =========================================================

DAYS_BACK = 90
MAX_INCIDENTS = 50  # Reduced from 300 for performance
MAX_VALID_INCIDENTS = 10  # Early exit after 10 valid incidents

cutoff_date = datetime.utcnow() - timedelta(days=DAYS_BACK)

# =========================================================
# PATHS
# =========================================================

BASE_DIR = os.path.dirname(
    os.path.dirname(
        os.path.dirname(__file__)
    )
)

NEWS_JSON_DIR = os.path.join(
    BASE_DIR,
    "static",
    "news_data"
)

os.makedirs(NEWS_JSON_DIR, exist_ok=True)

OUTPUT_JSON = os.path.join(
    NEWS_JSON_DIR,
    "incidents.json"
)

GEOCODE_CACHE_JSON = os.path.join(
    NEWS_JSON_DIR,
    "geocode_cache.json"
)

GEOCODE_CACHE_JSON = os.path.join(
    NEWS_JSON_DIR,
    "geocode_cache.json"
)

# =========================================================
# SEARCH QUERIES
# =========================================================

GOOGLE_QUERIES = [
    "forest fire India",
    "wildfire India",
    "illegal logging India",
    "tree cutting India",
    "deforestation India",
    "forest encroachment India",
    "forest mafia India",
    "forest reserve fire India",
]

GNEWS_QUERIES = [
    "deforestation India",
    "illegal logging India",
    "forest encroachment India",
    "forest fire India",
    "tree felling India",
    "forest mafia India",
]

# =========================================================
# FILTER TERMS
# =========================================================

INDIA_TERMS = [
    'india','assam','kerala','karnataka','odisha',
    'uttarakhand','maharashtra','tamil nadu',
    'himachal','arunachal','meghalaya',
    'mumbai','nashik','chhattisgarh',
    'jharkhand','madhya pradesh',
    'west bengal','andhra','telangana',
    'punjab','rajasthan','gujarat',
]

FOREST_TERMS = [
    'forest','fire','wildfire','blaze',
    'logging','deforestation',
    'tree','timber','encroach',
    'felling','jungle'
]

REMOVE_TERMS = [
    'exam','upsc','movie','celebrity',
    'stock','sports','cricket',
    'bollywood','recipe','fashion'
]

# =========================================================
# SPACY MODEL
# =========================================================

print("Loading spaCy model...")

nlp = spacy.load("en_core_web_sm")

print("spaCy loaded")

# =========================================================
# RELEVANCE FILTER
# =========================================================

def is_relevant(text):

    t = text.lower()

    if not any(i in t for i in INDIA_TERMS):
        return False

    if not any(f in t for f in FOREST_TERMS):
        return False

    if any(r in t for r in REMOVE_TERMS):
        return False

    return True

# =========================================================
# GOOGLE NEWS
# =========================================================

def fetch_google_news():

    results = []

    for q in GOOGLE_QUERIES:

        try:

            url = (
                "https://news.google.com/rss/search?"
                f"q={quote(q)}&hl=en-IN&gl=IN&ceid=IN:en"
            )

            feed = feedparser.parse(url)

            for entry in feed.entries:

                title = entry.get("title", "")

                summary = re.sub(
                    "<[^<]+?>",
                    "",
                    entry.get("summary", "")
                )

                link = entry.get("link", "")

                date = entry.get("published", "")

                text = title + " " + summary

                if not is_relevant(text):
                    continue

                try:

                    date_clean = (
                        date[:25]
                        .replace(" GMT", " +0000")
                    )

                    article_date = datetime.strptime(
                        date_clean,
                        "%a, %d %b %Y %H:%M:%S %z"
                    )

                    article_date = (
                        article_date
                        .replace(tzinfo=None)
                    )

                    if article_date < cutoff_date:
                        continue

                except:
                    article_date = datetime.utcnow()

                results.append({

                    "source": "Google News",

                    "outlet":
                        title.split(" - ")[-1].strip()
                        if " - " in title
                        else "Unknown",

                    "title":
                        title.split(" - ")[0].strip(),

                    "summary":
                        summary[:250],

                    "url":
                        link,

                    "date":
                        article_date,

                    "coords":
                        None,
                })

            time.sleep(0.2)

        except Exception as e:
            print("Google News error:", e)

    print(f"Google News -> {len(results)} articles")

    return results

# =========================================================
# GNEWS
# =========================================================

def fetch_gnews():

    results = []

    if not GNEWS_KEY:
        return results

    for q in GNEWS_QUERIES:

        try:

            r = requests.get(
                "https://gnews.io/api/v4/search",

                params={
                    "q": q,
                    "lang": "en",
                    "country": "in",
                    "max": 20,
                    "token": GNEWS_KEY,
                },

                timeout=10
            )

            if r.status_code != 200:
                continue

            for a in r.json().get("articles", []):

                text = (
                    a.get("title", "")
                    + " " +
                    a.get("description", "")
                )

                if not is_relevant(text):
                    continue

                try:

                    art_date = datetime.strptime(
                        a.get(
                            "publishedAt",
                            ""
                        )[:19],

                        "%Y-%m-%dT%H:%M:%S"
                    )

                    if art_date < cutoff_date:
                        continue

                except:
                    art_date = datetime.utcnow()

                results.append({

                    "source": "GNews",

                    "outlet":
                        a.get(
                            "source",
                            {}
                        ).get("name", ""),

                    "title":
                        a.get("title", ""),

                    "summary":
                        a.get(
                            "description",
                            ""
                        )[:250],

                    "url":
                        a.get("url", ""),

                    "date":
                        art_date,

                    "coords":
                        None,
                })

            time.sleep(0.2)

        except Exception as e:
            print("GNews error:", e)

    print(f"GNews -> {len(results)} articles")

    return results

# =========================================================
# INCIDENT CLASSIFIER
# =========================================================

INCIDENT_KEYWORDS = {

    "WILDFIRE": [
        "fire","blaze",
        "wildfire","burn"
    ],

    "DEFORESTATION": [
        "deforest","logging",
        "cutting","felling",
        "illegal"
    ],
}

def classify_incident(text):

    t = text.lower()

    scores = {
        k: sum(
            1 for kw in v
            if kw in t
        )

        for k, v in INCIDENT_KEYWORDS.items()
    }

    best = max(scores, key=scores.get)

    return best if scores[best] > 0 else "GENERAL"

# =========================================================
# LOCATION EXTRACTION
# =========================================================

INVALID_LOCATIONS = [
    "india","earth","world",
    "news","government"
]

def clean_location(name):

    if not name:
        return None

    name = name.strip()

    if name.lower() in INVALID_LOCATIONS:
        return None

    if len(name) < 3:
        return None

    return name

INDIA_STATES = [
    "uttarakhand","assam",
    "kerala","karnataka",
    "maharashtra","odisha",
    "jharkhand","gujarat",
    "rajasthan","telangana"
]

def extract_locations(text):

    doc = nlp(text[:512])

    locs = []
    seen = set()

    for ent in doc.ents:

        if ent.label_ in ("GPE", "LOC"):

            name = clean_location(ent.text)

            if not name:
                continue

            if name.lower() not in seen:

                seen.add(name.lower())
                locs.append(name)

    text_lower = text.lower()

    for state in INDIA_STATES:

        if state in text_lower:

            if state not in seen:

                locs.append(state.title())
                seen.add(state)

    return locs

# =========================================================
# GEOCODER
# =========================================================

geocoder = Nominatim(
    user_agent="forest-monitor"
)

# =========================================================
# PERSISTENT GEOCODING CACHE
# =========================================================

def load_geocode_cache():
    """Load geocoding cache from JSON file"""
    if os.path.exists(GEOCODE_CACHE_JSON):
        try:
            with open(GEOCODE_CACHE_JSON, "r", encoding="utf-8") as f:
                return json.load(f)
        except:
            return {}
    return {}

def save_geocode_cache(cache):
    """Save geocoding cache to JSON file"""
    try:
        with open(GEOCODE_CACHE_JSON, "w", encoding="utf-8") as f:
            json.dump(cache, f, indent=2)
    except Exception as e:
        print(f"Failed to save geocode cache: {e}")

geo_cache = load_geocode_cache()

def geocode_place(place_name):

    if not place_name:
        return None

    if place_name in geo_cache:
        return geo_cache[place_name]

    queries = [
        f"{place_name}, India",
        place_name
    ]

    for query in queries:

        try:

            # Reduced from 0.5s to 0.1s to respect rate limits minimally
            time.sleep(0.1)

            result = geocoder.geocode(
                query,
                timeout=5  # Reduced timeout
            )

            if result:

                out = {

                    "lat":
                        round(
                            result.latitude,
                            6
                        ),

                    "lon":
                        round(
                            result.longitude,
                            6
                        ),

                    "display_name":
                        result.address,

                    "matched_on":
                        query
                }

                geo_cache[place_name] = out
                save_geocode_cache(geo_cache)  # Persist cache

                return out

        except (
            GeocoderTimedOut,
            GeocoderServiceError
        ):
            continue

    geo_cache[place_name] = None
    save_geocode_cache(geo_cache)  # Persist failed lookups too

    return None

# =========================================================
# MAIN PIPELINE
# =========================================================

def generate_real_time_incidents():

    print("Fetching live forest news...")

    google_results = fetch_google_news()

    gnews_results = fetch_gnews()

    all_raw = (
        google_results +
        gnews_results
    )

    print(f"Total raw articles: {len(all_raw)}")

    # =====================================================
    # DEDUPLICATION
    # =====================================================

    seen_urls = set()
    seen_titles = set()

    deduped = []

    for item in all_raw:

        url = item.get(
            "url",
            ""
        ).strip()

        title = (
            item.get(
                "title",
                ""
            )
            .strip()
            .lower()[:100]
        )

        if url and url in seen_urls:
            continue

        if title and title in seen_titles:
            continue

        if url:
            seen_urls.add(url)

        if title:
            seen_titles.add(title)

        deduped.append(item)

    print(f"After dedupe: {len(deduped)}")

    # =====================================================
    # NLP + GEOCODING (with early exit)
    # =====================================================

    final_incidents = []

    for i, row in enumerate(deduped[:MAX_INCIDENTS]):

        # Early exit: stop after MAX_VALID_INCIDENTS
        if len(final_incidents) >= MAX_VALID_INCIDENTS:
            print(f"Reached {MAX_VALID_INCIDENTS} valid incidents, stopping processing")
            break

        try:

            full_text = (
                row["title"] + " " +
                row["summary"]
            )

            incident_type = classify_incident(
                full_text
            )

            locations = extract_locations(
                full_text
            )

            if not locations:
                continue

            coords = geocode_place(
                locations[0]
            )

            if not coords:
                continue

            incident = {

                "id": len(final_incidents) + 1,

                "title":
                    row["title"],

                "source":
                    row["source"],

                "date":
                    str(
                        row["date"]
                        .strftime("%Y-%m-%d")
                    ),

                "incident_type":
                    incident_type,

                "location":
                    locations,

                "coordinates": {

                    "lat":
                        coords["lat"],

                    "lon":
                        coords["lon"]
                },

            }

            final_incidents.append(
                incident
            )

        except Exception as e:

            print("Pipeline error:", e)

    # =====================================================
    # SAVE JSON
    # =====================================================

    with open(
        OUTPUT_JSON,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            final_incidents,
            f,
            indent=2,
            ensure_ascii=False
        )

    print(
        f"incidents.json updated "
        f"with {len(final_incidents)} incidents"
    )

    return final_incidents