import os
import re
import json
import time
import hashlib
import uuid
import warnings
from datetime import datetime, timedelta, timezone
from typing import List, Dict, Optional
from dataclasses import dataclass, asdict, field
from difflib import get_close_matches

import feedparser
import requests
import pandas as pd

import spacy
from geopy.geocoders import Nominatim
from geopy.extra.rate_limiter import RateLimiter

try:
    from bs4 import BeautifulSoup
    _BS4_AVAILABLE = True
except ImportError:
    _BS4_AVAILABLE = False
    print("[news_pipeline] beautifulsoup4 not installed — HTML cleaning will use regex.")

try:
    import trafilatura as _trafilatura
    _TRAFILATURA_AVAILABLE = True
except ImportError:
    _TRAFILATURA_AVAILABLE = False
    print("[news_pipeline] trafilatura not installed — full-text fetch disabled.")

warnings.filterwarnings("ignore")

# =============================================================
# PATHS
# =============================================================

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))

NEWS_JSON_DIR = os.path.join(BASE_DIR, "static", "news_data")
os.makedirs(NEWS_JSON_DIR, exist_ok=True)

OUTPUT_JSON = os.path.join(NEWS_JSON_DIR, "incidents.json")
GEOCODE_CACHE_JSON = os.path.join(NEWS_JSON_DIR, "geocode_cache.json")

# =============================================================
# CONFIG
# =============================================================

DAYS_BACK = 30
MAX_PROCESS = 80      # max articles to run NLP on
MAX_INCIDENTS = 40    # soft cap on final output

# =============================================================
# SPACY MODEL
# =============================================================

print("Loading spaCy model...")
nlp = spacy.load("en_core_web_sm")
print("spaCy loaded.")

# =============================================================
# SOURCES
# =============================================================

GNEWS_TOPICS = [
    "deforestation",
    '"forest fire"',
    '"illegal logging"',
    '"tree felling"',
    '"forest encroachment"',
    '"forest diversion"',
    '"forest clearance"',
]

CORE_FEEDS = [
    {"name": "The Hindu - Environment",
     "url": "https://www.thehindu.com/sci-tech/energy-and-environment/feeder/default.rss",
     "priority": 5},
    {"name": "Mongabay India",
     "url": "https://india.mongabay.com/feed/",
     "priority": 5},
    {"name": "The Wire Science",
     "url": "https://science.thewire.in/feed/",
     "priority": 4},
    {"name": "Times of India - Environment",
     "url": "https://timesofindia.indiatimes.com/rssfeeds/2647163.cms",
     "priority": 3},
    {"name": "Down To Earth",
     "url": "https://www.downtoearth.org.in/feed",
     "priority": 5},
    {"name": "Centre for Science and Environment",
     "url": "https://www.cseindia.org/feed",
     "priority": 2},
    {"name": "India Together - Environment",
     "url": "https://indiatogether.org/environment/rss",
     "priority": 2},
]

INDIA_TRUSTED_SOURCES = {
    "The Hindu - Environment", "Mongabay India", "The Wire Science",
    "Times of India - Environment", "Down To Earth",
}


def _build_gnews_feeds() -> List[Dict]:
    base = (
        "https://news.google.com/rss/search"
        "?q={q}+when:{d}d&hl=en-IN&gl=IN&ceid=IN:en"
    )
    return [
        {
            "name": f"Google News - {t.strip(chr(34))}",
            "url": base.format(q=t.replace(" ", "+"), d=DAYS_BACK),
            "priority": 2,
        }
        for t in GNEWS_TOPICS
    ]


ALL_FEEDS = CORE_FEEDS + _build_gnews_feeds()

# =============================================================
# ARTICLE SCHEMA
# =============================================================

@dataclass
class Article:
    title: str
    url: str
    source: str
    published_at: str
    content: str = ""
    description: str = ""
    article_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    ingestion_time: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )

    def to_dict(self) -> dict:
        return asdict(self)

# =============================================================
# RSS COLLECTION
# =============================================================

def _parse_date(entry) -> str:
    if hasattr(entry, "published_parsed") and entry.published_parsed:
        try:
            return datetime(
                *entry.published_parsed[:6], tzinfo=timezone.utc
            ).isoformat()
        except Exception:
            pass
    return datetime.now(timezone.utc).isoformat()


def _collect_from_rss(feeds: List[Dict]) -> List[Article]:
    articles: List[Article] = []
    print(f"Collecting from {len(feeds)} feeds...")
    for feed_info in feeds:
        name, url = feed_info["name"], feed_info["url"]
        try:
            feed = feedparser.parse(url)
            count = 0
            for entry in feed.entries:
                title = getattr(entry, "title", "").strip()
                link = getattr(entry, "link", "").strip()
                if not title or not link:
                    continue
                description = getattr(entry, "summary", "") or getattr(
                    entry, "description", ""
                )
                content = description
                if hasattr(entry, "content"):
                    try:
                        content = entry.content[0].value
                    except Exception:
                        pass
                articles.append(
                    Article(
                        title=title,
                        url=link,
                        source=name,
                        published_at=_parse_date(entry),
                        description=description[:1000],
                        content=content[:5000],
                    )
                )
                count += 1
            print(f"  {name}: {count} articles")
        except Exception as e:
            print(f"  {name}: error — {e}")
        time.sleep(0.8)
    print(f"Total collected: {len(articles)}")
    return articles

# =============================================================
# CLEANING + DEDUPLICATION
# =============================================================

def _clean_text(text: str) -> str:
    if not text:
        return ""
    if _BS4_AVAILABLE:
        text = BeautifulSoup(text, "html.parser").get_text(separator=" ")
    else:
        text = re.sub(r"<[^<]+?>", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def _clean_article(article: dict) -> Optional[dict]:
    cleaned = dict(article)
    cleaned["title"] = _clean_text(article.get("title", ""))
    cleaned["description"] = _clean_text(article.get("description", ""))
    cleaned["content"] = _clean_text(article.get("content", ""))
    return cleaned if len(cleaned["title"]) >= 10 else None


def _content_hash(text: str) -> str:
    if not text:
        return ""
    return hashlib.sha256(" ".join(text.lower().split()).encode()).hexdigest()


def _deduplicate(articles: List[dict]) -> List[dict]:
    seen_urls: set = set()
    seen_hashes: set = set()
    unique: List[dict] = []
    for art in articles:
        url = art.get("url", "").strip().lower()
        content = (
            art.get("content") or art.get("description") or art.get("title") or ""
        )
        h = _content_hash(content)
        if url and url in seen_urls:
            continue
        if h and h in seen_hashes:
            continue
        if url:
            seen_urls.add(url)
        if h:
            seen_hashes.add(h)
        unique.append(art)
    return unique

# =============================================================
# FULL-TEXT FALLBACK
# =============================================================

def _fetch_full_text(url: str, min_len: int = 300) -> str:
    if not _TRAFILATURA_AVAILABLE:
        return ""
    try:
        downloaded = _trafilatura.fetch_url(url)
        if downloaded:
            extracted = _trafilatura.extract(downloaded) or ""
            if len(extracted) >= min_len:
                return extracted
    except Exception:
        pass
    return ""

# =============================================================
# RELEVANCE FILTER
# =============================================================

FOREST_KEYWORDS = [
    "deforestation", "forest cover", "forest land", "forest fire", "forest fires",
    "wildfire", "illegal logging", "tree felling", "tree cutting", "trees felled",
    "trees cut", "timber smuggling", "encroachment", "forest encroachment",
    "forest diversion", "forest clearance", "compensatory afforestation",
    "afforestation", "forest degradation", "loss of forest", "forest loss",
    "tree cover loss", "reserved forest", "protected forest", "mining in forest",
    "coal block", "forest rights", "green cover", "canopy loss",
    "eco-sensitive zone", "wildlife corridor", "forest survey of india", "fsi report",
]

INDIA_KEYWORDS = [
    "india", "indian", "odisha", "uttarakhand", "chhattisgarh", "madhya pradesh",
    "maharashtra", "karnataka", "kerala", "assam", "arunachal", "jharkhand",
    "telangana", "andhra", "tamil nadu", "gujarat", "rajasthan", "himachal",
    "sikkim", "meghalaya", "manipur", "nagaland", "mizoram", "tripura",
    "west bengal", "bihar", "uttar pradesh", "goa", "delhi", "national park",
    "tiger reserve", "wildlife sanctuary", "forest department", "moefcc",
    "bodoland", "kaziranga", "bandipur", "simlipal", "kanha", "corbett",
    "sundarbans", "rishikesh", "angul", "raimona", "shillong",
]

REMOVE_TERMS = [
    "exam", "upsc", "movie", "celebrity", "stock market",
    "cricket", "bollywood", "recipe", "fashion",
]


def _is_relevant(article: dict, min_score: float = 3.0):
    text = " ".join([
        article.get("title", ""),
        article.get("description", ""),
        article.get("content", ""),
    ]).lower()

    if any(r in text for r in REMOVE_TERMS):
        return False, 0.0, "noise"
    if not any(kw in text for kw in FOREST_KEYWORDS):
        return False, 0.0, "no forest signal"
    if not any(kw in text for kw in INDIA_KEYWORDS):
        return False, 0.0, "no India signal"

    score = 3.0
    if any(x in text for x in [
        "forest fire", "deforestation", "tree felling", "encroachment"
    ]):
        score += 2.0
    if article.get("source") in INDIA_TRUSTED_SOURCES:
        score += 0.5

    return score >= min_score, score, "India + forest"

# =============================================================
# EVENT TYPE
# =============================================================

EVENT_PATTERNS = {
    "WILDFIRE":       [r"forest fire", r"wildfire", r"blaze"],
    "DEFORESTATION":  [r"deforestation", r"forest loss", r"tree cover loss"],
    "TREE_FELLING":   [r"tree felling", r"tree cutting", r"trees felled", r"trees cut"],
    "ENCROACHMENT":   [r"encroachment", r"forest encroachment"],
    "FOREST_DIVERSION": [
        r"forest diversion", r"forest clearance",
        r"land diversion", r"mining in forest", r"coal block",
    ],
}


def _detect_event_type(text: str) -> str:
    t = text.lower()
    for event, patterns in EVENT_PATTERNS.items():
        for pat in patterns:
            if re.search(pat, t):
                return event
    return "GENERAL"

# =============================================================
# GAZETTEER
# =============================================================

GAZETTEER = {
    # --- National Parks & Tiger Reserves ---
    "raimona national park":      (26.66, 89.98, "Raimona National Park, Assam", "national_park"),
    "raimona":                    (26.66, 89.98, "Raimona National Park, Assam", "national_park"),
    "kaziranga national park":    (26.58, 93.17, "Kaziranga National Park, Assam", "national_park"),
    "kaziranga":                  (26.58, 93.17, "Kaziranga National Park, Assam", "national_park"),
    "manas national park":        (26.72, 91.00, "Manas National Park, Assam", "national_park"),
    "manas":                      (26.72, 91.00, "Manas National Park, Assam", "national_park"),
    "nameri":                     (26.92, 92.85, "Nameri National Park, Assam", "national_park"),
    "orang":                      (26.55, 92.30, "Orang National Park, Assam", "national_park"),
    "dibru saikhowa":             (27.65, 95.35, "Dibru-Saikhowa National Park, Assam", "national_park"),
    "corbett national park":      (29.53, 78.95, "Jim Corbett National Park, Uttarakhand", "national_park"),
    "jim corbett":                (29.53, 78.95, "Jim Corbett National Park, Uttarakhand", "national_park"),
    "corbett":                    (29.53, 78.95, "Jim Corbett National Park, Uttarakhand", "national_park"),
    "rajaji national park":       (30.05, 78.20, "Rajaji National Park, Uttarakhand", "national_park"),
    "rajaji":                     (30.05, 78.20, "Rajaji National Park, Uttarakhand", "national_park"),
    "valley of flowers":          (30.73, 79.60, "Valley of Flowers National Park, Uttarakhand", "national_park"),
    "nanda devi":                 (30.38, 79.97, "Nanda Devi National Park, Uttarakhand", "national_park"),
    "bandipur national park":     (11.66, 76.63, "Bandipur National Park, Karnataka", "national_park"),
    "bandipur":                   (11.66, 76.63, "Bandipur National Park, Karnataka", "national_park"),
    "nagarhole national park":    (12.00, 76.15, "Nagarhole National Park, Karnataka", "national_park"),
    "nagarhole":                  (12.00, 76.15, "Nagarhole National Park, Karnataka", "national_park"),
    "bhadra":                     (13.50, 75.50, "Bhadra Tiger Reserve, Karnataka", "tiger_reserve"),
    "bannerghatta":               (12.80, 77.58, "Bannerghatta National Park, Karnataka", "national_park"),
    "kanha national park":        (22.28, 80.61, "Kanha National Park, Madhya Pradesh", "national_park"),
    "kanha":                      (22.28, 80.61, "Kanha National Park, Madhya Pradesh", "national_park"),
    "bandhavgarh national park":  (23.70, 81.03, "Bandhavgarh National Park, Madhya Pradesh", "national_park"),
    "bandhavgarh":                (23.70, 81.03, "Bandhavgarh National Park, Madhya Pradesh", "national_park"),
    "pench national park":        (21.70, 79.30, "Pench National Park, Madhya Pradesh", "national_park"),
    "pench":                      (21.70, 79.30, "Pench National Park, Madhya Pradesh", "national_park"),
    "satpura":                    (22.50, 78.30, "Satpura National Park, Madhya Pradesh", "national_park"),
    "panna national park":        (24.72, 80.18, "Panna National Park, Madhya Pradesh", "national_park"),
    "panna":                      (24.72, 80.18, "Panna National Park, Madhya Pradesh", "national_park"),
    "simlipal national park":     (21.83, 86.40, "Simlipal National Park, Odisha", "national_park"),
    "simlipal":                   (21.83, 86.40, "Simlipal National Park, Odisha", "national_park"),
    "bhitarkanika":               (20.70, 86.90, "Bhitarkanika National Park, Odisha", "national_park"),
    "sundarbans national park":   (21.95, 89.18, "Sundarbans National Park, West Bengal", "national_park"),
    "sundarbans":                 (21.95, 89.18, "Sundarbans National Park, West Bengal", "national_park"),
    "sundarban":                  (21.95, 89.18, "Sundarbans National Park, West Bengal", "national_park"),
    "gorumara":                   (26.75, 88.80, "Gorumara National Park, West Bengal", "national_park"),
    "jaldapara":                  (26.68, 89.30, "Jaldapara National Park, West Bengal", "national_park"),
    "periyar national park":      (9.46, 77.17, "Periyar National Park, Kerala", "national_park"),
    "periyar":                    (9.46, 77.17, "Periyar National Park, Kerala", "national_park"),
    "silent valley":              (11.13, 76.45, "Silent Valley National Park, Kerala", "national_park"),
    "eravikulam":                 (10.20, 77.07, "Eravikulam National Park, Kerala", "national_park"),
    "mudumalai national park":    (11.57, 76.55, "Mudumalai National Park, Tamil Nadu", "national_park"),
    "mudumalai":                  (11.57, 76.55, "Mudumalai National Park, Tamil Nadu", "national_park"),
    "anamalai":                   (10.40, 77.00, "Anamalai Tiger Reserve, Tamil Nadu", "tiger_reserve"),
    "annamalai":                  (10.40, 77.00, "Anamalai Tiger Reserve, Tamil Nadu", "tiger_reserve"),
    "guindy":                     (13.00, 80.23, "Guindy National Park, Chennai", "national_park"),
    "gir national park":          (21.13, 70.80, "Gir National Park, Gujarat", "national_park"),
    "gir":                        (21.13, 70.80, "Gir National Park, Gujarat", "national_park"),
    "ranthambore national park":  (26.02, 76.50, "Ranthambore National Park, Rajasthan", "national_park"),
    "ranthambore":                (26.02, 76.50, "Ranthambore National Park, Rajasthan", "national_park"),
    "sariska":                    (27.33, 76.45, "Sariska Tiger Reserve, Rajasthan", "tiger_reserve"),
    "keoladeo":                   (27.17, 77.52, "Keoladeo National Park, Rajasthan", "national_park"),
    "bharatpur":                  (27.17, 77.52, "Keoladeo Ghana National Park, Bharatpur", "national_park"),
    "tadoba andhari":             (20.25, 79.30, "Tadoba Andhari Tiger Reserve, Maharashtra", "tiger_reserve"),
    "tadoba":                     (20.25, 79.30, "Tadoba Andhari Tiger Reserve, Maharashtra", "tiger_reserve"),
    "melghat":                    (21.40, 77.20, "Melghat Tiger Reserve, Maharashtra", "tiger_reserve"),
    "nawegaon":                   (20.90, 80.20, "Nawegaon Nagzira Tiger Reserve, Maharashtra", "tiger_reserve"),
    "indravati":                  (19.10, 81.00, "Indravati National Park, Chhattisgarh", "national_park"),
    "kanger valley":              (18.85, 81.95, "Kanger Valley National Park, Chhattisgarh", "national_park"),
    "guru ghasidas":              (23.50, 82.50, "Guru Ghasidas National Park, Chhattisgarh", "national_park"),
    "palamau":                    (23.80, 84.20, "Palamau Tiger Reserve, Jharkhand", "tiger_reserve"),
    "betla":                      (23.88, 84.19, "Betla National Park, Jharkhand", "national_park"),
    "valmiki":                    (27.30, 84.15, "Valmiki Tiger Reserve, Bihar", "tiger_reserve"),
    "dudhwa national park":       (28.50, 80.70, "Dudhwa National Park, Uttar Pradesh", "national_park"),
    "dudhwa":                     (28.50, 80.70, "Dudhwa National Park, Uttar Pradesh", "national_park"),
    "namdapha":                   (27.50, 96.40, "Namdapha National Park, Arunachal Pradesh", "national_park"),
    "mouling":                    (28.60, 94.80, "Mouling National Park, Arunachal Pradesh", "national_park"),
    "keibul lamjao":              (24.50, 93.80, "Keibul Lamjao National Park, Manipur", "national_park"),
    "intanki":                    (25.50, 93.70, "Intanki National Park, Nagaland", "national_park"),
    "balphakram":                 (25.40, 90.80, "Balphakram National Park, Meghalaya", "national_park"),
    "nokrek":                     (25.50, 90.30, "Nokrek National Park, Meghalaya", "national_park"),
    # --- Districts / Cities / Forest Hotspots ---
    "angul district":   (20.84, 85.15, "Angul, Odisha", "district"),
    "angul":            (20.84, 85.15, "Angul, Odisha", "district"),
    "rishikesh":        (30.09, 78.27, "Rishikesh, Uttarakhand", "city"),
    "dehradun":         (30.32, 78.03, "Dehradun, Uttarakhand", "city"),
    "haridwar":         (29.95, 78.16, "Haridwar, Uttarakhand", "city"),
    "nainital":         (29.38, 79.45, "Nainital, Uttarakhand", "city"),
    "shillong":         (25.57, 91.89, "Shillong, Meghalaya", "city"),
    "guwahati":         (26.14, 91.74, "Guwahati, Assam", "city"),
    "kokrajhar":        (26.40, 90.27, "Kokrajhar, Assam", "district"),
    "bodoland":         (26.55, 90.50, "Bodoland Territorial Region, Assam", "region"),
    "jorhat":           (26.75, 94.20, "Jorhat, Assam", "district"),
    "dibrugarh":        (27.47, 94.91, "Dibrugarh, Assam", "district"),
    "tinsukia":         (27.50, 95.36, "Tinsukia, Assam", "district"),
    "sonitpur":         (26.65, 92.80, "Sonitpur, Assam", "district"),
    "karbi anglong":    (26.00, 93.50, "Karbi Anglong, Assam", "district"),
    "kohima":           (25.67, 94.12, "Kohima, Nagaland", "city"),
    "dimapur":          (25.90, 93.73, "Dimapur, Nagaland", "city"),
    "itanagar":         (27.10, 93.62, "Itanagar, Arunachal Pradesh", "city"),
    "tawang":           (27.58, 91.87, "Tawang, Arunachal Pradesh", "district"),
    "tura":             (25.52, 90.22, "Tura, Meghalaya", "city"),
    "imphal":           (24.82, 93.94, "Imphal, Manipur", "city"),
    "aizawl":           (23.73, 92.72, "Aizawl, Mizoram", "city"),
    "agartala":         (23.83, 91.28, "Agartala, Tripura", "city"),
    "bhubaneswar":      (20.27, 85.84, "Bhubaneswar, Odisha", "city"),
    "sambalpur":        (21.47, 83.97, "Sambalpur, Odisha", "district"),
    "kalahandi":        (19.91, 83.17, "Kalahandi, Odisha", "district"),
    "koraput":          (18.81, 82.71, "Koraput, Odisha", "district"),
    "malkangiri":       (18.35, 81.98, "Malkangiri, Odisha", "district"),
    "mayurbhanj":       (21.93, 86.73, "Mayurbhanj, Odisha", "district"),
    "sundargarh":       (22.12, 84.03, "Sundargarh, Odisha", "district"),
    "raipur":           (21.25, 81.63, "Raipur, Chhattisgarh", "city"),
    "bastar":           (19.20, 81.93, "Bastar, Chhattisgarh", "district"),
    "dantewada":        (18.90, 81.35, "Dantewada, Chhattisgarh", "district"),
    "sukma":            (18.40, 81.67, "Sukma, Chhattisgarh", "district"),
    "bhopal":           (23.26, 77.41, "Bhopal, Madhya Pradesh", "city"),
    "jabalpur":         (23.18, 79.99, "Jabalpur, Madhya Pradesh", "city"),
    "mandla":           (22.60, 80.38, "Mandla, Madhya Pradesh", "district"),
    "balaghat":         (21.81, 80.19, "Balaghat, Madhya Pradesh", "district"),
    "seoni":            (22.09, 79.55, "Seoni, Madhya Pradesh", "district"),
    "chhindwara":       (22.06, 78.94, "Chhindwara, Madhya Pradesh", "district"),
    "nagpur":           (21.15, 79.09, "Nagpur, Maharashtra", "city"),
    "chandrapur":       (19.97, 79.30, "Chandrapur, Maharashtra", "district"),
    "gadchiroli":       (20.18, 80.00, "Gadchiroli, Maharashtra", "district"),
    "gondia":           (21.46, 80.20, "Gondia, Maharashtra", "district"),
    "ranchi":           (23.34, 85.31, "Ranchi, Jharkhand", "city"),
    "west singhbhum":   (22.50, 85.50, "West Singhbhum, Jharkhand", "district"),
    "gumla":            (23.04, 84.54, "Gumla, Jharkhand", "district"),
    "latehar":          (23.75, 84.50, "Latehar, Jharkhand", "district"),
    "bengaluru":        (12.97, 77.59, "Bengaluru, Karnataka", "city"),
    "mysuru":           (12.30, 76.65, "Mysuru, Karnataka", "city"),
    "mysore":           (12.30, 76.65, "Mysuru, Karnataka", "city"),
    "kodagu":           (12.42, 75.74, "Kodagu (Coorg), Karnataka", "district"),
    "coorg":            (12.42, 75.74, "Kodagu (Coorg), Karnataka", "district"),
    "chikkamagaluru":   (13.32, 75.77, "Chikkamagaluru, Karnataka", "district"),
    "shivamogga":       (13.93, 75.57, "Shivamogga, Karnataka", "district"),
    "uttara kannada":   (14.60, 74.60, "Uttara Kannada, Karnataka", "district"),
    "kochi":            (9.93, 76.27, "Kochi, Kerala", "city"),
    "idukki":           (9.85, 76.97, "Idukki, Kerala", "district"),
    "wayanad":          (11.69, 76.13, "Wayanad, Kerala", "district"),
    "palakkad":         (10.79, 76.65, "Palakkad, Kerala", "district"),
    "thiruvananthapuram": (8.52, 76.94, "Thiruvananthapuram, Kerala", "city"),
    "coimbatore":       (11.02, 76.96, "Coimbatore, Tamil Nadu", "city"),
    "ooty":             (11.41, 76.70, "Udhagamandalam (Ooty), Tamil Nadu", "city"),
    "nilgiris":         (11.40, 76.70, "Nilgiris, Tamil Nadu", "district"),
    "nilgiri":          (11.40, 76.70, "Nilgiris, Tamil Nadu", "district"),
    "visakhapatnam":    (17.69, 83.22, "Visakhapatnam, Andhra Pradesh", "city"),
    "vizag":            (17.69, 83.22, "Visakhapatnam, Andhra Pradesh", "city"),
    "east godavari":    (17.00, 81.80, "East Godavari, Andhra Pradesh", "district"),
    "hyderabad":        (17.39, 78.49, "Hyderabad, Telangana", "city"),
    "adilabad":         (19.67, 78.53, "Adilabad, Telangana", "district"),
    "khammam":          (17.25, 80.15, "Khammam, Telangana", "district"),
    "junagadh":         (21.52, 70.46, "Junagadh, Gujarat", "district"),
    "jaipur":           (26.91, 75.79, "Jaipur, Rajasthan", "city"),
    "udaipur":          (24.59, 73.71, "Udaipur, Rajasthan", "city"),
    "alwar":            (27.57, 76.61, "Alwar, Rajasthan", "district"),
    "sawai madhopur":   (25.99, 76.37, "Sawai Madhopur, Rajasthan", "district"),
    "kolkata":          (22.57, 88.36, "Kolkata, West Bengal", "city"),
    "darjeeling":       (27.04, 88.26, "Darjeeling, West Bengal", "district"),
    "jalpaiguri":       (26.52, 88.73, "Jalpaiguri, West Bengal", "district"),
    "alipurduar":       (26.49, 89.53, "Alipurduar, West Bengal", "district"),
    "patna":            (25.59, 85.14, "Patna, Bihar", "city"),
    "west champaran":   (27.10, 84.50, "West Champaran, Bihar", "district"),
    "lucknow":          (26.85, 80.95, "Lucknow, Uttar Pradesh", "city"),
    "lakhimpur kheri":  (27.95, 80.78, "Lakhimpur Kheri, Uttar Pradesh", "district"),
    "pilibhit":         (28.64, 79.81, "Pilibhit, Uttar Pradesh", "district"),
    "shimla":           (31.10, 77.17, "Shimla, Himachal Pradesh", "city"),
    "kullu":            (31.96, 77.11, "Kullu, Himachal Pradesh", "district"),
    "kangra":           (32.10, 76.27, "Kangra, Himachal Pradesh", "district"),
    # --- State centroids (lowest-priority fallback) ---
    "assam":            (26.20, 92.94, "Assam", "state"),
    "uttarakhand":      (30.07, 79.02, "Uttarakhand", "state"),
    "chhattisgarh":     (21.28, 81.87, "Chhattisgarh", "state"),
    "madhya pradesh":   (22.97, 78.66, "Madhya Pradesh", "state"),
    "maharashtra":      (19.75, 75.71, "Maharashtra", "state"),
    "karnataka":        (15.32, 75.71, "Karnataka", "state"),
    "kerala":           (10.85, 76.27, "Kerala", "state"),
    "tamil nadu":       (11.13, 78.66, "Tamil Nadu", "state"),
    "andhra pradesh":   (15.91, 79.74, "Andhra Pradesh", "state"),
    "telangana":        (18.11, 79.02, "Telangana", "state"),
    "gujarat":          (22.26, 71.19, "Gujarat", "state"),
    "rajasthan":        (27.02, 74.22, "Rajasthan", "state"),
    "west bengal":      (22.99, 87.86, "West Bengal", "state"),
    "jharkhand":        (23.61, 85.28, "Jharkhand", "state"),
    "odisha":           (20.95, 85.10, "Odisha", "state"),
    "arunachal pradesh": (28.22, 94.73, "Arunachal Pradesh", "state"),
    "arunachal":        (28.22, 94.73, "Arunachal Pradesh", "state"),
    "meghalaya":        (25.47, 91.37, "Meghalaya", "state"),
    "nagaland":         (26.16, 94.56, "Nagaland", "state"),
    "manipur":          (24.66, 93.90, "Manipur", "state"),
    "mizoram":          (23.36, 92.80, "Mizoram", "state"),
    "tripura":          (23.94, 91.99, "Tripura", "state"),
    "himachal pradesh": (31.10, 77.17, "Himachal Pradesh", "state"),
    "himachal":         (31.10, 77.17, "Himachal Pradesh", "state"),
    "bihar":            (25.10, 85.31, "Bihar", "state"),
    "uttar pradesh":    (26.85, 80.91, "Uttar Pradesh", "state"),
    "goa":              (15.30, 74.12, "Goa", "state"),
    "delhi":            (28.61, 77.21, "Delhi", "state"),
    "punjab":           (31.15, 75.34, "Punjab", "state"),
    "haryana":          (29.06, 76.09, "Haryana", "state"),
    "sikkim":           (27.53, 88.51, "Sikkim", "state"),
}

BROAD_REJECT = {
    "india", "indian", "indian subcontinent", "south asia",
    "asia", "subcontinent", "bharat", "republic of india",
}

# =============================================================
# GEOCODING (Nominatim + cache)
# =============================================================

geolocator = Nominatim(user_agent="forest_event_pipeline_india_v2")
_geocode_rate = RateLimiter(geolocator.geocode, min_delay_seconds=1.2)


def _load_geocode_cache() -> dict:
    if os.path.exists(GEOCODE_CACHE_JSON):
        try:
            with open(GEOCODE_CACHE_JSON, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return {}
    return {}


def _save_geocode_cache(cache: dict):
    try:
        with open(GEOCODE_CACHE_JSON, "w", encoding="utf-8") as f:
            json.dump(cache, f, indent=2)
    except Exception as e:
        print(f"Failed to save geocode cache: {e}")


_geo_cache = _load_geocode_cache()

# =============================================================
# LOCATION EXTRACTION
# =============================================================

def _extract_location_candidates(text: str) -> List[str]:
    candidates: List[str] = []
    doc = nlp(text[:2000])
    for ent in doc.ents:
        if ent.label_ in ("GPE", "LOC", "FAC"):
            candidates.append(ent.text.strip())
    patterns = [
        r"(?:in|at|near|around|from)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,3})",
        r"([A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,2})\s+"
        r"(?:National Park|Wildlife Sanctuary|Tiger Reserve|Forest|District)",
    ]
    for pat in patterns:
        for match in re.finditer(pat, text):
            candidates.append(match.group(1).strip())
    cleaned: List[str] = []
    seen: set = set()
    for c in candidates:
        c = re.sub(r"[^\w\s]", "", c).strip()
        if len(c) > 2 and c.lower() not in seen:
            seen.add(c.lower())
            cleaned.append(c)
    return cleaned


def _is_broad(name: str) -> bool:
    if not name:
        return True
    n = name.lower().strip()
    return any(b in n for b in BROAD_REJECT) or n in BROAD_REJECT


def _resolve_location(text: str, title: str = "") -> dict:
    full_text = (title + " " + text).lower()
    sorted_keys = sorted(GAZETTEER.keys(), key=len, reverse=True)

    best = None
    for key in sorted_keys:
        if key in full_text:
            lat, lon, name, typ = GAZETTEER[key]
            conf = "high" if typ in (
                "national_park", "tiger_reserve", "city", "district", "place"
            ) else "medium"
            candidate = {
                "lat": lat, "lon": lon,
                "matched_name": name,
                "location_type": typ,
                "geo_confidence": conf,
                "geo_method": "gazetteer",
            }
            if best is None:
                best = candidate
            elif best["location_type"] == "state" and typ != "state":
                best = candidate
            if typ != "state":
                return candidate

    if best:
        return best

    candidates = _extract_location_candidates(title + ". " + text)
    for cand in candidates:
        cand_l = cand.lower().strip()
        if not cand_l or len(cand_l) < 3:
            continue
        if cand_l in GAZETTEER:
            lat, lon, name, typ = GAZETTEER[cand_l]
            return {
                "lat": lat, "lon": lon,
                "matched_name": name,
                "location_type": typ,
                "geo_confidence": "high" if typ != "state" else "medium",
                "geo_method": "gazetteer_candidate",
            }
        close = get_close_matches(cand_l, GAZETTEER.keys(), n=1, cutoff=0.85)
        if close:
            lat, lon, name, typ = GAZETTEER[close[0]]
            return {
                "lat": lat, "lon": lon,
                "matched_name": name,
                "location_type": typ,
                "geo_confidence": "medium" if typ != "state" else "low",
                "geo_method": "gazetteer_fuzzy",
            }

    for cand in candidates[:3]:
        cache_key = f"{cand}__India"
        if cache_key in _geo_cache:
            cached = _geo_cache[cache_key]
            if cached:
                return cached
            continue
        try:
            location = _geocode_rate(
                f"{cand}, India", exactly_one=True, timeout=10
            )
            if location and not _is_broad(location.address):
                addr_lower = location.address.lower()
                if addr_lower.strip() in BROAD_REJECT or addr_lower.startswith("india"):
                    _geo_cache[cache_key] = None
                    continue
                result = {
                    "lat": location.latitude,
                    "lon": location.longitude,
                    "matched_name": location.address.split(",")[0].strip() + ", India",
                    "location_type": "geocoded",
                    "geo_confidence": "medium",
                    "geo_method": "nominatim",
                }
                _geo_cache[cache_key] = result
                _save_geocode_cache(_geo_cache)
                return result
            else:
                _geo_cache[cache_key] = None
        except Exception:
            continue

    return {
        "lat": None, "lon": None,
        "matched_name": None,
        "location_type": None,
        "geo_confidence": "none",
        "geo_method": None,
    }

# =============================================================
# AOI + DATE WINDOW
# =============================================================

BUFFER_BY_TYPE = {
    "national_park": 0.025,
    "tiger_reserve": 0.025,
    "place":         0.025,
    "city":          0.05,
    "district":      0.15,
    "region":        0.2,
    "state":         0.5,
    "geocoded":      0.05,
}
DEFAULT_BUFFER = 0.1
PRECISE_TYPES = {
    "national_park", "tiger_reserve", "city", "district", "place", "geocoded"
}


def _build_aoi(lat: float, lon: float, location_type: Optional[str]) -> dict:
    buf = BUFFER_BY_TYPE.get(location_type or "", DEFAULT_BUFFER)
    return {
        "precision_tier": "precise" if location_type in PRECISE_TYPES else "broad",
        "aoi_buffer_deg": buf,
        "aoi_min_lat": round(lat - buf, 6),
        "aoi_max_lat": round(lat + buf, 6),
        "aoi_min_lon": round(lon - buf, 6),
        "aoi_max_lon": round(lon + buf, 6),
        "aoi_buffer_km_approx": round(buf * 111, 1),
    }


def _date_window(published_at: str, days: int = 90) -> dict:
    try:
        dt = datetime.fromisoformat(
            published_at.replace("Z", "+00:00")
        ).replace(tzinfo=None)
    except Exception:
        dt = datetime.utcnow()
    return {
        "event_date": dt.strftime("%Y-%m-%d"),
        "date_before": (dt - timedelta(days=days)).strftime("%Y-%m-%d"),
        "date_after":  (dt + timedelta(days=days)).strftime("%Y-%m-%d"),
    }

# =============================================================
# MAIN PIPELINE ENTRY POINT
# =============================================================

# =============================================================
# KEYWORD EXTRACTION
# =============================================================

# All forest-related keyword labels and their canonical tag names
KEYWORD_TAGS: List[tuple] = [
    # (search phrase in text,  tag label)
    ("forest fire",               "Forest Fire"),
    ("wildfire",                  "Wildfire"),
    ("deforestation",             "Deforestation"),
    ("forest loss",               "Forest Loss"),
    ("tree cover loss",           "Tree Cover Loss"),
    ("illegal logging",           "Illegal Logging"),
    ("tree felling",              "Tree Felling"),
    ("tree cutting",              "Tree Cutting"),
    ("trees felled",              "Tree Felling"),
    ("trees cut",                 "Tree Cutting"),
    ("timber smuggling",          "Timber Smuggling"),
    ("encroachment",              "Encroachment"),
    ("forest encroachment",       "Forest Encroachment"),
    ("forest diversion",          "Forest Diversion"),
    ("forest clearance",          "Forest Clearance"),
    ("compensatory afforestation","Afforestation"),
    ("afforestation",             "Afforestation"),
    ("forest degradation",        "Forest Degradation"),
    ("reserved forest",           "Reserved Forest"),
    ("protected forest",          "Protected Forest"),
    ("mining in forest",          "Mining"),
    ("coal block",                "Coal Block"),
    ("forest rights",             "Forest Rights"),
    ("canopy loss",               "Canopy Loss"),
    ("eco-sensitive zone",        "Eco-Sensitive Zone"),
    ("wildlife corridor",         "Wildlife Corridor"),
    ("national park",             "National Park"),
    ("tiger reserve",             "Tiger Reserve"),
    ("wildlife sanctuary",        "Wildlife Sanctuary"),
    ("forest department",         "Forest Department"),
    ("forest survey of india",    "FSI Report"),
    ("fsi report",                "FSI Report"),
    ("green cover",               "Green Cover"),
    ("forest cover",              "Forest Cover"),
]

# Event type → canonical tag
EVENT_TYPE_TAG = {
    "WILDFIRE":          "Wildfire",
    "DEFORESTATION":     "Deforestation",
    "TREE_FELLING":      "Tree Felling",
    "ENCROACHMENT":      "Encroachment",
    "FOREST_DIVERSION":  "Forest Diversion",
    "GENERAL":           "General",
}


def _extract_keywords(text: str, event_type: str, geo: dict) -> List[str]:
    """Return a deduplicated list of keyword tags for an article."""
    tags: List[str] = []
    seen: set = set()

    def _add(tag: str):
        t = tag.strip()
        if t and t.lower() not in seen:
            seen.add(t.lower())
            tags.append(t)

    # 1. Event-type tag first
    _add(EVENT_TYPE_TAG.get(event_type, event_type.title()))

    # 2. Keyword matches from text
    text_lower = text.lower()
    for phrase, label in KEYWORD_TAGS:
        if phrase in text_lower:
            _add(label)

    # 3. Location tags
    matched_name = geo.get("matched_name") or ""
    if matched_name:
        # Full resolved name (e.g. "Wayanad, Kerala")
        _add(matched_name)
        # Also add just the state part if present (after last comma)
        parts = [p.strip() for p in matched_name.split(",")]
        if len(parts) >= 2:
            _add(parts[-1])   # state
        if len(parts) >= 1 and parts[0] != matched_name:
            _add(parts[0])    # district / park name

    return tags


def generate_real_time_incidents() -> List[dict]:
    print("=" * 60)
    print("FOREST EVENT NLP PIPELINE — improved v2")
    print("=" * 60)

    # --- Collect ---
    raw_articles = _collect_from_rss(ALL_FEEDS)
    articles_data = [a.to_dict() for a in raw_articles]

    # --- Clean ---
    cleaned = [c for a in articles_data if (c := _clean_article(a))]

    # --- Deduplicate ---
    unique = _deduplicate(cleaned)
    print(f"After dedup: {len(unique)} (from {len(cleaned)})")

    # --- Relevance filter ---
    relevant = []
    for art in unique:
        is_rel, score, _ = _is_relevant(art)
        if is_rel:
            art["relevance_score"] = score
            relevant.append(art)
    print(f"Relevant articles: {len(relevant)}")

    # --- NLP + geocoding ---
    final_incidents = []
    fetched_full = 0

    for i, art in enumerate(relevant[:MAX_PROCESS]):
        if len(final_incidents) >= MAX_INCIDENTS:
            break
        try:
            text = (
                art.get("title", "") + " "
                + art.get("description", "") + " "
                + art.get("content", "")
            )
            # Fetch full text when the RSS snippet is thin
            if len(text.strip()) < 300:
                full = _fetch_full_text(art["url"])
                if full:
                    text = art.get("title", "") + " " + full
                    fetched_full += 1

            event_type = _detect_event_type(text)
            geo = _resolve_location(text, title=art.get("title", ""))

            if geo["lat"] is None:
                continue

            aoi = _build_aoi(geo["lat"], geo["lon"], geo["location_type"])
            dates = _date_window(art.get("published_at", ""))
            keywords = _extract_keywords(text, event_type, geo)

            # Preserve old frontend-compatible fields + new enriched fields
            incident = {
                "id": len(final_incidents) + 1,
                "article_id": art.get("article_id"),
                "title": art.get("title"),
                "source": art.get("source"),
                "date": dates["event_date"],
                "incident_type": event_type,
                "event_type": event_type,
                "location": [geo["matched_name"]] if geo["matched_name"] else [],
                "coordinates": {
                    "lat": geo["lat"],
                    "lon": geo["lon"],
                },
                "matched_name": geo["matched_name"],
                "location_type": geo["location_type"],
                "geo_confidence": geo["geo_confidence"],
                "geo_method": geo["geo_method"],
                "relevance_score": art.get("relevance_score"),
                "keywords": keywords,
                "precision_tier": aoi["precision_tier"],
                "aoi_buffer_km_approx": aoi["aoi_buffer_km_approx"],
                "aoi_min_lat": aoi["aoi_min_lat"],
                "aoi_max_lat": aoi["aoi_max_lat"],
                "aoi_min_lon": aoi["aoi_min_lon"],
                "aoi_max_lon": aoi["aoi_max_lon"],
                "date_before": dates["date_before"],
                "date_after": dates["date_after"],
                "url": art.get("url"),
            }
            final_incidents.append(incident)

        except Exception as e:
            print(f"  Pipeline error on article {i+1}: {e}")

        time.sleep(0.2)

    # --- Save ---
    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(final_incidents, f, indent=2, ensure_ascii=False)

    print("=" * 60)
    print(f"Total incidents : {len(final_incidents)}")
    print(f"Full-text fetched: {fetched_full}")
    precise = sum(1 for x in final_incidents if x.get("precision_tier") == "precise")
    print(f"Precise / Broad : {precise} / {len(final_incidents) - precise}")
    print(f"Saved → {OUTPUT_JSON}")
    print("=" * 60)

    return final_incidents