# 🌿 Forest Cover Monitoring & Change Detection System

> Research-grade Earth Observation platform for AI-powered forest change analysis.  
> Built for ISRO faculty presentation.

---

## Architecture

```
Frontend (Next.js + Tailwind)
        ↓  multipart form upload
FastAPI Backend (Python 3.11)
        ↓
AI Inference Service (PyTorch)
        ↓
Change Detection Engine (NumPy / OpenCV)
        ↓
Visualization + Static File Serving
```

## Prerequisites

- **Python**: 3.11 only
- **Node.js**: 18.x or higher (LTS recommended)
- **System Dependencies**:
  - GDAL binaries (required for `rasterio` on Linux/macOS: `sudo apt-get install gdal-bin` or `brew install gdal`)

---
## Quick Start

### Backend

```bash
cd backend

# Create virtual environment
python -m venv .venv
.venv\Scripts\activate          # Windows
# source .venv/bin/activate     # Linux/macOS

# Install dependencies
pip install -r requirements.txt

# Download spaCy language model (required for NLP incident detection)
python -m spacy download en_core_web_sm

# Configure API keys for News
cp .env.example .env
# In backend/.env, obtain the keys from GNews.io and Sentinel Hub, then replace the placeholders

# (Optional) Set environment variables
set MODEL_NAME=attention_unet   # or resnet_unet / transnet
set MODEL_WEIGHTS=weights/attention_unet.pth

# Run the server
uvicorn app.main:app --reload --port 8000
```

**Important**: Place pretrained `.pth` model weight files in `backend/weights/` before running inference. See the [Model Weights](#model-weights) section for details.

Swagger UI → [http://localhost:8000/docs](http://localhost:8000/docs)

### Frontend

```bash
cd frontend
npm install	# Install dependencies
npm run dev	# Run the development server
```

Dashboard → [http://localhost:3000](http://localhost:3000)

---

## Model Weights

Place pretrained `.pth` state-dict files in `backend/weights/`:

| Filename                  | Architecture     | Notes                        |
|---------------------------|------------------|------------------------------|
| `attention_unet.pth`      | Attention U-Net  | Default at startup           |
| `resnet_unet.pth`         | ResNet-34 U-Net  | Requires `smp` library       |
| `transnet.pth`            | TransNet (ViT)   | Input must be 512×512        |

Switch model via env: `set MODEL_NAME=resnet_unet`

---

## API Reference

### `POST /api/v1/detect-change`

**Input:** multipart/form-data

| Field      | Type | Description                  |
|------------|------|------------------------------|
| `image_t1` | File | Satellite image — Time 1     |
| `image_t2` | File | Satellite image — Time 2     |

**Response:**

```json
{
  "forest_area_t1": 1245.60,
  "forest_area_t2": 1102.30,
  "forest_loss": 183.40,
  "forest_gain": 40.10,
  "percentage_change": -11.50,
  "change_map_url": "/static/change_maps/abc123_change.png",
  "mask_t1_url": "/static/change_maps/abc123_mask_t1.png",
  "mask_t2_url": "/static/change_maps/abc123_mask_t2.png"
}
```

# Live Forest Incident Monitoring

The platform now includes a real-time forest incident intelligence module powered by live environmental news feeds and Earth Observation workflows.

## Features

- Real-time forest incident monitoring
- NLP-based wildfire and deforestation detection
- India-focused environmental news filtering
- Automatic geocoding of incident locations
- Historic EO analysis using Sentinel-2 imagery
- Before/After satellite visualization (T1/T2)
- NDVI-based vegetation change analysis
- Interactive monitoring dashboard

---

## Live News Pipeline

The backend continuously fetches and filters environmental incidents from:

- Google News RSS
- GNews API

The pipeline performs:

1. Forest-related keyword filtering
2. India-region filtering
3. Duplicate removal
4. NLP-based incident classification
5. Location extraction and geocoding
6. Coordinate generation for EO analysis

Generated incidents are stored in:

```bash
backend/static/news_data/incidents.json

News Monitoring API
GET /news

Returns latest live forest incidents.

Example response:

[
  {
    "title": "Forest Fire in Uttarakhand",
    "incident_type": "WILDFIRE",
    "location": ["Uttarakhand"],
    "coordinates": {
      "lat": 30.0668,
      "lon": 79.0193
    },
    "date": "2026-05-19"
  }
]

## Project Structure

```
forest-monitoring/
├── backend/
│   ├── app/
│   │   ├── main.py                      # FastAPI app entry point
│   │   ├── routes/detection.py          # POST /api/v1/detect-change
│   │   ├── services/
│   │   │   ├── model_loader.py          # Singleton model loading
│   │   │   ├── inference_service.py     # Forward pass → binary mask
│   │   │   └── change_detection.py      # Pixel-wise comparison + colour map
│   │   ├── models/
│   │   │   ├── attention_unet.py        # Attention U-Net architecture
│   │   │   ├── resnet_unet.py           # ResNet-34 U-Net (via smp)
│   │   │   └── transnet.py              # Transformer-based segmentation
│   │   ├── utils/
│   │   │   ├── image_preprocessing.py   # Resize, normalise, tensor conversion
│   │   │   ├── metrics.py               # Area (ha) + change statistics
│   │   │   └── visualization.py         # Save PNGs to static dir
│   │   └── schemas/detection.py         # Pydantic response schema
│   ├── weights/                         # ← Place your .pth files here
│   ├── static/change_maps/              # Generated output images
│   └── requirements.txt
│
├── frontend/
│   └── src/
│       ├── app/
│       │   ├── upload/page.tsx          # Image upload UI
│       │   └── results/page.tsx         # Results dashboard
│       ├── components/
│       │   ├── Navbar.tsx
│       │   ├── UploadCard.tsx           # Drag-and-drop uploader
│       │   ├── StatCard.tsx             # Stat display cards
│       │   ├── ForestChart.tsx          # Recharts visualisations
│       │   └── ChangeMap.tsx            # Leaflet map overlay
│       └── lib/api.ts                   # Typed API client (axios)
│
└── README.md
```

---


## Future Roadmap

- [x] Sentinel-2 automatic tile ingestion
- [x] GeoTIFF export with georeferenced masks
- [ ] Multi-temporal time series monitoring
- [ ] Deforestation alert system
- [ ] PostgreSQL + PostGIS integration
- [ ] Docker Compose deployment
- [ ] Research publication export (PDF report)
