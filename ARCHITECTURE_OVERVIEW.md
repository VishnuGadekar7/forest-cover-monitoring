# 🌿 Forest Cover Monitoring System - Architecture Overview

## 1. Project Purpose & Overview

**ForestMonitor** is a research-grade Earth Observation (EO) platform designed for AI-powered forest change detection and analysis. It leverages deep learning semantic segmentation models to detect and monitor pixel-wise forest cover changes across multi-temporal satellite imagery.

**Key Purpose:**
- Detect forest loss (deforestation) and forest gain (reforestation)
- Analyze changes between two satellite image timestamps (T1 and T2)
- Provide change statistics and visualizations using standard remote sensing conventions
- Support both manual image uploads and automated STAC (SpatioTemporal Asset Catalog) queries from Sentinel-2
- Integrate real-time forest-related news and incidents

**Target Audience:** Researchers, environmental scientists, ISRO faculty presentations

---

## 2. Main Entry Points & Application Flow

### Backend Application Entry Point

**[backend/app/main.py](backend/app/main.py)** creates the FastAPI application with the following startup sequence:

```
FastAPI Initialization
  ↓
CORS Middleware Setup (allow localhost:3000)
  ↓
Static File Mounting (/static → serve change maps, metadata, news data)
  ↓
Model Warm-up (loads default AI model into GPU/CPU memory)
  ↓
Pre-compute News Pipeline (fetch and cache forest incidents)
  ↓
Route Registration (detection, export, news, historic)
  ↓
Ready to accept requests on port 8000
```

**Lifespan Management:**
- **Startup:** Warm up inference model with dummy tensor (prevents first-request latency)
- **Shutdown:** Graceful cleanup of model resources

### Frontend Application Flow

**[frontend/src/app/page.tsx](frontend/src/app/page.tsx)** - Landing page
→ User navigates to `/upload` for analysis
→ [frontend/src/app/upload/page.tsx](frontend/src/app/upload/page.tsx)
→ User selects tab: Manual Upload / Automated Query / Snow Detection
→ Submits images/parameters to backend
→ Results saved to localStorage
→ Redirect to `/results?id={task_id}`
→ [frontend/src/app/results/page.tsx](frontend/src/app/results/page.tsx) - displays change map + statistics

### Complete Data Flow (Change Detection)

```
User Upload (Frontend)
  ↓ multipart/form-data
POST /api/v1/detect-change
  ↓
Validation (image dimensions, format, max pixel limit 50M)
  ↓
Load Images (Rasterio for GeoTIFF, PIL for standard formats)
  ↓
Inference Service (run T1 and T2 through segmentation model)
  ↓
Generate Change Map (pixel-wise comparison: loss/gain/stable/non-forest)
  ↓
Compute Statistics (area calculations, percentages)
  ↓
Save Results to /static/change_maps/{task_id}/ (PNG images)
  ↓
Return JSON response with URLs and statistics
  ↓
Frontend displays results with Leaflet map visualization
```

---

## 3. Key Dependencies & Frameworks Used

### Backend Stack

#### Web Framework & API
- **FastAPI** (v0.110.0) - Modern async web framework with automatic Swagger UI
- **Uvicorn** (v0.29.0) - ASGI server
- **Python-multipart** - Handle file uploads

#### Deep Learning & Computer Vision
- **PyTorch** (v2.2.2) - Primary tensor framework for segmentation models
- **TensorFlow** (v2.17.0) - Used for Keras attention_unet model
- **Torchvision** (v0.17.2) - Vision utilities
- **OpenCV** (v4.9.0.80) - Image processing operations
- **scikit-image** (v0.22.0) - Advanced image algorithms
- **Pillow** (v10.3.0) - Image I/O and manipulation
- **segmentation-models-pytorch** (v0.3.3) - Pre-built segmentation architectures (ResNet encoders)
- **timm** - Vision transformer models for advanced architectures

#### Geospatial Processing
- **Rasterio** (v1.5.0) - Read/write GeoTIFF and raster data, critical for satellite imagery
- **Boto3** - AWS S3 integration for remote data access
- **pystac-client** (v0.9.0) - Query STAC catalogs (AWS Element84 Earth Search)
- **Shapely, pyproj, geopandas** (commented out but available) - Vector operations

#### Data & ML Utilities
- **NumPy** (v1.26.4) - Numerical computing
- **Pandas** (v2.0.3) - Tabular data manipulation
- **Pydantic** (v2.6.4) - Data validation and schemas
- **Matplotlib** (v3.8.0) - Plotting and visualization

#### NLP & News Pipeline
- **spaCy** - NLP for entity extraction in forest incident analysis
- **feedparser** - Parse RSS feeds from news sources
- **geopy** - Geocoding and location services
- **sentinelhub** - Sentinel Hub integration (commented but available)
- **requests** - HTTP client for API calls

#### Environment & Configuration
- **python-dotenv** - Load .env files for API keys (GNews, Sentinel Hub)
- **httpx** - Async HTTP client
- **aiofiles** - Async file I/O

### Frontend Stack

#### Framework & Build
- **Next.js** (v16.2.6) - React full-stack framework with SSR
- **React** (v19.2.3) - UI component library
- **TypeScript** (v5) - Type-safe JavaScript
- **Tailwind CSS** (v4) - Utility-first CSS framework

#### UI & Visualization
- **Leaflet** (v1.9.4) - Interactive mapping for change map visualization
- **Lucide React** - Icon library
- **Recharts** (v3.7.0) - Chart library for statistics visualization
- **Framer Motion** (v12.34.3) - Animation library (scroll animations, transitions)
- **Geoman-io/Leaflet Geoman** - Drawing tools on maps for STAC queries

#### HTTP & State
- **Axios** (v1.16.1) - HTTP client
- **clsx, tailwind-merge** - CSS utility helpers

---

## 4. Data Pipeline Overview

### Phase 1: Image Input & Validation

**Location:** [backend/app/routes/detection.py](backend/app/routes/detection.py) - `validate_image_dimensions_safe()`

```
Uploaded Files (T1, T2)
  ↓
Read header (4MB chunk, fall back to 16MB if needed)
  ↓
Extract dimensions:
  - For GeoTIFF: Use Rasterio to read spatial metadata (CRS, transform)
  - For PNG/JPG: Use PIL to read dimensions
  ↓
Validation:
  ✓ Check image dimensions match (same W×H)
  ✓ Enforce MAX_PIXEL_LIMIT (50,000,000 pixels)
  ✓ Extract CRS and geotransform
  ↓
Save metadata JSON → /static/metadata/{task_id}.json
```

**Key Constants:**
- `MAX_PIXEL_LIMIT = 50,000,000` pixels (~7000×7000 boundary)
- Prevents out-of-memory crashes on large files

### Phase 2: Image Loading & Preprocessing

**Location:** [backend/app/routes/detection.py](backend/app/routes/detection.py) - `load_image_to_numpy()`

```
Image Bytes → Numpy Array (H, W, C)
  ↓
Route by format:
  - GeoTIFF: Rasterio opens, extracts bands, handles CRS
  - Standard (PNG/JPG): PIL opens, converts to RGB/RGBA
  ↓
Ensure 4-channel format (RGBN or RGBA):
  - If 3-channel (RGB): duplicate green as NIR
  - If single-channel: replicate to 4 channels
  ↓
Optional tiling (if image > 7000×7000):
  Split into 512×512 patches
  Process patches independently
  Stitch results back together
```

### Phase 3: Inference Pipeline

**Location:** [backend/app/services/inference_service.py](backend/app/services/inference_service.py)

```
Input: Numpy array (H, W, 4) [R, G, B, NIR]
  ↓
Route by model type:
  
  KERAS (attention_unet):
    - Resize to 512×512 if needed
    - Apply contrast stretching (2%-98% percentile normalization)
    - Optional NDVI veto (exclude water/non-vegetation)
    - XLA JIT compile for speed
    - Inference → sigmoid output → threshold at 0.5
    
  PYTORCH (resnet_unet, transnet):
    - Normalize to [0, 1]
    - Forward pass through model
    - Apply sigmoid + threshold
  ↓
Output: Binary mask (H, W) uint8 {0, 1}
  - 1 = forest, 0 = non-forest
```

**Model Options:**
1. **Attention U-Net** (default) - Keras, fast, good accuracy
2. **ResNet-34 U-Net** - PyTorch, encoder-decoder with ResNet backbone
3. **Trans U-Net** - Transformer-based, requires 512×512 input
4. **TransNet** - Lightweight transformer variant

### Phase 4: Change Detection

**Location:** [backend/app/services/change_detection.py](backend/app/services/change_detection.py)

```
Inputs: mask_t1 (H, W), mask_t2 (H, W)
  ↓
Pixel-wise comparison:
  
  Loss    = (T1=1) & (T2=0) → RED [220, 50, 47]
  Gain    = (T1=0) & (T2=1) → GREEN [40, 167, 69]
  Stable  = (T1=1) & (T2=1) → WHITE [255, 255, 255]
  NonForest = (T1=0) & (T2=0) → DARK [20, 20, 30]
  ↓
Output: RGB change map (H, W, 3) uint8
        + class masks dict {loss, gain, stable_forest}
```

### Phase 5: Statistics Computation

**Location:** [backend/app/utils/metrics.py](backend/app/utils/metrics.py)

```
Compute from class masks:
  - Area lost (loss_mask.sum() * pixel_area)
  - Area gained (gain_mask.sum() * pixel_area)
  - Stable forest area
  - Percentage changes
  - Confidence metrics
```

### Phase 6: Save Results

**Location:** [backend/app/utils/visualization.py](backend/app/utils/visualization.py)

```
Output saved to /static/change_maps/{task_id}/:
  - change.png (RGB change map)
  - t1_mask.png (T1 forest segmentation)
  - t2_mask.png (T2 forest segmentation)
  - overlay.png (blended visualization)
  
Metadata saved to /static/metadata/{task_id}.json:
  - CRS, transform, spatial bounds
```

---

## 5. Machine Learning Models

### Model Registry

**[backend/app/services/model_loader.py](backend/app/services/model_loader.py)** manages model loading via singleton pattern.

| Model | Framework | Input Size | Architecture | Status |
|-------|-----------|-----------|--------------|--------|
| **attention_unet** | Keras | 512×512 | Attention gates + U-Net | ✅ Default |
| **resnet_unet** | PyTorch | 512×512 | ResNet-34 encoder | ✅ Available |
| **transnet** | PyTorch | 512×512 | Transformer backbone | ✅ Available |
| **trans_unet** | PyTorch | 512×512 | TransUNet (Vision Transformer) | ✅ Available |

### Attention U-Net (Default)

**[backend/app/models/attention_unet.py](backend/app/models/attention_unet.py)**

Architecture: Encoder-Decoder with Attention Gates

```
Input (4, 512, 512)
  ↓
Encoder (downsampling with max pooling):
  Conv Block (4 → 64) → Conv Block (64 → 128) → Conv Block (128 → 256) → Conv Block (256 → 512)
  ↓
Bottleneck:
  Conv Block (512 → 1024)
  ↓
Decoder (upsampling with Attention Gates):
  UpConv (1024 → 512) + AttentionGate + skip → Conv Block
  UpConv (512 → 256) + AttentionGate + skip → Conv Block
  UpConv (256 → 128) + AttentionGate + skip → Conv Block
  UpConv (128 → 64) + AttentionGate + skip → Conv Block
  ↓
Final Conv (64 → 1)
  ↓
Sigmoid activation
  ↓
Output (1, 512, 512) → threshold at 0.5 → Binary mask
```

**Key Feature:** Attention gates learn where to focus on salient regions, suppressing irrelevant background.

### Model Warm-up

On startup, `main.py` runs a dummy forward pass:

```python
dummy_input = np.zeros((512, 512, 4), dtype=np.float32)
inference.predict(dummy_input)
```

This prevents first-request latency by pre-allocating GPU/CPU memory and compiling the model.

### Weight Files

Located in `backend/weights/`:
- `attention_unet_best.h5` - Keras model (loaded as default)
- `resnet_unet.pth` - PyTorch state dict
- `trans_unet.pth` - PyTorch state dict

Weights are **optional** — models run with random initialization if weights missing.

---

## 6. API Endpoints & Purposes

### Change Detection Endpoints

#### **POST /api/v1/detect-change** *(Manual Upload)*
**Purpose:** Detect forest change from manually uploaded satellite images

**Request:**
```
multipart/form-data:
  - image_t1: File (satellite image at time 1)
  - image_t2: File (satellite image at time 2)
  - (optional) model_name: str (attention_unet | resnet_unet | transnet)
  - (optional) inference settings (threshold, NDVI veto, contrast stretch, etc.)
```

**Response:**
```json
{
  "id": "uuid-task-id",
  "t1_mask_url": "/static/change_maps/{id}/t1_mask.png",
  "t2_mask_url": "/static/change_maps/{id}/t2_mask.png",
  "change_map_url": "/static/change_maps/{id}/change.png",
  "overlay_url": "/static/change_maps/{id}/overlay.png",
  "statistics": {
    "forest_loss_hectares": 123.45,
    "forest_gain_hectares": 23.10,
    "loss_percentage": 12.5,
    "gain_percentage": 2.3
  },
  "timestamp": 1718361600000
}
```

**Processing:**
- Validates image dimensions and pixel limits
- Extracts spatial metadata (CRS, transform)
- Loads images as numpy arrays
- Runs inference on both images
- Generates change map
- Computes statistics
- Saves visualizations

---

#### **POST /api/v1/detect-change-automated** *(STAC Query)*
**Purpose:** Automatically fetch satellite images from Sentinel-2 via STAC and detect changes

**Request:**
```json
{
  "bbox": [minLon, minLat, maxLon, maxLat],
  "date_t1": "2022-01-01/2022-03-31",
  "date_t2": "2023-01-01/2023-03-31",
  "max_cloud_cover": 20,
  "model_name": "attention_unet"
}
```

**Backend Flow:**
```
Query STAC Catalog (AWS Element84)
  ↓
Find Sentinel-2 L2A tiles matching bbox + date range + cloud cover
  ↓
Stream Red, Green, Blue, NIR bands from AWS S3 (no auth needed)
  ↓
Merge 4 bands into single array
  ↓
Run detection pipeline
  ↓
Return change map + statistics
```

---

#### **POST /api/v1/detect-forest-snow** *(Snow Detection)*
**Purpose:** Distinguish forest-covered areas from snow for accurate change detection

**Request:** Same as `/detect-change` + snow detection flags

**Processing:**
- Runs standard detection
- Applies additional snow/cloud masking
- Filters out false positives from snow cover

---

### Export Endpoints

#### **GET /api/v1/export-tif**
**Purpose:** Download change map as a GeoTIFF file with geospatial metadata

**Query Parameters:**
```
task_id: str (UUID from detection response)
```

**Response:** 
- Content-Type: `image/tiff`
- 16-bit RGB GeoTIFF with embedded CRS and transform

**Processing:**
1. Loads PNG change map from disk
2. Reads spatial metadata JSON
3. Converts to rasterio-compatible format
4. Embeds CRS (if available) and transform matrix
5. Streams binary GeoTIFF back to client

---

### News Endpoints

#### **GET /news**
**Purpose:** Fetch real-time forest-related incidents and news

**Query Parameters:**
```
limit: int (1-50, default 10)
offset: int (default 0)
```

**Response:**
```json
{
  "data": [
    {
      "title": "Major deforestation in Amazon",
      "description": "...",
      "location": "Brazil",
      "coordinates": [lat, lon],
      "source": "GNews",
      "date": "2024-06-14"
    }
  ],
  "total": 150,
  "offset": 0,
  "limit": 10,
  "has_more": true
}
```

**Caching:** 6-hour TTL; refreshes automatically if cache expires

---

#### **GET /historic-news**
**Purpose:** Fetch historical forest incidents (pre-computed at startup)

**Response:** Array of historic incidents with coordinates and details

---

### Health Check

#### **GET /health**
**Purpose:** Verify API is running

**Response:**
```json
{ "status": "ok" }
```

---

## 7. Frontend Structure & Components

### Page Structure

```
frontend/src/app/
  ├── page.tsx           (Landing page)
  ├── upload/page.tsx    (Upload & analysis interface)
  ├── results/page.tsx   (Results display)
  ├── history/page.tsx   (Prediction history)
  ├── news/page.tsx      (Forest news feed)
  └── globals.css        (Global styles)
```

### Key Components

#### **[Navbar.tsx](frontend/src/components/Navbar.tsx)**
- Navigation across pages
- Branding with ForestMonitor logo
- Links to Launch App, History, Documentation

#### **[UploadCard.tsx](frontend/src/components/UploadCard.tsx)**
- Drag-and-drop zone for image upload
- File preview with size information
- Clear button to reset selection
- Validates file format (PNG, JPG, TIFF)

#### **[ChangeMap.tsx](frontend/src/components/ChangeMap.tsx)**
- Interactive Leaflet map
- Change map overlay on CartoDB dark basemap
- Change legend (Loss/Gain/Stable/Non-forest colors)
- Bounds fitted to image extent
- Opacity controls

#### **[STACMap.tsx](frontend/src/components/STACMap.tsx)**
- Leaflet map with drawing tools (Geoman)
- Draw bounding box for STAC query
- Select date range (T1 and T2)
- Trigger automated detection

#### **[AdvancedSettings.tsx](frontend/src/components/AdvancedSettings.tsx)**
- Inference parameters
- Model selection
- Threshold adjustment
- NDVI veto toggle
- Contrast stretching options

#### **[ForestChart.tsx](frontend/src/components/ForestChart.tsx)**
- Recharts-based statistics visualization
- Area lost/gained bar charts
- Percentage breakdown

#### **[StatCard.tsx](frontend/src/components/StatCard.tsx)**
- Key metric displays
- Forest loss area
- Forest gain area
- Confidence scores

### Frontend Pages Details

#### **Landing Page** ([page.tsx](frontend/src/app/page.tsx))
- Hero section with gradient background
- Feature cards (Advanced Segmentation, Geographic Integration, Research Grade)
- CTAs to Upload page
- Parallax background effects (gradient blurs)
- Conditional History button based on localStorage

#### **Upload Page** ([upload/page.tsx](frontend/src/app/upload/page.tsx))
- **3 Tabs:**
  1. **Manual Upload** - Drag-drop two satellite images
  2. **Automated Query** - Draw bbox on map, select dates, fetch from STAC
  3. **Snow Detection** - Special mode for snow cover handling

- Model selector dropdown
- Advanced settings accordion
- Upload progress bar
- Error handling with user-friendly messages
- Results saved to `localStorage.prediction_history` (capped at 100 items)

#### **Results Page** ([results/page.tsx](frontend/src/app/results/page.tsx) + [results-client.tsx](frontend/src/app/results/results-client.tsx))
- Retrieve result from localStorage via URL query param
- Display:
  - Change map visualization (Leaflet)
  - Original masks (T1, T2)
  - Statistics cards
  - Download GeoTIFF button
  - Share/copy result ID button

#### **History Page** ([history/page.tsx](frontend/src/app/history/page.tsx))
- List all previous analyses from localStorage
- Timestamps and summary statistics
- Navigate to specific result
- Delete individual results
- Clear entire history

#### **News Page** ([news/page.tsx](frontend/src/app/news/page.tsx))
- Real-time forest incidents feed
- Paginated list of incidents
- Incident cards with location, source, date
- Interactive map showing incident locations
- Filter/search options

### Frontend State Management

**Local Storage** (`prediction_history`)
- Stores all detection results locally
- Array of ChangeDetectionResult objects
- Capped at 100 items (FIFO eviction)
- Persists across browser sessions
- Used to build History and Results pages

### Frontend Styling

- **Tailwind CSS v4** - Utility-first responsive design
- **Dark theme** - `bg-[#0a0f1a]` dark background
- **Glass-morphism** - Semi-transparent cards with blur effects
- **Animations** - Framer Motion for scroll-triggered reveals
- **Gradients** - Green-to-blue theme (forest + tech)
- **Icons** - Lucide React icons throughout

### Frontend API Integration

**[lib/api.ts](frontend/src/lib/api.ts)** - HTTP client wrapper

```typescript
async detectChange(file1, file2, model, settings, onProgress)
  → POST /api/v1/detect-change

async detectChangeAutomated(stacQuery)
  → POST /api/v1/detect-change-automated

async detectForestSnow(file1, file2, model, settings, onProgress)
  → POST /api/v1/detect-forest-snow

async exportGeoTIFF(taskId)
  → GET /api/v1/export-tif?task_id={taskId}

async fetchNews(limit, offset)
  → GET /news

async assetUrl(path)
  → Construct full URL to static assets
```

---

## 8. Data Flow Diagrams

### Complete User Journey

```
User arrives at landing page
  ↓
Click "Start Monitoring"
  ↓ → /upload
Choose tab (Manual/Automated/Snow)
  ↓
Manual:  Upload 2 images → Click "Detect Change" → Inference
Auto:    Draw bbox, select dates → STAC search → Inference
Snow:    Upload 2 images + snow settings → Inference
  ↓
POST /api/v1/detect-change (or variant)
  ↓ Backend processes...
Results saved to localStorage
  ↓
Redirect → /results?id={uuid}
  ↓
Display change map + statistics
  ↓
User can:
  - View detailed masks
  - Export GeoTIFF
  - Share result
  - Browse history
```

### Model Inference Pipeline

```
Input Images (manual or STAC)
  ↓
Validation (dimensions, format, pixel limit)
  ↓
Load as numpy arrays
  ↓
Create task directory /static/change_maps/{task_id}/
  ↓
Route image through selected model:
  ├─ Attention U-Net (Keras) → XLA compiled
  ├─ ResNet U-Net (PyTorch)
  ├─ TransNet (PyTorch)
  └─ Trans U-Net (Vision Transformer)
  ↓
Generate binary masks for T1 and T2
  ↓
Pixel-wise change map generation
  ↓
Compute statistics (area, percentage)
  ↓
Visualize and save PNGs
  ↓
Return JSON with URLs + statistics to frontend
  ↓
Frontend displays interactive results
```

### STAC Integration

```
User defines search criteria (bbox, date_t1, date_t2, cloud cover)
  ↓
Backend: pystac_client queries AWS Element84 Earth Search
  ↓
STAC search returns Sentinel-2 L2A items
  ↓
Sort by cloud cover, select best match
  ↓
Extract Red, Green, Blue, NIR band HREFs from item assets
  ↓
Rasterio streams bands from AWS S3 (no authentication)
  ↓
Merge 4 bands → (H, W, 4) numpy array
  ↓
Send to inference pipeline (same as manual upload)
  ↓
Generate change map + statistics
```

---

## 9. File Organization Summary

### Backend
```
backend/
├── app/
│   ├── main.py                      (FastAPI app, lifespan, routing)
│   ├── models/                      (Segmentation architectures)
│   │   ├── attention_unet.py       (Default model)
│   │   ├── resnet_unet.py
│   │   ├── transnet.py
│   │   └── trans_unet.py
│   ├── routes/                      (API endpoints)
│   │   ├── detection.py            (POST /detect-change)
│   │   ├── export.py               (GET /export-tif)
│   │   ├── news.py                 (GET /news)
│   │   └── historic.py             (GET /historic-news)
│   ├── services/                    (Business logic)
│   │   ├── inference_service.py    (Run models)
│   │   ├── model_loader.py         (Singleton model loading)
│   │   ├── change_detection.py     (Pixel comparison)
│   │   ├── stac_service.py         (STAC queries)
│   │   ├── news_pipeline.py        (News aggregation)
│   │   └── historic_pipeline.py    (Historic incidents)
│   ├── schemas/                     (Pydantic models)
│   │   ├── detection.py
│   │   └── stac_query.py
│   └── utils/                       (Utilities)
│       ├── image_preprocessing.py
│       ├── metrics.py              (Statistics)
│       └── visualization.py         (Save images)
├── weights/                         (Model weights)
│   ├── attention_unet_best.h5
│   ├── resnet_unet.pth
│   └── trans_unet.pth
├── static/                          (Served files)
│   ├── change_maps/                (Results per task)
│   ├── metadata/                   (Geospatial metadata)
│   └── news_data/                  (Cached incidents)
├── requirements.txt
├── model_config.json
└── main.py (entry point)
```

### Frontend
```
frontend/
├── src/
│   ├── app/
│   │   ├── page.tsx                (Landing)
│   │   ├── globals.css
│   │   ├── layout.tsx              (Root layout)
│   │   ├── upload/page.tsx         (Upload + analysis)
│   │   ├── results/page.tsx        (Results container)
│   │   ├── results/results-client.tsx (Results logic)
│   │   ├── history/page.tsx        (History)
│   │   └── news/page.tsx           (News feed)
│   ├── components/
│   │   ├── Navbar.tsx
│   │   ├── UploadCard.tsx
│   │   ├── ChangeMap.tsx           (Leaflet map)
│   │   ├── STACMap.tsx             (Drawing + STAC)
│   │   ├── AdvancedSettings.tsx
│   │   ├── ForestChart.tsx
│   │   └── StatCard.tsx
│   └── lib/
│       └── api.ts                  (HTTP client)
├── public/
├── package.json
└── next.config.js
```

---

## 10. Key Technology Decisions

1. **FastAPI + Uvicorn** - Async-native, high performance, automatic Swagger UI
2. **PyTorch + Keras** - Flexibility to support multiple model formats (PyTorch + Keras)
3. **Rasterio** - Robust geospatial I/O, handles GeoTIFF metadata seamlessly
4. **STAC + pystac-client** - Standard catalog format, no authentication for public Sentinel-2 data
5. **Leaflet** - Lightweight, mature mapping library without heavy dependencies
6. **Next.js** - Full-stack React, SSR support, easy file routing
7. **Local Storage for History** - Simple, persists across sessions, no backend DB needed
8. **Model Warm-up on Startup** - Eliminates first-request latency spike
9. **Tiling for Large Images** - Process 512×512 patches independently, stitch results (memory efficient)
10. **Attention Gates in U-Net** - Improves segmentation of small forest patches

---

## 11. Performance Considerations

- **Max Image Size:** 50M pixels (~7000×7000) to prevent OOM
- **Model Inference:** 1-2s per 512×512 tile on GPU
- **STAC Queries:** ~5-10s (includes S3 streaming)
- **News Caching:** 6-hour TTL to avoid redundant API calls
- **Model Warm-up:** ~2-5s on startup (XLA compilation)
- **Static Files:** CDN-ready (Leaflet maps, change PNGs)
- **Frontend:** Lazy-loaded components, no initial bundle bloat

---

## 12. Security & Validation

- **CORS:** Only allow localhost:3000 (dev) — configure for production
- **Image Validation:** Header inspection, format verification, dimension checks
- **Pixel Limit Enforcement:** Prevent DoS attacks via massive uploads
- **Metadata Sanitization:** CRS/transform validated before use
- **Error Messages:** User-friendly, no internal stack traces exposed
- **Static Files:** Mounted read-only, no arbitrary access

---

This architecture enables researchers to quickly analyze satellite imagery for forest change detection while maintaining research-grade accuracy and geospatial compliance.
