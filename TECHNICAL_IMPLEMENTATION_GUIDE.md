# 🔧 ForestCover: Technical Implementation & Code Guide
## Deep-Dive into Code Structure, Algorithms, and Performance Considerations

---

## 1. Model Architecture Implementation Details

### 1.1 AttentionUNet (Keras/TensorFlow)

**File:** `backend/app/models/keras_unet.py`

**Architecture Breakdown:**
```python
class AttentionUNet(nn.Module):
    def __init__(self, in_channels=4, out_channels=1, 
                 features=[64, 128, 256, 512]):
        """
        Encoder depths:
          Level 0: 512×512 → features[0]=64 channels
          Level 1: 256×256 → features[1]=128 channels
          Level 2: 128×128 → features[2]=256 channels
          Level 3: 64×64 → features[3]=512 channels
          Bottleneck: 32×32 → 1024 channels
        """
```

**Forward Pass Flow:**

```
INPUT: (B, 512, 512, 4) [Batch, Height, Width, RGBN]
   ↓
ENCODER PATH (Downsampling × 4):
   ↓
   Encoder Block 1:
   ├─ ConvBlock(4→64): Conv(3×3) + BN + ReLU + Conv(3×3) + BN + ReLU
   ├─ MaxPool2d(2×2, stride=2)
   └─ Output: (B, 256, 256, 64)
   ↓ [Save skip_connection[0]]
   
   Encoder Block 2:
   ├─ ConvBlock(64→128)
   ├─ MaxPool2d(2×2, stride=2)
   └─ Output: (B, 128, 128, 128)
   ↓ [Save skip_connection[1]]
   
   Encoder Block 3:
   ├─ ConvBlock(128→256)
   ├─ MaxPool2d(2×2, stride=2)
   └─ Output: (B, 64, 64, 256)
   ↓ [Save skip_connection[2]]
   
   Encoder Block 4:
   ├─ ConvBlock(256→512)
   ├─ MaxPool2d(2×2, stride=2)
   └─ Output: (B, 32, 32, 512)
   ↓ [Save skip_connection[3]]

BOTTLENECK:
   ├─ ConvBlock(512→1024)
   └─ Output: (B, 32, 32, 1024)

DECODER PATH (Upsampling × 4):
   ↓
   Decoder Block 1:
   ├─ UpConv(1024→512): Transposed Conv 2×2 stride 2
   ├─ Output: (B, 64, 64, 512)
   ├─ AttentionGate(skip_connection[3])
   │  └─ Learn to suppress irrelevant features
   ├─ Concatenate: skip(256,256,512) + upsampled(256,256,512) → (256,256,1024)
   ├─ ConvBlock(1024→512)
   └─ Output: (B, 64, 64, 512)
   ↓
   Decoder Block 2:
   ├─ UpConv(512→256)
   ├─ AttentionGate(skip_connection[2])
   ├─ Concatenate + ConvBlock(512→256)
   └─ Output: (B, 128, 128, 256)
   ↓
   Decoder Block 3:
   ├─ UpConv(256→128)
   ├─ AttentionGate(skip_connection[1])
   ├─ Concatenate + ConvBlock(256→128)
   └─ Output: (B, 256, 256, 128)
   ↓
   Decoder Block 4:
   ├─ UpConv(128→64)
   ├─ AttentionGate(skip_connection[0])
   ├─ Concatenate + ConvBlock(128→64)
   └─ Output: (B, 512, 512, 64)

FINAL OUTPUT:
   ├─ Conv(64→1, kernel_size=1×1)
   ├─ Sigmoid activation (for binary classification)
   └─ Output: (B, 512, 512, 1) [Binary forest mask, values ∈ [0,1]]

THRESHOLD APPLICATION:
   └─ mask = (output > 0.5).astype(uint8)  # Convert to 0/1
```

**Attention Gate Mechanism (Detailed):**

```python
class AttentionGate(nn.Module):
    def forward(self, g, x):
        """
        Args:
            g: Gating signal from decoder (lower spatial resolution)
            x: Skip connection from encoder (higher spatial resolution)
        
        Returns:
            Attention-weighted skip connection
        """
        # Project both inputs to same dimension
        g1 = W_g(g)          # (B, H, W, F_int)
        x1 = W_x(x)          # (B, H, W, F_int)
        
        # Combine via element-wise addition (mimics concatenation then conv)
        combined = ReLU(g1 + x1)  # (B, H, W, F_int)
        
        # Learn attention weights (0-1 per pixel)
        alpha = Sigmoid(psi(combined))  # (B, H, W, 1)
        
        # Scale skip connection by attention weights
        return x * alpha  # Element-wise multiplication broadcasts alpha
```

**Why This Design?**
- ✅ Bottleneck (1024 channels at 32×32) concentrates semantic information
- ✅ Skip connections preserve spatial details from encoder
- ✅ Attention gates learn to suppress background noise
- ✅ Symmetric encoder-decoder allows fine-grained predictions

---

### 1.2 ResNet U-Net (PyTorch + SMP)

**File:** `backend/app/models/resnet_unet.py`

**Architecture:**
```
Uses segmentation-models-pytorch (SMP) library
Model: smp.Unet(encoder_name="resnet34", ...)

Encoder (ResNet-34):
  • Conv1: 4-channel input → 64 filters (stride 2, kernel 7×7)
  • ResLayer1: 64 filters × 3 blocks (stride 1)
  • ResLayer2: 128 filters × 4 blocks (stride 2) [2× downsampling]
  • ResLayer3: 256 filters × 6 blocks (stride 2) [4× downsampling]
  • ResLayer4: 512 filters × 3 blocks (stride 2) [8× downsampling]

Decoder (U-Net):
  • Bilinear upsampling from 8× to 1× resolution
  • Concatenate skip connections from each encoder stage
  • 4 decoder stages (mirrors encoder levels)
  • Final 1×1 conv → binary output

Total Parameters: ~26M
```

**Why ResNet Encoder?**
- Residual blocks enable deep networks (34 layers) without vanishing gradients
- Pre-training on ImageNet provides strong low-level feature extraction
- Proven architecture (top-1 accuracy 73% on ImageNet with 34-layer variant)

**4-Channel Input Adaptation:**
```python
# Standard ResNet expects 3-channel RGB
# Modified to accept 4-channel RGBN:

# Original: Conv2d(3, 64, kernel_size=7, stride=2, padding=3)
# Modified: Conv2d(4, 64, kernel_size=7, stride=2, padding=3)

# Weight adaptation:
# If loading ImageNet pretrained weights for 3-channel:
#   - Take average of R, G, B channel weights
#   - Initialize NIR channel weights to 0.5 (neutral)
#   - Fine-tune allows model to learn NIR-specific patterns
```

---

### 1.3 TransNet (Transformer-based)

**File:** `backend/app/models/transnet.py`

**Architecture (High-Level):**
```
Patch Embedding:
  Input: 512×512×4
  Patch size: 16×16 pixels
  Number of patches: (512/16) × (512/16) = 32×32 = 1024 patches
  Embedding dimension: 768

Transformer Encoder (12 blocks):
  For each block:
    ├─ LayerNorm(input)
    ├─ Multi-Head Self-Attention (12 heads, 768/12=64 dims each)
    │  └─ Learns which patches relate to each other
    ├─ Feed-forward network (3072 hidden dims)
    └─ Residual connections (x + attention(x))

Decoder:
  • Extract features from multiple transformer layers
  • Create pyramid: [patch-level, layer-6, layer-12]
  • Bilinear upsampling: 32×32 → 512×512
  • Progressive refinement with skip connections
  
Output: 512×512×1 binary mask
```

**Attention Mechanism (Simplified):**
```
Query, Key, Value projections of input patches
Attention(Q, K, V) = softmax(Q × K^T / √d) × V

In Multi-Head:
  • 12 separate attention heads
  • Each head attends to different subspaces
  • Results concatenated → linear projection → output

Why Transformers for Segmentation?
  + Global receptive field: Each patch can attend to all other patches
  + Flexible: Learns to focus on relevant patches (vs fixed CNN receptive field)
  - Slower: O(n²) complexity (n = number of patches = 1024)
```

---

### 1.4 Trans U-Net (Vision Transformer + U-Net)

**File:** `backend/app/models/trans_unet.py`

**Architecture:**
```
Combines ViT (Vision Transformer) encoder with U-Net decoder

Encoder (Segformer/MiT-B1):
  • Multi-scale patch embeddings (Overlapping: 7×7, 3×3, 3×3 patches)
  • 4 stages, each with transformer blocks + downsampling
  • Output: Hierarchical feature pyramid (Stage1→Stage4)

Decoder (U-Net-style):
  • Fuse features from all 4 encoder stages
  • Progressive upsampling with skip connections
  • 1×1 convolutions to align channel dimensions
  
Output: 512×512×1 binary mask

Key Innovation:
  • Hierarchical (vs standard ViT which flattens entire image)
  • Multi-scale features (combines fine + coarse information)
  • Efficient (fewer parameters than full ViT)
```

**Why Vision Transformers?**
- 📊 SOTA performance on semantic segmentation benchmarks (2020-2024)
- 🎯 Global context: Every pixel can influence every other pixel's prediction
- 🔬 Better boundary delineation (vs CNNs with limited receptive fields)
- ⚠️ Tradeoff: Requires massive training data + GPUs

---

## 2. Model Loading & Singleton Pattern

**File:** `backend/app/services/model_loader.py`

**Singleton Implementation:**
```python
class ModelLoader:
    _instances: dict[str, "ModelLoader"] = {}
    
    @classmethod
    def get_model(cls, model_name: str = None) -> "ModelLoader":
        """
        Lazy initialization + caching
        First call: Load model from disk
        Subsequent calls: Return cached instance
        """
        if model_name not in cls._instances:
            cls._instances[model_name] = cls(model_name)  # Expensive!
        return cls._instances[model_name]
```

**Why Singleton?**
1. **Model weights loaded once** (not per-request)
   - AttentionUNet weights: 85MB → takes 1-2 seconds to load
   - If loaded per-request → 5+ second overhead per request ❌
   
2. **GPU memory pre-allocated** (model stays in VRAM)
   - Keeps 1-3.5GB VRAM allocated continuously
   - Inference-only: No gradient computation overhead
   - Alternative: Reload per-request → causes memory fragmentation

3. **Thread-safe** (Python GIL)
   - FastAPI's async/await runs in single thread
   - Multiple coroutines share same Python GIL
   - Singleton safely shared across concurrent requests

**Model Resolution Strategy:**
```python
# Priority order for selecting weight file:
1. Check environment variable MODEL_WEIGHTS
2. Check default location: /backend/weights/<model_name>_best.h5 or .pth
3. If no weights found:
   - Log warning
   - Load model architecture only (random init)
   - Inference runs but gives garbage results (useful for debugging)
```

**Keras vs PyTorch Detection:**
```python
if weights_path.suffix in [".h5", ".keras"]:
    # Route to Keras model loading
    from app.models.keras_unet import build_keras_unet
    self.model = build_keras_unet()
    self.model.load_weights(str(weights_path))
    self.is_keras = True
else:
    # Route to PyTorch model loading
    self.model = _import_and_build(model_name, device)
    self.model.load_state_dict(torch.load(weights_path))
    self.model.eval()  # Disable batch norm updates, dropout
```

---

## 3. Inference Service Pipeline

**File:** `backend/app/services/inference_service.py`

### 3.1 Image Preprocessing Flow

**Input:** PIL Image or NumPy array (H, W, C)
**Output:** Binary mask (H, W) with values 0/1

**Step 1: Format Detection**
```python
raw_max = tf.reduce_max(tensor).numpy()

if raw_max <= 2.0:
    # Float image already normalized [0, 1] → Use as-is
    
elif raw_max <= 255.0:
    # 8-bit image (PNG/JPG) → Need to synthesize NIR
    # Formula: Excess Green Index (ExG)
    red, green, blue = tensor[..., 0:1], tensor[..., 1:2], tensor[..., 2:3]
    exg = (2.0 * green) - red - (2.0 * blue)
    fake_nir = tf.clip_by_value(green + exg, 0.0, 1.0)
    tensor = tf.concat([red*0.35, green*0.35, blue*0.5, fake_nir], axis=-1)
    
else:  # raw_max > 255
    # 16-bit satellite data → Percentile stretch
    for each channel:
        p2  = sorted_values[0.02 * N]
        p98 = sorted_values[0.98 * N]
        stretched[c] = (channel[c] - p2) / (p98 - p2)
        stretched[c] = clip(stretched[c], 0, 1)
```

**Why These Ranges?**
- ✅ 2.0 threshold: Separates float [0,1] from uint8 [0,255]
- ✅ 255.0 threshold: Separates uint8 from uint16 raw sensor data
- ✅ 2%-98% percentile: Removes outliers without destroying contrast

**Step 2: Resizing (if needed)**
```python
original_h, original_w = tensor.shape[:2]

if original_h != 512 or original_w != 512:
    tensor = tf.image.resize(tensor, [512, 512], method='bilinear')
    is_padded = True
else:
    is_padded = False
```

**Why Bilinear?**
- Fast interpolation (supported by GPU/CPU)
- Smooth results (vs Nearest-Neighbor which has jagged edges)
- Standard choice for satellite imagery

**Step 3: XLA Compilation (TensorFlow-only)**
```python
self._compiled_predict = tf.function(
    self.model,
    jit_compile=True,        # JIT compile → ~30-40% speedup
    reduce_retracing=True    # Prevent memory leaks from repeated compiles
)
```

**Why XLA?**
- XLA = Accelerated Linear Algebra
- Fuses multiple TF ops into single GPU kernel
- Example: Conv + BatchNorm + ReLU → 1 fused op (vs 3 separate ops)
- Result: ~30-40% faster inference

---

### 3.2 Model Inference

```python
def predict(self, image_data):
    if self.is_keras:
        return self._predict_keras(image_data)
    else:
        return self._predict_pytorch(image_data)

def _predict_keras(self, arr):
    """Keras-specific inference pipeline"""
    # Input: (H, W, 4) float32
    
    # Forward pass
    output = self._compiled_predict(tensor)  # (512, 512, 1)
    
    # Sigmoid already applied in model output layer
    logits = output.numpy()  # (512, 512, 1)
    
    # Thresholding
    mask = (logits > self.threshold).astype(np.uint8)  # (512, 512)
    
    return mask

def _predict_pytorch(self, arr):
    """PyTorch-specific inference pipeline"""
    with torch.no_grad():
        # Convert numpy → tensor
        tensor = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)  # (1, 4, 512, 512)
        tensor = tensor.to(self.device).float()
        
        # Forward pass
        logits = self.model(tensor)  # (1, 1, 512, 512)
        
        # Sigmoid + threshold
        probs = torch.sigmoid(logits)  # (1, 1, 512, 512)
        mask = (probs > self.threshold).squeeze().cpu().numpy().astype(np.uint8)
        
        return mask
```

---

## 4. Change Detection Algorithm

**File:** `backend/app/services/change_detection.py`

### Pixel-wise Comparison Logic

```python
def generate_change_map(mask_t1, mask_t2):
    """
    Args:
        mask_t1: (H, W) binary array, dtype=uint8 {0, 1}
        mask_t2: (H, W) binary array, dtype=uint8 {0, 1}
    
    Returns:
        change_rgb: (H, W, 3) uint8 RGB image
        class_masks: dict of boolean masks for each class
    """
    
    # Compute boolean class masks (vectorized NumPy operations)
    loss_mask = (mask_t1 == 1) & (mask_t2 == 0)
    gain_mask = (mask_t1 == 0) & (mask_t2 == 1)
    stable_mask = (mask_t1 == 1) & (mask_t2 == 1)
    # non_forest stays as zeros (black pixels)
    
    # Initialize RGB array
    change_rgb = np.zeros((H, W, 3), dtype=np.uint8)
    
    # Color-code each class
    change_rgb[loss_mask] = (220, 50, 47)          # Red
    change_rgb[gain_mask] = (40, 167, 69)          # Green
    change_rgb[stable_mask] = (255, 255, 255)      # White
    # Non-forest: black (0, 0, 0) — default value
    
    return change_rgb, {
        "loss": loss_mask,
        "gain": gain_mask,
        "stable_forest": stable_mask
    }
```

**Computational Complexity:**
- Boolean operations: O(H × W) = O(n)
- Color assignment: O(H × W)
- Total: **Linear in image pixels** ✅ (negligible vs inference time)

**Color Choice Rationale:**
- 🔴 Red (Loss): Standard GIS convention (warm color = negative change)
- 🟢 Green (Gain): Standard GIS convention (positive/cool color = positive change)
- ⚪ White (Stable): High visibility, easy to spot unchanged forests
- ⬛ Black (Non-Forest): Neutral, doesn't distract from changes

---

## 5. STAC Catalog Integration

**File:** `backend/app/services/stac_service.py`

### STAC Query Flow

```python
def fetch_tile_array(self, bbox, date_range, max_cloud_cover=10):
    """
    Workflow:
    1. Search STAC catalog (Element84 AWS)
    2. Select best item (lowest cloud %)
    3. Stream bands from S3
    4. Radiometrically correct
    5. Return (H, W, 4) RGBN array
    """
    
    # Step 1: Search
    search = self.client.search(
        collections=["sentinel-2-l2a"],
        bbox=bbox,                              # [minx, miny, maxx, maxy]
        datetime=date_range,                    # "2024-01-01/2024-12-31"
        query={"eo:cloud_cover": {"lt": max_cloud_cover}},
        max_items=20                            # Get top 20 matches
    )
    items = list(search.items())  # List of STAC Items
    
    # Step 2: Select best
    items.sort(key=lambda x: x.properties["eo:cloud_cover"])
    best_item = items[0]  # Lowest cloud cover
    
    # Step 3: Stream bands from S3
    with rasterio.Env(AWS_NO_SIGN_REQUEST='YES'):  # No auth required
        band_arrays = []
        for band_name in ['red', 'green', 'blue', 'nir']:
            href = best_item.assets[band_name].href  # S3 URL
            
            # Read only the windowed region (not entire TIFF)
            with rasterio.open(href) as src:
                window = from_bounds(*bbox_transformed, src.transform)
                data = src.read(1, window=window)  # Single band
                
                # Step 4: Radiometric correction
                if data.dtype == np.uint16:
                    # Sentinel-2 L2A: DN values are reflectance × 10,000
                    data_uint8 = (np.clip(data / 4000.0, 0, 1) * 255).astype(np.uint8)
                
                band_arrays.append(data_uint8)
        
        # Step 5: Stack into RGBN
        arr = np.stack(band_arrays, axis=0)    # (4, H, W)
        arr = np.transpose(arr, (1, 2, 0))     # (H, W, 4)
        return arr
```

**Why AWS Element84?**
| Feature | Element84 | Google Cloud | Microsoft Azure |
|---------|---|---|---|
| **Authentication** | None (public) | API key required | API key required |
| **Latency** | 5-20 days after acquisition | 2-5 days | 5-20 days |
| **Data Format** | COG (Cloud-Optimized) | GeoTIFF | GeoTIFF |
| **Bandwidth** | Very fast (AWS-to-EC2) | Moderate | Moderate |
| **Cost** | Free → $ (AWS data transfer) | Free API → $ | Free API → $ |

---

## 6. Tiling Strategy for Large Images

**File:** `backend/app/services/tiling.py`

### Non-Overlapping Tiles

**Problem:**
- A 7000×7000 satellite image = 196M pixels
- Loading full image: 196M × 4 bytes × 4 bands = 3GB RAM ❌

**Solution:**
```python
def split_array_into_tiles(array, tile_size=512):
    """
    Divide array into non-overlapping 512×512 patches
    Yields (TileSpec, tile_data) for memory-efficient processing
    """
    H, W = array.shape[:2]
    
    for y in range(0, H, 512):
        for x in range(0, W, 512):
            # Calculate actual tile dimensions (edge tiles may be smaller)
            actual_w = min(512, W - x)
            actual_h = min(512, H - y)
            
            # Extract tile
            tile = array[y:y+512, x:x+512, :]
            
            # Pad if edge tile
            if actual_h < 512 or actual_w < 512:
                tile = np.pad(tile, 
                    ((0, 512-actual_h), (0, 512-actual_w), (0, 0)),
                    mode='reflect')
            
            yield TileSpec(x=x, y=y, width=actual_w, height=actual_h), tile
```

### Stitching Masks Back Together

```python
def stitch_mask_tiles(tiles_data, image_hw):
    """
    Reconstruct full mask from individual tile predictions
    Assumes non-overlapping tiling
    """
    H, W = image_hw
    full_mask = np.zeros((H, W), dtype=np.uint8)
    
    for (tile_spec, tile_mask) in tiles_data:
        x, y = tile_spec.x, tile_spec.y
        actual_w, actual_h = tile_spec.width, tile_spec.height
        
        # Crop padded tile back to actual size
        cropped = tile_mask[:actual_h, :actual_w]
        
        # Place in full mask
        full_mask[y:y+actual_h, x:x+actual_w] = cropped
    
    return full_mask
```

**Complexity:**
- Splitting: O(n) where n = total pixels (1 pass through array)
- Stitching: O(n) (1 pass to place tiles back)
- Inference: O(n/262144) forward passes (n / 512² tiles)

**Total time for 7000×7000:**
```
Images:      2 (T1 + T2) × 7000² = 98M pixels
Tiles:       2 × 169 tiles = 338 tiles
Inference:   338 × 150ms ≈ 50.7s
Stitching:   ~500ms
Change detection: 50ms
Total: ~51s (acceptable for batch processing)
```

---

## 7. Request Validation Pipeline

**File:** `backend/app/routes/detection.py`

### Pre-Flight Validation Gate

```python
def validate_image_dimensions_safe(image_t1, image_t2, task_id):
    """
    Unified validation before inference:
    1. Load image headers (4MB read max)
    2. Extract spatial metadata (CRS, transform)
    3. Validate dimension matching
    4. Enforce pixel limit
    5. Save metadata
    """
    
    for idx, upload_file in enumerate([image_t1, image_t2]):
        filename = upload_file.filename.lower()
        
        # --- GeoTIFF Header Reading ---
        if filename.endswith('.tif'):
            for chunk_size in [4*1024*1024, 16*1024*1024]:
                try:
                    upload_file.file.seek(0)
                    header_bytes = upload_file.file.read(chunk_size)
                    
                    # Parse TIFF header without loading full image
                    with rasterio.open(io.BytesIO(header_bytes)) as src:
                        h, w = src.height, src.width
                        crs = src.crs
                        transform = src.transform
                        break
                except rasterio.errors.RasterioIOError as e:
                    if chunk_size == 16*1024*1024:
                        raise HTTPException(400, f"Invalid TIFF: {e}")
        
        # --- PNG/JPG Header Reading ---
        else:
            upload_file.file.seek(0)
            with Image.open(upload_file.file) as img:
                w, h = img.size
        
        # --- Validation ---
        total_pixels = h * w
        if total_pixels > 50_000_000:
            raise HTTPException(413,
                "Image too large! Max 50M pixels (~7000×7000)")
        
        dimensions.append((w, h))
    
    # --- Dimension Alignment Check ---
    (w1, h1), (w2, h2) = dimensions[0], dimensions[1]
    if w1 != w2 or h1 != h2:
        raise HTTPException(400,
            f"Dimension mismatch: T1={w1}×{h1}, T2={w2}×{h2}")
    
    # --- Save Metadata ---
    metadata = {
        "crs_wkt": spatial_meta["crs_wkt"],
        "transform": spatial_meta["transform"],
        "width": w1,
        "height": h1
    }
    with open(f"static/metadata/{task_id}.json", "w") as f:
        json.dump(metadata, f)
```

**Why These Checks?**
1. **Dimension validation:** Prevents misalignment in change detection
2. **Pixel limit:** Protects against OOM (50M = ~2GB uncompressed)
3. **Metadata extraction:** Enables GeoTIFF export with correct CRS
4. **Early failure:** Rejects bad data before expensive inference

---

## 8. Request-to-Response Flow (Complete)

### POST /api/v1/detect-change Endpoint

```python
@router.post("/detect-change")
async def detect_change(
    image_t1: UploadFile = File(...),
    image_t2: UploadFile = File(...),
    model_name: str = Form("attention_unet"),
    threshold: float = Form(0.5)
):
    """
    Complete change detection workflow
    Time: ~5-60s depending on image size + model choice
    """
    
    # Generate unique task ID for result tracking
    task_id = str(uuid.uuid4())
    
    # Step 1: Validate images
    try:
        validate_image_dimensions_safe(image_t1, image_t2, task_id)
    except HTTPException as e:
        return JSONResponse(status_code=e.status_code, content={"error": e.detail})
    
    # Step 2: Load images into memory
    image_t1_bytes = await image_t1.read()
    image_t2_bytes = await image_t2.read()
    
    t1_array = load_image_to_numpy(image_t1_bytes, image_t1.filename)
    t2_array = load_image_to_numpy(image_t2_bytes, image_t2.filename)
    
    # Step 3: Initialize inference service (or reuse cached)
    try:
        inference_service = InferenceService(
            model_name=model_name,
            threshold=threshold
        )
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})
    
    # Step 4: Run inference on both images
    try:
        mask_t1 = inference_service.predict(t1_array)
        mask_t2 = inference_service.predict(t2_array)
    except Exception as e:
        logger.error(f"Inference failed: {e}")
        return JSONResponse(status_code=500, content={"error": "Inference failed"})
    
    # Step 5: Generate change map
    try:
        change_rgb, class_masks = generate_change_map(mask_t1, mask_t2)
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})
    
    # Step 6: Compute statistics
    statistics = compute_statistics(
        class_masks,
        pixel_size_m=10.0  # Sentinel-2 resolution
    )
    
    # Step 7: Save outputs
    output_dir = f"static/change_maps/{task_id}"
    os.makedirs(output_dir, exist_ok=True)
    
    change_map_pil = Image.fromarray(change_rgb)
    change_map_pil.save(f"{output_dir}/change_map.png")
    
    # Save individual class masks
    Image.fromarray((class_masks["loss"] * 255).astype(np.uint8)).save(
        f"{output_dir}/loss_mask.png"
    )
    Image.fromarray((class_masks["gain"] * 255).astype(np.uint8)).save(
        f"{output_dir}/gain_mask.png"
    )
    
    # Save result JSON
    result_json = {
        "task_id": task_id,
        "status": "success",
        "metrics": statistics,
        "model_used": model_name,
        "threshold": threshold,
        "timestamp": datetime.utcnow().isoformat()
    }
    with open(f"{output_dir}/result.json", "w") as f:
        json.dump(result_json, f, indent=2)
    
    # Step 8: Return response
    return ChangeDetectionResponse(
        task_id=task_id,
        status="success",
        change_map_url=f"/static/change_maps/{task_id}/change_map.png",
        metrics=statistics,
        bbox=None,  # If CRS available from GeoTIFF
        crs="EPSG:4326"
    )
```

---

## 9. Performance Optimization Techniques

### Technique 1: Model Quantization

```python
# Convert float32 model → int8 (4× smaller, 2-3× faster)
import torch.quantization as quantization

quantized_model = quantization.quantize_dynamic(
    model,
    {torch.nn.Linear},  # Only quantize linear layers
    dtype=torch.qint8
)

# Trade-off: ~1-2% accuracy loss, but 4× speedup
```

### Technique 2: Batch Processing

```python
# Process multiple images together (GPU utilization ↑)
batch_size = 10  # 10 image pairs in parallel

for batch in batches:
    # Stack into single tensor
    batch_tensor = torch.stack([img for img in batch])  # (10, 512, 512, 4)
    
    # Single forward pass
    output = model(batch_tensor)  # (10, 512, 512, 1)
    
    # GPU utilization: 40% → 90%
```

### Technique 3: Mixed Precision (float16)

```python
from torch.cuda.amp import autocast

with autocast():
    output = model(tensor)  # Forward pass in float16

# Benefits:
#  + 2× speedup (float16 ops faster than float32)
#  + 2× less memory (float16 = 2 bytes vs float32 = 4 bytes)
#  - Negligible accuracy loss (<0.1%)
```

---

## 10. Debugging & Profiling

### Memory Profiling

```python
import tracemalloc

tracemalloc.start()

# Run inference
mask = inference_service.predict(image)

current, peak = tracemalloc.get_traced_memory()
print(f"Current: {current / 1e6}MB, Peak: {peak / 1e6}MB")
```

### Timing Profiling

```python
import time

start = time.perf_counter()

# Operation to profile
mask_t1 = inference_service.predict(t1_array)

elapsed = time.perf_counter() - start
print(f"Inference time: {elapsed*1000:.1f}ms")
```

### Model Profiling (PyTorch)

```python
import torch.autograd.profiler as profiler

with profiler.profile(use_cuda=True) as prof:
    output = model(input_tensor)

prof.print_table()  # See where time is spent
# Output: CPU time, CUDA time, memory allocation, etc.
```

---

## Conclusion

ForestCover's implementation combines:
- **Modular architecture:** Each service independently testable
- **Optimization:** Streaming I/O, GPU acceleration, singleton caching
- **Robustness:** Validation gates, graceful error handling
- **Scalability:** Async APIs, distributed inference capability

The codebase is production-ready and extensible for:
- ✅ Additional models (add to `_MODEL_REGISTRY`)
- ✅ Alternative data sources (extend `STACService`)
- ✅ Custom preprocessing (add to `InferenceService`)
- ✅ Distributed processing (Celery + Redis queue)

