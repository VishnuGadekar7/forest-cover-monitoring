"""
Model Loader — Singleton Pattern
==================================
Loads the selected segmentation model ONCE at application startup and
caches it for the lifetime of the process. Thread-safe via Python's GIL
combined with the single-initialisation guard.

Supported model keys:
  - "attention_unet"  (default)
  - "resnet_unet"
  - "transnet"

Configuration via environment variables:
  MODEL_NAME   : which architecture to load (default: attention_unet)
  MODEL_WEIGHTS: path to the .pth state-dict file (optional; random if absent)
  DEVICE       : "cuda" | "cpu" | "auto" (default: auto)
"""

import os
import logging
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)

# Map of registered model names → their build_model factories
_MODEL_REGISTRY: dict[str, str] = {
    "attention_unet": "app.models.attention_unet",
    "resnet_unet":    "app.models.resnet_unet",
    "transnet":       "app.models.transnet",
}

WEIGHTS_DIR = Path(__file__).resolve().parents[2] / "weights"


def _resolve_device(preference: str) -> torch.device:
    if preference == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(preference)


def _import_and_build(model_name: str, device: torch.device) -> nn.Module:
    """Dynamically import the requested model module and call build_model()."""
    if model_name not in _MODEL_REGISTRY:
        raise ValueError(
            f"Unknown model '{model_name}'. "
            f"Available: {list(_MODEL_REGISTRY.keys())}"
        )
    import importlib
    module = importlib.import_module(_MODEL_REGISTRY[model_name])
    model: nn.Module = module.build_model()
    return model.to(device)


def _load_weights(model: nn.Module, weights_path: Optional[Path], device: torch.device) -> nn.Module:
    """Load pretrained weights if a .pth file exists; otherwise run with random init."""
    if weights_path and weights_path.exists():
        state_dict = torch.load(str(weights_path), map_location=device)
        # Handle both raw state_dicts and checkpoint dicts with 'model_state_dict' key
        if "model_state_dict" in state_dict:
            state_dict = state_dict["model_state_dict"]
        model.load_state_dict(state_dict, strict=True)
        logger.info(f"✅ Loaded weights from {weights_path}")
    else:
        logger.warning(
            "⚠️  No weight file found — running with random initialisation. "
            "Place your .pth or .h5 file in backend/weights/ to enable real inference."
        )
    return model


class ModelLoader:
    """
    Singleton that owns the loaded inference model.

    Usage:
        loader = ModelLoader.get_instance()
        model  = loader.model
    """
    _instance: Optional["ModelLoader"] = None

    def __init__(self):
        model_name  = os.getenv("MODEL_NAME", "attention_unet")
        device_pref = os.getenv("DEVICE", "auto")
        weights_env = os.getenv("MODEL_WEIGHTS", "")

        # Resolve weights path: env var takes priority, else look in weights/
        if weights_env:
            weights_path = Path(weights_env)
        else:
            # Try conventional names including .h5
            candidates = [
                WEIGHTS_DIR / f"{model_name}.pth",
                WEIGHTS_DIR / f"{model_name}.h5",
                WEIGHTS_DIR / "attention_unet_forest_trained_final.h5",
                WEIGHTS_DIR / "model.pth",
                WEIGHTS_DIR / "model.h5",
            ]
            weights_path = next((p for p in candidates if p.exists()), None)

        self.is_keras = False
        
        # Bypass PyTorch if it's a Keras/TensorFlow model
        if weights_path and weights_path.suffix in [".h5", ".keras"]:
            import tensorflow as tf
            logger.info(f"✅ Loading Keras model from {weights_path}")
            
            # The model architecture saved in the older Keras .h5 causes ValueError 
            # and AttributeError during deserialization in TF >= 2.13.
            # Instead of loading the corrupted architecture JSON, we build it
            # fresh and just load the raw layer weights.
            from app.models.keras_unet import build_keras_unet
            try:
                logger.info("🛠️  Building fresh Keras U-Net architecture...")
                self.model = build_keras_unet()
                
                logger.info(f"📥 Loading weights from {weights_path}...")
                # load_weights by_name=True precisely maps parameters to their named components
                self.model.load_weights(str(weights_path), by_name=True)
                logger.info("✅ Successfully injected weights into Keras model.")
                
            except Exception as e:
                logger.error(f"❌ Failed to load Keras weights: {e}")
                raise e
                
            self.is_keras = True
            self.device = "tf-auto"
            return

        self.device = _resolve_device(device_pref)
        logger.info(f"🖥️  Inference device: {self.device}")

        self.model = _import_and_build(model_name, self.device)
        self.model.eval()

        _load_weights(self.model, weights_path, self.device)

    @classmethod
    def get_instance(cls) -> "ModelLoader":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance
