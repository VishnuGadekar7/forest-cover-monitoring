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
        logger.info(f"Loaded weights from {weights_path}")
    else:
        logger.warning(
            "No weight file found -- running with random initialisation. "
            "Place your .pth or .h5 file in backend/weights/ to enable real inference."
        )
    return model


class ModelLoader:
    """
    Manages loading and caching of different segmentation models.
    Supports Attention U-Net, ResNet U-Net, and TransNet.
    """
    _instances: dict[str, "ModelLoader"] = {}

    def __init__(self, model_name: str):
        self.model_name = model_name
        device_pref = os.getenv("DEVICE", "auto")
        
        # Determine weight file
        weights_dir = Path(__file__).resolve().parents[2] / "weights"
        
        # Map of model names to expected weight filenames
        weight_map = {
            "attention_unet": "attention_unet_best.h5",
            "resnet_unet": "resnet_unet.pth",
            "transnet": "transnet.pth"
        }
        
        specific_weight = weight_map.get(model_name)
        weights_path = weights_dir / specific_weight if specific_weight else None
        
        # Fallback logic: If weights don't exist, log warning
        if weights_path and not weights_path.exists():
            logger.warning(f"Weights for {model_name} not found at {weights_path}. Inference will be untrained.")
            weights_path = None

        self.is_keras = False
        
        # Handle Keras/TensorFlow (.h5) models
        if weights_path and weights_path.suffix in [".h5", ".keras"]:
            import tensorflow as tf
            import tensorflow.keras.backend as K
            from app.models.keras_unet import build_keras_unet

            # Custom loss functions matching Colab training setup
            def dice_coef(y_true, y_pred, smooth=1):
                intersection = K.sum(y_true * y_pred)
                return (2. * intersection + smooth) / (K.sum(y_true) + K.sum(y_pred) + smooth)

            def bce_dice_loss(y_true, y_pred):
                bce = tf.keras.losses.binary_crossentropy(y_true, y_pred)
                dice = 1 - dice_coef(y_true, y_pred)
                return bce + dice

            try:
                logger.info(f"Building Keras model for {model_name}...")
                self.model = build_keras_unet()

                # Compile with same loss/metrics used in Colab training
                self.model.compile(
                    optimizer=tf.keras.optimizers.Adam(1e-4),
                    loss=bce_dice_loss,
                    metrics=["accuracy", dice_coef]
                )

                logger.info(f"Loading weights from {weights_path}...")
                self.model.load_weights(str(weights_path))
                self.is_keras = True
                self.device = "tf-auto"
                return
            except Exception as e:
                logger.error(f"Failed to load Keras weights: {e}")
                raise e

        # Handle PyTorch (.pth) models
        self.device = _resolve_device(device_pref)
        logger.info(f"Loading PyTorch model {model_name} on {self.device}...")
        
        self.model = _import_and_build(model_name, self.device)
        self.model.eval()

        if weights_path:
            _load_weights(self.model, weights_path, self.device)
        else:
            logger.warning(f"Starting {model_name} with random weights.")

    @classmethod
    def get_model(cls, model_name: str = None) -> "ModelLoader":
        """Get or create a model instance by name. Defaults to attention_unet."""
        if model_name is None:
            model_name = os.getenv("MODEL_NAME", "attention_unet")
            
        if model_name not in cls._instances:
            cls._instances[model_name] = cls(model_name)
        return cls._instances[model_name]

    @classmethod
    def get_instance(cls) -> "ModelLoader":
        """Legacy support for singleton access."""
        return cls.get_model()
