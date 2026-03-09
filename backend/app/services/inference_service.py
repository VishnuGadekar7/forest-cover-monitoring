"""
Inference Service
==================
Wraps the loaded model and exposes a single predict() method that
accepts a PIL Image and returns a binary numpy mask.

Design:
  - Stateless: no instance variables mutate after construction.
  - GPU-safe: tensors moved to the correct device before forward pass.
  - Batch-safe: processes one image at a time (extend to batch if needed).
  - torch.no_grad() context prevents gradient computation overhead.
"""

import numpy as np
import torch
from PIL import Image
from typing import Union

from app.services.model_loader import ModelLoader
from app.utils.image_preprocessing import preprocess, postprocess_mask


class InferenceService:
    """
    Runs semantic segmentation inference using one of the available models.
    """

    def __init__(self, model_name: str = "attention_unet", threshold: float = 0.5):
        """
        Args:
            model_name: Name of the model to use (attention_unet, resnet_unet, transnet).
            threshold: Sigmoid output threshold for forest classification.
        """
        self.threshold = threshold
        loader = ModelLoader.get_model(model_name)
        self.model = loader.model
        self.device = loader.device
        self.is_keras = loader.is_keras

    def predict(self, image_data: Union[Image.Image, np.ndarray]) -> np.ndarray:
        """
        Run segmentation inference. Accepts PIL Image (RGB) or Numpy Array (H, W, C).
        Automatically routes to PyTorch or Keras depending on the model.
        """
        if isinstance(image_data, Image.Image):
            image_data = np.array(image_data)
            
        if self.is_keras:
            return self._predict_keras(image_data)
        return self._predict_pytorch(image_data)

    def _predict_keras(self, arr: np.ndarray) -> np.ndarray:
        """Keras (.h5) inference pipeline for 4-channel models."""
        h, w = arr.shape[:2]
        original_size = (w, h)
        
        # Preprocess: resize + normalize (0-1) + pad to 4 channels
        img = Image.fromarray(arr)
        img_resized = img.resize((512, 512), Image.BILINEAR)
        arr_proc = np.array(img_resized, dtype=np.float32) / 255.0
        
        # Consistent padding: ensure 4 channels (RGB + NIR)
        if arr_proc.shape[-1] == 3:
            padding = np.zeros((512, 512, 1), dtype=np.float32)
            arr_proc = np.concatenate([arr_proc, padding], axis=-1)
            
        arr_proc = np.expand_dims(arr_proc, axis=0)  # (1, 512, 512, 4)

        # Forward pass
        probs = self.model.predict(arr_proc, verbose=0)
        probs = np.squeeze(probs)  # Extract (512, 512)

        # Thresholding
        binary = (probs > self.threshold).astype(np.uint8)

        # Resize back to original
        mask_img = Image.fromarray(binary)
        mask_img = mask_img.resize(original_size, Image.NEAREST)
        return np.array(mask_img, dtype=np.uint8)

    @torch.no_grad()
    def _predict_pytorch(self, image_data: np.ndarray) -> np.ndarray:
        """PyTorch (.pth) inference pipeline."""
        # Step 1 — Preprocess: resize, normalise, add batch dim, pad to 4
        tensor, original_size = preprocess(image_data)
        tensor = tensor.to(self.device)

        # Step 2 — Forward pass (no gradient computation)
        raw_logits = self.model(tensor)             # (1, 1, 512, 512)

        # Step 3 — Postprocess: sigmoid → threshold → resize to original
        mask = postprocess_mask(raw_logits, original_size, self.threshold)

        return mask
