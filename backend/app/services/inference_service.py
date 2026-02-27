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

from app.services.model_loader import ModelLoader
from app.utils.image_preprocessing import preprocess, postprocess_mask


class InferenceService:
    """
    Runs semantic segmentation inference using the singleton model.

    Usage:
        service = InferenceService()
        mask = service.predict(pil_image)   # numpy array (H, W), values {0,1}
    """

    def __init__(self, threshold: float = 0.5):
        """
        Args:
            threshold: Sigmoid output threshold for classifying a pixel as
                       forest (default 0.5 — can be tuned per dataset).
        """
        self.threshold = threshold
        loader = ModelLoader.get_instance()
        self.model = loader.model
        self.device = loader.device
        self.is_keras = getattr(loader, "is_keras", False)

    def predict(self, image: Image.Image) -> np.ndarray:
        """
        Run segmentation inference on a single PIL image.
        Automatically routes to PyTorch or Keras depending on the model.
        """
        if self.is_keras:
            return self._predict_keras(image)
        return self._predict_pytorch(image)

    def _predict_keras(self, image: Image.Image) -> np.ndarray:
        """Keras (.h5) inference pipeline."""
        original_size = image.size
        # Resize to 512x512
        img_resized = image.resize((512, 512), Image.BILINEAR)
        # Normalize to [0, 1] — common for Keras segmentation models
        arr = np.array(img_resized, dtype=np.float32) / 255.0
        arr = np.expand_dims(arr, axis=0)  # (1, 512, 512, 3)

        # Forward pass
        probs = self.model.predict(arr, verbose=0)
        probs = np.squeeze(probs)  # Extract (512, 512)

        # Thresholding
        binary = (probs > self.threshold).astype(np.uint8)

        # Resize back to original
        mask_img = Image.fromarray(binary)
        mask_img = mask_img.resize(original_size, Image.NEAREST)
        return np.array(mask_img, dtype=np.uint8)

    @torch.no_grad()
    def _predict_pytorch(self, image: Image.Image) -> np.ndarray:
        """PyTorch (.pth) inference pipeline."""
        # Step 1 — Preprocess: resize, normalise, add batch dim
        tensor, original_size = preprocess(image)
        tensor = tensor.to(self.device)

        # Step 2 — Forward pass (no gradient computation)
        raw_logits = self.model(tensor)             # (1, 1, 512, 512)

        # Step 3 — Postprocess: sigmoid → threshold → resize to original
        mask = postprocess_mask(raw_logits, original_size, self.threshold)

        return mask
