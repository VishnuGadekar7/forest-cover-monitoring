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
import tensorflow as tf
import torch.nn.functional as F

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
		# Pre-compile the graph with XLA once during startup
        # reduce_retracing=True prevents memory leaks from repeated compiles
        self._compiled_predict = tf.function(
            self.model, 
            jit_compile=True, 
            reduce_retracing=True
        )

    def predict(self, image: Image.Image) -> np.ndarray:
        """
        Run segmentation inference on a single PIL image.
        Automatically routes to PyTorch or Keras depending on the model.
        """
        if self.is_keras:
            return self._predict_keras(image)
        return self._predict_pytorch(image)

    def _predict_keras(self, arr: np.ndarray) -> np.ndarray:
        # 1. Resize if not already 512x512
        if arr.shape[0] != 512 or arr.shape[1] != 512:
            resized = tf.image.resize(arr, [512, 512])
            arr = resized.numpy()

        # 2. Normalize
        max_val = 255.0 if np.max(arr) <= 255.0 else 10000.0
        arr = arr.astype(np.float32) / max_val

        # 3. FIX: Add the Batch Dimension
        # This changes shape from (512, 512, 4) to (1, 512, 512, 4)
        input_tensor = tf.convert_to_tensor(arr)
        input_tensor = tf.expand_dims(input_tensor, axis=0) 

        # 4. Forward pass via XLA-compiled function
        # training=False is passed here to ensure correct behavior
        probs = self._compiled_predict(input_tensor, training=False)
        
        # 5. Remove batch dimension from output to get back to (512, 512)
        return np.squeeze(probs.numpy())
    
    # def _predict_keras(self, image_array: np.ndarray) -> np.ndarray:
    #     """Keras (.h5) inference pipeline for multispectral numpy arrays."""
    #     original_h, original_w = image_array.shape[:2]

	# 	# Prepare the tensor as we did before
    #     input_tensor = tf.convert_to_tensor(image_array, dtype=tf.float32)
        
    #     # Resize to 512x512 using TensorFlow (safely handles 4 channels)
    #     resized_tensor = tf.image.resize(image_array, [512, 512])
    #     arr = resized_tensor.numpy().astype(np.float32)

    #     # Normalize the array. 
    #     # Note: If your raw Sentinel-2 data values go up to 10,000, change 255.0 to 10000.0
    #     max_val = 255.0 if np.max(arr) <= 255.0 else 10000.0
    #     arr = arr / max_val
        
    #     # Add batch dimension: (1, 512, 512, 4)
    #     arr = np.expand_dims(arr, axis=0)  

    #     # Forward pass
    #     probs = self.model.predict(arr, verbose=0)
    #     probs = np.squeeze(probs)  # Extract (512, 512)

    #     # Thresholding
    #     binary = (probs > self.threshold).astype(np.uint8)

    #     # Resize the 512x512 binary mask back to the original image dimensions
    #     binary_tensor = tf.expand_dims(binary, axis=-1) # Add channel dim for tf.image
    #     restored_tensor = tf.image.resize(
    #         binary_tensor, 
    #         [original_h, original_w], 
    #         method=tf.image.ResizeMethod.NEAREST_NEIGHBOR
    #     )
        
    #     return restored_tensor.numpy().squeeze().astype(np.uint8)

    @torch.no_grad()
    def _predict_pytorch(self, image_array: np.ndarray) -> np.ndarray:
        """PyTorch (.pth) inference pipeline for 4-channel numpy arrays."""
        original_h, original_w = image_array.shape[:2]

        # Step 1: Convert to Tensor and fix dimensions (HWC -> CHW)
        # image_array is (H, W, 4), PyTorch needs (4, H, W)
        # FIX: Cast the uint16 Sentinel-2 array to float32 before PyTorch touches it
        image_array = image_array.astype(np.float32)
        tensor = torch.from_numpy(image_array).float()
        tensor = tensor.permute(2, 0, 1)

        # Step 2: Add batch dimension -> (1, 4, H, W)
        tensor = tensor.unsqueeze(0)

        # Step 3: Resize to 512x512
        tensor = F.interpolate(
            tensor, 
            size=(512, 512), 
            mode='bilinear', 
            align_corners=False
        )

		# Step 4: Normalization & Distribution Shift
        if tensor.max() <= 255.0:
            # Fallback for standard 8-bit standard images
            tensor = tensor / 255.0
        else:
            # 1. Base Sentinel-2 Normalization (Clip glare to 10000)
            tensor = torch.clamp(tensor, min=0.0, max=10000.0) / 10000.0
            
            # 2. ESA 2022 Offset Fix (Removes the +1000 artificial brightness)
            tensor = torch.clamp(tensor - 0.1, min=0.0)

        # Move to execution device (CPU/GPU)
        tensor = tensor.to(self.device)

        # Step 5: Forward pass (no gradient computation)
        raw_logits = self.model(tensor)  # Expected output: (1, 1, 512, 512)

        # Step 6: Postprocess
        # Apply Sigmoid to convert logits to probabilities [0, 1]
        probs = torch.sigmoid(raw_logits)

        # Resize the probability map back to the original image dimensions
        probs_resized = F.interpolate(
            probs, 
            size=(original_h, original_w), 
            mode='nearest'
        )

        # Apply threshold and convert to binary mask {0, 1}
        binary_mask = (probs_resized > self.threshold).byte()

        # Move back to CPU, strip the batch/channel dimensions, and convert to numpy
        mask_np = binary_mask.squeeze().cpu().numpy()

        return mask_np