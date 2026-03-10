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
        self.is_keras = getattr(loader, "is_keras", False)
		# Pre-compile the graph with XLA once during startup
        # reduce_retracing=True prevents memory leaks from repeated compiles
        self._compiled_predict = tf.function(
            self.model, 
            jit_compile=True, 
            reduce_retracing=True
        )

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
        original_h, original_w = arr.shape[:2]

        # Resize if not already 512x512
        if arr.shape[0] != 512 or arr.shape[1] != 512:
            arr_resized = tf.image.resize(arr, [512, 512])
        else:
            arr_resized = tf.cast(arr, tf.float32)

        # Radiometric Translation & Normalization
        # Use a strict check: if raw array max is <= 255, it's a PNG/JPG
        if tf.reduce_max(arr_resized) <= 255.0:
            arr_norm = arr_resized / 255.0
            
            red = arr_norm[..., 0:1]
            green = arr_norm[..., 1:2]
            blue = arr_norm[..., 2:3]
            
            # Synthesize Fake NIR with Blue Penalty
            exg = (2.0 * green) - red - (blue * 2.0)
            fake_nir = tf.clip_by_value(green + exg, 0.0, 1.0)
            
            # Darken to simulate satellite reflectance
            red = red * 0.35
            green = green * 0.35
            blue = blue * 0.5
            
            input_tensor = tf.concat([red, green, blue, fake_nir], axis=-1)
        else:
            # 16-bit GeoTIFF Base Normalization
            arr_norm = tf.clip_by_value(arr_resized, 0.0, 10000.0) / 10000.0
            
            # ESA OFFSET APPLIED
            input_tensor = tf.clip_by_value(arr_norm - 0.1, 0.0, 1.0)

        # Add the Batch Dimension -> (1, 512, 512, 4)
        input_tensor = tf.expand_dims(input_tensor, axis=0) 

        # Forward pass via XLA-compiled function
        probs = self._compiled_predict(input_tensor, training=False)
        
        # Create initial 512x512 binary mask
        mask_512 = tf.cast(probs > self.threshold, tf.float32)

        # THE NDVI VETO (Pure TensorFlow)
        # Keras is Channel-Last! Red is index 0, NIR is index 3.
        red_band = input_tensor[..., 0:1]
        nir_band = input_tensor[..., 3:4]
        
        # Calculate NDVI 
        ndvi = (nir_band - red_band) / (nir_band + red_band + 1e-8)
        
        # Apply Veto: If mask is 1 AND ndvi is < 0.25, force it to 0. 
        veto_condition = tf.logical_and(mask_512 == 1.0, ndvi < 0.25)
        mask_512_cleaned = tf.where(veto_condition, tf.zeros_like(mask_512), mask_512)

        # Resize the CLEANED map back to original dimensions
        mask_resized = tf.image.resize(
            mask_512_cleaned, 
            [original_h, original_w], 
            method=tf.image.ResizeMethod.NEAREST_NEIGHBOR
        )

        # Clean up dims and convert back to numpy
        mask_np = tf.squeeze(mask_resized).numpy().astype(np.uint8)

        return mask_np

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
            # 1. Base normalization to 0.0 - 1.0
            tensor = tensor / 255.0
            
            # Extract visible bands (Works with shape [C, H, W] or [B, C, H, W])
            red = tensor[..., 0:1, :, :]
            green = tensor[..., 1:2, :, :]
            blue = tensor[..., 2:3, :, :]
            
            # 2. Synthesize Physical NIR (With aggressive Blue Penalty for Water)
            # We multiply blue by 2.0. If the pixel is a river, the high blue value 
            # will completely destroy the ExG score, dropping NIR to 0.
            exg = (2.0 * green) - red - (blue * 2.0)
            
            # Create the Fake NIR and prevent negative numbers
            fake_nir = torch.clamp(green + exg, min=0.0, max=1.0)
            
            # 3. Simulate Satellite Reflectance (Darken visible bands)
            red = red * 0.35
            green = green * 0.35
            blue = blue * 0.5
            
            # 4. Re-stack the tensor with the synthesized NIR band
            tensor = torch.cat([red, green, blue, fake_nir], dim=-3)
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