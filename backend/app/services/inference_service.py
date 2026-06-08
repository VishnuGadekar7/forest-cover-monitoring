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
#from app.utils.image_preprocessing import preprocess, postprocess_mask


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

    def predict(self, image_data: Union[Image.Image, np.ndarray], **kwargs) -> np.ndarray:
        """
        Run segmentation inference. Accepts PIL Image (RGB) or Numpy Array (H, W, C).
        Automatically routes to PyTorch or Keras depending on the model.
        Passes dynamic processing parameters (kwargs) to the underlying pipelines.
        """
        if isinstance(image_data, Image.Image):
            image_data = np.array(image_data)
            
        if self.is_keras:
            return self._predict_keras(image_data, **kwargs)
        return self._predict_pytorch(image_data, **kwargs)

    def _predict_keras(
        self,
        arr: np.ndarray,
        contrast_stretch: bool = True,
        percentile_2_98: bool = True,
        esa_offset_fix: bool = False,
        enable_ndvi_veto: bool = True,
        ndvi_threshold: float = 0.25,
        band_order: str = "RGBN"
    ) -> np.ndarray:
        """
        Keras (.h5) inference pipeline for 4-channel numpy arrays.
        Natively handles axis alignment, percentile contrast stretching, and NDVI validation.
        """
        # Cast the raw input array to float32 precision
        arr = arr.astype(np.float32)
        shape = arr.shape

        if len(shape) != 3:
            raise ValueError(f"Expected a 3D spectral matrix stack, but received shape: {shape}")
        
        # If channel-first (C, H, W) native Rasterio layout, transpose the numpy array to channel-last (H, W, C)
        # Keras models strictly require channel-last inputs.
        if shape[0] < shape[1] and shape[0] < shape[2]:
            arr = np.transpose(arr, (1, 2, 0))

        # Capture original dimensions safely (arr is now guaranteed to be HWC here)
        original_h, original_w = arr.shape[:2]
        is_padded = original_h < 512 or original_w < 512

        # Convert to a TensorFlow Tensor
        tensor = tf.convert_to_tensor(arr, dtype=tf.float32)

        # Resize to 512x512 ONLY if necessary (Saves execution overhead)
        if original_h != 512 or original_w != 512:
            tensor = tf.image.resize(tensor, [512, 512], method='bilinear')

        # Calculate the image maximum value for routing normalization
        raw_max = tf.reduce_max(tensor).numpy()

        # Radiometric Translation, Normalization, Contrast Stretching & Band Alignment Gate
        if raw_max <= 2.0 or raw_max > 255.0:
            if contrast_stretch:
                # Pure core-TF percentile stretch using sort to isolate crowded spectral bands
                channels = []
                for c in range(4):
                    channel_data = tensor[..., c]
                    if percentile_2_98:
                        flat_channel = tf.reshape(channel_data, [-1])
                        total_pixels = tf.cast(tf.shape(flat_channel)[0], tf.float32)
                        sorted_channel = tf.sort(flat_channel, direction='ASCENDING')
                        p2 = sorted_channel[tf.cast(tf.math.round(total_pixels * 0.02), tf.int32)]
                        p98 = sorted_channel[tf.cast(tf.math.round(total_pixels * 0.98), tf.int32)]
                    else:
                        p2, p98 = tf.reduce_min(channel_data), tf.reduce_max(channel_data)
                    
                    if p98 - p2 > 1e-5:
                        stretched = tf.clip_by_value((channel_data - p2) / (p98 - p2), 0.0, 1.0)
                    else:
                        stretched = tf.clip_by_value(channel_data, 0.0, 1.0)
                    channels.append(tf.expand_dims(stretched, axis=-1))
                tensor = tf.concat(channels, axis=-1)
            elif raw_max > 255.0:
                # Raw 16-bit Integer Satellite GeoTIFF (0 - 10000 DN)
                tensor = tf.clip_by_value(tensor, 0.0, 10000.0) / 10000.0
                if esa_offset_fix:
                    tensor = tf.clip_by_value(tensor - 0.1, 0.0, 1.0)
        elif raw_max <= 255.0:
            # Standard 3-Band User Graphic Image (PNG/JPG 0 - 255)
            arr_norm = tensor / 255.0
            
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
            
            tensor = tf.concat([red, green, blue, fake_nir], axis=-1)
        else:
            if band_order == "RGBN":
                red, green, blue, nir = tensor[..., 0:1], tensor[..., 1:2], tensor[..., 2:3], tensor[..., 3:4]
            elif band_order == "NRGB":
                nir, red, green, blue = tensor[..., 0:1], tensor[..., 1:2], tensor[..., 2:3], tensor[..., 3:4]
            else:
                blue, green, red, nir = tensor[..., 0:1], tensor[..., 1:2], tensor[..., 2:3], tensor[..., 3:4]
            tensor = tf.concat([red, green, blue, nir], axis=-1)

        # Add the Batch Dimension -> (1, 512, 512, 4)
        input_tensor = tf.expand_dims(tensor, axis=0) 

        # Forward pass via XLA-compiled function
        probs = self._compiled_predict(input_tensor, training=False)
        
        # Create initial 512x512 binary mask
        mask_512 = tf.cast(probs > self.threshold, tf.float32)

        # THE NDVI VETO
        if enable_ndvi_veto:
            # Keras is Channel-Last! Red is index 0, NIR is index 3.
            red_band = input_tensor[..., 0:1]
            nir_band = input_tensor[..., 3:4]
            
            # Calculate NDVI 
            ndvi = (nir_band - red_band) / (nir_band + red_band + 1e-8)
            
            # Apply Veto: If mask is 1 AND ndvi is < 0.25, force it to 0. 
            veto_condition = tf.logical_and(mask_512 == 1.0, ndvi < ndvi_threshold)
            mask_512_cleaned = tf.where(veto_condition, tf.zeros_like(mask_512), mask_512)
        else:
            mask_512_cleaned = mask_512
        
        # For padded inputs (smaller than 512x512), CROP instead of resize
        # This preserves the full 512x512 prediction quality and lets stitch_mask_tiles handle trimming
        if is_padded:
            # Crop to original dimensions from top-left corner
            mask_cropped = mask_512_cleaned[0, :original_h, :original_w, :]
            mask_np = tf.squeeze(mask_cropped).numpy().astype(np.uint8)
        else:
            if original_h != 512 or original_w != 512:
                mask_512_cleaned = tf.image.resize(mask_512_cleaned, [original_h, original_w], method='nearest')
            # For non-padded 512x512 inputs, return as-is
            mask_np = tf.squeeze(mask_512_cleaned).numpy().astype(np.uint8)

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
    def _predict_pytorch(
        self,
        image_array: np.ndarray,
        contrast_stretch: bool = True,
        percentile_2_98: bool = True,
        esa_offset_fix: bool = False,
        enable_ndvi_veto: bool = True,
        ndvi_threshold: float = 0.25,
        band_order: str = "RGBN"
    ) -> np.ndarray:
        """PyTorch (.pth) inference pipeline for 4-channel numpy arrays."""

        # Cast the uint16 Sentinel-2 array to float32
        image_array = image_array.astype(np.float32)
        shape = image_array.shape

        if len(shape) != 3:
            raise ValueError(f"Expected a 3D spectral matrix stack, but received shape: {shape}")

        # If channel-first (C, H, W), transpose the numpy array to channel-last (H, W, C)
        if shape[0] < shape[1] and shape[0] < shape[2]:
            image_array = np.transpose(image_array, (1, 2, 0))

        # Capture original dimensions safely (image_array is guaranteed to be HWC here)
        original_h, original_w = image_array.shape[:2]
        is_padded = original_h < 512 or original_w < 512

        # Convert to PyTorch format: HWC -> CHW -> Add batch dimension (1, 4, H, W)
        tensor = torch.from_numpy(image_array).permute(2, 0, 1).unsqueeze(0)

        # Step 3: Resize to 512x512 ONLY if necessary (Saves CPU time)
        if original_h != 512 or original_w != 512:
            tensor = F.interpolate(tensor, size=(512, 512), mode='bilinear', align_corners=False)

        raw_max = tensor.max().item()
        # Step 4: Normalization & Enhancement Pipeline
        if raw_max <= 2.0:
            if contrast_stretch:
                # 2%-98% percentile stretch to pull apart crowded spectral bands
                for c in range(tensor.shape[1]):
                    channel_data = tensor[:, c, :, :]
                    if percentile_2_98:
                        p2 = torch.quantile(channel_data.flatten(), 0.02)
                        p98 = torch.quantile(channel_data.flatten(), 0.98)
                    else:
                        p2, p98 = channel_data.min(), channel_data.max()
                    
                    if p98 - p2 > 1e-5:
                        tensor[:, c, :, :] = torch.clamp((channel_data - p2) / (p98 - p2), 0.0, 1.0)
        elif raw_max <= 255.0:
            tensor = tensor / 255.0
        else:
            if contrast_stretch:
                tensor = torch.clamp(tensor, min=0.0, max=10000.0)
                for c in range(tensor.shape[1]):
                    channel_data = tensor[:, c, :, :]
                    if percentile_2_98:
                        p2 = torch.quantile(channel_data.flatten(), 0.02)
                        p98 = torch.quantile(channel_data.flatten(), 0.98)
                    else:
                        p2, p98 = channel_data.min(), channel_data.max()
                    
                    if p98 - p2 > 1e-5:
                        tensor[:, c, :, :] = torch.clamp((channel_data - p2) / (p98 - p2), 0.0, 1.0)
            else:
                tensor = torch.clamp(tensor, min=0.0, max=10000.0) / 10000.0
                if esa_offset_fix:
                    tensor = torch.clamp(tensor - 0.1, min=0.0, max=1.0)

        # Unified Sensor Band Order Mapping Gate
        if raw_max <= 255.0:
            # Standard 3-Band User Graphic Image (PNG/JPG 0 - 255)
            # Extract visible bands (Works with shape [C, H, W] or [B, C, H, W])
            red, green, blue = tensor[:, 0:1, :, :], tensor[:, 1:2, :, :], tensor[:, 2:3, :, :]
            # Synthesize Physical NIR (With aggressive Blue Penalty for Water)
            # We multiply blue by 2.0. If the pixel is a river, the high blue value 
            # will completely destroy the ExG score, dropping NIR to 0.
            exg = (2.0 * green) - red - (blue * 2.0)
            # Create the Fake NIR and prevent negative numbers
            nir = torch.clamp(green + exg, min=0.0, max=1.0)
            # Simulate Satellite Reflectance (Darken visible bands)
            red, green, blue = red * 0.35, green * 0.35, blue * 0.5
        else:
            if band_order == "RGBN":     
                red, green, blue, nir = tensor[:, 0:1, :, :], tensor[:, 1:2, :, :], tensor[:, 2:3, :, :], tensor[:, 3:4, :, :]
            elif band_order == "NRGB":   
                nir, red, green, blue = tensor[:, 0:1, :, :], tensor[:, 1:2, :, :], tensor[:, 2:3, :, :], tensor[:, 3:4, :, :]
            else:                        
                blue, green, red, nir = tensor[:, 0:1, :, :], tensor[:, 1:2, :, :], tensor[:, 2:3, :, :], tensor[:, 3:4, :, :]
        
        tensor = torch.cat([red, green, blue, nir], dim=1)

        # Move to execution device (CPU/GPU)
        tensor = tensor.to(self.device)

        # Step 5: Forward pass (no gradient computation)
        raw_logits = self.model(tensor)  # Expected output: (1, 1, 512, 512)

        # Step 6: Postprocess
        # Apply Sigmoid to convert logits to probabilities [0, 1]
        probs = torch.sigmoid(raw_logits)

        if enable_ndvi_veto:
            ndvi = (nir - red) / (nir + red + 1e-8)
            probs = torch.where(ndvi < ndvi_threshold, torch.zeros_like(probs), probs)

        # For padded inputs (smaller than 512x512), CROP instead of resize
        # This preserves the full 512x512 prediction quality and lets stitch_mask_tiles handle trimming
        if is_padded:
            # Crop to original dimensions from top-left corner
            probs_cropped = probs[:, :, :original_h, :original_w]
            binary_mask = (probs_cropped > self.threshold).byte()
            mask_np = binary_mask.squeeze().cpu().numpy()
        else:
            # Resize back ONLY if necessary (for non-padded images larger than 512x512)
            if original_h != 512 or original_w != 512:
                probs = F.interpolate(probs, size=(original_h, original_w), mode='nearest')

            # Apply threshold and convert to binary mask {0, 1}
            binary_mask = (probs > self.threshold).byte()
            # Move back to CPU, strip the batch/channel dimensions, and convert to numpy
            mask_np = binary_mask.squeeze().cpu().numpy()

        return mask_np