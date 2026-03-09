
import os
import sys
import numpy as np
import torch
from PIL import Image

# Add backend to path
sys.path.append(os.path.abspath('.'))

from app.services.model_loader import ModelLoader
from app.services.inference_service import InferenceService

def debug_prediction():
    print("Loading model...")
    service = InferenceService(model_name="attention_unet")
    
    # Create a dummy 4-channel image (mostly green/NIR)
    # R, G, B, NIR
    arr = np.zeros((512, 512, 4), dtype=np.uint8)
    arr[:, :, 1] = 150 # Green
    arr[:, :, 3] = 200 # NIR
    
    img = Image.fromarray(arr)
    
    print("Running prediction (Simple 0-1 Normalization)...")
    if service.is_keras:
        # Simple 0-1
        arr_proc = np.array(img, dtype=np.float32) / 255.0
        arr_proc = np.expand_dims(arr_proc, axis=0)
        probs = service.model.predict(arr_proc, verbose=0)
        probs = np.squeeze(probs)
        print(f"Keras (0-1) - Max: {np.max(probs):.4f}, Mean: {np.mean(probs):.4f}")
        
        # ImageNet Normalization
        from app.utils.image_preprocessing import IMAGENET_MEAN, IMAGENET_STD
        arr_imgnet = np.array(img, dtype=np.float32) / 255.0
        # Normalise first 3 channels
        arr_imgnet[:, :, :3] = (arr_imgnet[:, :, :3] - IMAGENET_MEAN) / IMAGENET_STD
        arr_imgnet = np.expand_dims(arr_imgnet, axis=0)
        probs_in = service.model.predict(arr_imgnet, verbose=0)
        probs_in = np.squeeze(probs_in)
        print(f"Keras (ImageNet) - Max: {np.max(probs_in):.4f}, Mean: {np.mean(probs_in):.4f}")
    else:
        # PyTorch
        from app.utils.image_preprocessing import preprocess
        tensor, _ = preprocess(arr)
        tensor = tensor.to(service.device)
        with torch.no_grad():
            raw_logits = service.model(tensor)
            probs = torch.sigmoid(raw_logits).cpu().numpy()
            probs = np.squeeze(probs)
            print(f"PyTorch Probs - Max: {np.max(probs):.4f}, Min: {np.min(probs):.4f}, Mean: {np.mean(probs):.4f}")

    binary = (probs > 0.5).astype(np.uint8)
    print(f"Forest pixels detected (threshold 0.5): {np.sum(binary)}")
    
    # Try different thresholds
    for t in [0.1, 0.01, 0.001]:
        b = (probs > t).astype(np.uint8)
        print(f"Forest pixels detected (threshold {t}): {np.sum(b)}")

if __name__ == '__main__':
    debug_prediction()
