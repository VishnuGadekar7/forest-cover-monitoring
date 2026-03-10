
import logging
import sys
import os
import numpy as np
from PIL import Image

# Add backend to path
sys.path.append(os.path.abspath('.'))

from app.services.stac_service import STACService

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_alignment():
    service = STACService()
    
    # The BBox from the user's previous request
    bbox = [-63.05, -9.05, -63.0, -9.0]
    
    # Dates used in my last successful test (which the user says look different)
    date_t1 = "2023-01-01/2023-03-31"
    date_t2 = "2023-10-01/2023-12-31"
    
    print(f"--- Querying T1: {date_t1} ---")
    arr1, mgrs1 = service.fetch_tile_array(bbox, date_t1, max_cloud_cover=20)
    print(f"T1 MGRS: {mgrs1}")
    
    print(f"\n--- Querying T2: {date_t2} ---")
    # Simulate the backend logic: pass mgrs1 to T2 query
    arr2, mgrs2 = service.fetch_tile_array(bbox, date_t2, max_cloud_cover=20, preferred_mgrs=mgrs1)
    print(f"T2 MGRS: {mgrs2}")
    
    print(f"\nResults:")
    print(f"T1 Shape: {arr1.shape}")
    print(f"T2 Shape: {arr2.shape}")
    
    # Save thumbnails to verify visually if needed (though I can't see them directly, 
    # the logs of stac_service will tell us the Item IDs)
    Image.fromarray(arr1[:,:,:3]).save("diag_t1.png")
    Image.fromarray(arr2[:,:,:3]).save("diag_t2.png")
    print("Saved diag_t1.png and diag_t2.png")

if __name__ == "__main__":
    test_alignment()
