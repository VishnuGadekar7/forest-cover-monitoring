import numpy as np
from app.utils.metrics import compute_area, compute_statistics

def test_compute_area():
    mask = np.zeros((100, 100), dtype=np.uint8)
    mask[0:10, 0:10] = 1  # 100 pixels
    
    # Sentinel-2 resolution (10m): 100 pixels * 100 m^2 = 10,000 m^2 = 1.0 ha
    area = compute_area(mask, pixel_resolution_m=10.0)
    assert area == 1.0

def test_compute_statistics():
    t1 = np.zeros((100, 100), dtype=np.uint8)
    t2 = np.zeros((100, 100), dtype=np.uint8)
    
    # T1: 100 pixels forest
    t1[0:10, 0:10] = 1 
    
    # T2: Keep 50 pixels (stable), lose 50, gain 20
    t2[0:5, 0:10] = 1      # 50 pixels stable
    t2[20:25, 0:4] = 1     # 20 pixels gain
    
    stats = compute_statistics(t1, t2, pixel_resolution_m=10.0)
    
    # 1 pixel = 0.01 ha
    assert stats["forest_area_t1"] == 1.0    # 100 * 0.01
    assert stats["forest_area_t2"] == 0.7    # (50 + 20) * 0.01
    assert stats["forest_loss"] == 0.5       # 50 * 0.01
    assert stats["forest_gain"] == 0.2       # 20 * 0.01
    
    # percentage: (0.7 - 1.0) / 1.0 = -30.0%
    assert round(stats["percentage_change"], 2) == -30.0
