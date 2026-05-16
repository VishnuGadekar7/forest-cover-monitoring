import numpy as np
from app.services.tiling import stitch_mask_tiles, split_array_into_tiles
from app.services.inference_service import InferenceService

def test_split_and_stitch_rt():
    # non divisible image dimensions
    arr = np.zeros((2501, 1849, 4), dtype=np.uint16)

    tiles = list(split_array_into_tiles(arr, tile_size=512))
    assert len(tiles) > 0

    predicted = []
    for spec, tile in tiles:
        # NEW: Verify the generator successfully padded the array to 512x512
        assert tile.shape == (512, 512, 4)
        
        # fake mask: model outputs 512x512 mask regardless of original edge size
        mask = np.ones((512, 512), dtype=np.uint8)
        predicted.append((spec, mask))
    
    (h, w) = arr.shape[:2]
    stitched = stitch_mask_tiles(predicted, (h, w))
    
    assert stitched.shape == (2501, 1849)
    assert np.all(stitched == 1)

def test_smaller_than_tile():
    arr = np.zeros((300, 400, 4), dtype=np.uint16)

    tiles = list(split_array_into_tiles(arr, tile_size=512))
    assert len(tiles) == 1

    predicted = []
    for spec, tile in tiles:
        # Verify it was padded UP to 512x512
        assert tile.shape == (512, 512, 4)
        # Verify the Spec remembers the true dimensions
        assert spec.width == 400
        assert spec.height == 300
        
        mask = np.ones((512, 512), dtype=np.uint8)
        predicted.append((spec, mask))

    (h, w) = arr.shape[:2]
    stitched = stitch_mask_tiles(predicted, (h, w))
    assert stitched.shape == (300, 400)
    assert np.all(stitched == 1)

def test_edge_tile_sizes():
    arr = np.zeros((550, 770, 4), dtype=np.uint16)

    tiles = list(split_array_into_tiles(arr, tile_size=512))
    assert len(tiles) == 4

    # Verify the TileSpec remembers the actual unpadded slices to trim later
    sizes = [(spec.width, spec.height) for spec, _ in tiles]
    assert (512, 512) in sizes
    assert (258, 512) in sizes
    assert (512, 38) in sizes
    assert (258, 38) in sizes

    # Verify the actual arrays yielded to the ML model are ALWAYS 512x512
    for _, tile in tiles:
        assert tile.shape == (512, 512, 4)

def test_tile_positions():
    arr = np.zeros((600, 600, 4), dtype=np.uint16)

    tiles = list(split_array_into_tiles(arr, tile_size=256))
    predicted = []

    for idx, (spec, tile) in enumerate(tiles, start=1):
        # Verify padding to 256
        assert tile.shape == (256, 256, 4)
        
        mask = np.full((256, 256), idx, dtype=np.uint8)
        predicted.append((spec, mask))

    h, w = arr.shape[:2]
    stitched = stitch_mask_tiles(predicted, (h, w))

    assert stitched.shape == (600, 600)

    # Verify the stitched values map correctly to the coordinates
    for idx, (spec, _) in enumerate(predicted, start=1):
        y = spec.y
        x = spec.x
        assert stitched[y, x] == idx

def test_empty_tiles():
    stitched = stitch_mask_tiles([], (100, 100))
    assert stitched.shape == (100, 100)
    assert np.all(stitched == 0)

def test_full_coverage():
    arr = np.zeros((777, 911, 4), dtype=np.uint16)

    tiles = list(split_array_into_tiles(arr, tile_size=256))
    predicted = []

    for spec, tile in tiles:
        assert tile.shape == (256, 256, 4)
        mask = np.ones((256, 256), dtype=np.uint8)
        predicted.append((spec, mask))

    h, w = arr.shape[:2]
    stitched = stitch_mask_tiles(predicted, (h, w))

    assert stitched.shape == (777, 911)
    assert np.all(stitched == 1)
