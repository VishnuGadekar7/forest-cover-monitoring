from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator, Iterable

import numpy as np
import rasterio
from rasterio.windows import Window


@dataclass
class TileSpec:
    x: int
    y: int
    width: int
    height: int


def stream_tiles_from_disk(
    file_path: str, 
    tile_size: int = 512
) -> Iterator[tuple[TileSpec, np.ndarray]]:
    """
    Streams a massive GeoTIFF straight from the hard drive patch-by-patch.
    Keeps RAM footprint incredibly low.
    """
    with rasterio.open(file_path) as src:
        h, w = src.height, src.width
        
        for y in range(0, h, tile_size):
            for x in range(0, w, tile_size):
                # Calculate actual dimensions (for edge cases)
                tw = min(tile_size, w - x)
                th = min(tile_size, h - y)
                
                # Create a Rasterio Window to read ONLY this specific chunk from the hard drive
                window = Window(x, y, tw, th)
                
                # Read just this window (Channels, Height, Width) and transpose to HWC
                tile_bands = src.read(window=window)
                actual_tile = np.transpose(tile_bands, (1, 2, 0))
                
                # Check for empty tiles
                if actual_tile.size == 0:
                    continue

                # Pad edge tiles if necessary (so the ML model gets exactly 512x512)
                if th < tile_size or tw < tile_size:
                    pad_h = tile_size - th
                    pad_w = tile_size - tw
                    tile = np.pad(actual_tile, ((0, pad_h), (0, pad_w), (0, 0)), mode='reflect')
                else:
                    tile = actual_tile

                yield TileSpec(x=x, y=y, width=tw, height=th), tile


def read_tiff_in_windows(
    file_bytes: bytes,
    tile_size: int = 512,
) -> tuple[np.ndarray, list[TileSpec], tuple[int, int]]:
    """
    Legacy method: Read a GeoTIFF into an in-memory full array.
    Warning: Highly RAM intensive for massive images. Use stream_tiles_from_disk instead.

    Returns:
        full_array: (H, W, C)
        tiles: list of TileSpec
        image_size: (H, W)
    """
    from rasterio.io import MemoryFile

    with MemoryFile(file_bytes) as memfile:
        with memfile.open() as src:
            bands = src.read()  # (C, H, W)
            full_array = np.transpose(bands, (1, 2, 0))  # (H, W, C)
            h, w = full_array.shape[:2]

            tiles: list[TileSpec] = []
            for y in range(0, h, tile_size):
                for x in range(0, w, tile_size):
                    tw = min(tile_size, w - x)
                    th = min(tile_size, h - y)
                    tiles.append(TileSpec(x=x, y=y, width=tw, height=th))

            return full_array, tiles, (h, w)


def split_array_into_tiles(
    array: np.ndarray,
    tile_size: int = 512,
) -> Iterator[tuple[TileSpec, np.ndarray]]:
    """
    Split an in-memory HWC array into non-overlapping tiles.
    Pads edge tiles so the ML models receive perfectly uniform inputs.
    """
    h, w = array.shape[:2]
    for y in range(0, h, tile_size):
        for x in range(0, w, tile_size):
            tw = min(tile_size, w - x)
            th = min(tile_size, h - y)

            actual_tile = array[y:y + th, x:x + tw]
            if actual_tile.size == 0:
                continue

            # Pad edge tiles to guarantee exactly (tile_size, tile_size, C)
            if th < tile_size or tw < tile_size:
                pad_h = tile_size - th
                pad_w = tile_size - tw
                # Pad using 'reflect' to simulate natural edges
                tile = np.pad(actual_tile, ((0, pad_h), (0, pad_w), (0, 0)), mode='reflect')
            else:
                tile = actual_tile

            # Yield the TileSpec with ACTUAL dimensions, but the array is fully padded
            yield TileSpec(x=x, y=y, width=tw, height=th), tile


def stitch_mask_tiles(
    tiles: Iterable[tuple[TileSpec, np.ndarray]],
    image_size: tuple[int, int],
) -> np.ndarray:
    """
    Stitch predicted masks back into a full-size mask.
    Consumes generators lazily to preserve memory.
    """
    h, w = image_size
    out = np.zeros((h, w), dtype=np.uint8)

    for spec, mask in tiles:
        if mask.ndim > 2:
            mask = np.squeeze(mask)

        # Trims off any padding from edge tiles using the true spec dimensions
        th = min(spec.height, mask.shape[0])
        tw = min(spec.width, mask.shape[1])
        
        out[spec.y:spec.y + th, spec.x:spec.x + tw] = mask[:th, :tw].astype(np.uint8)

    return out