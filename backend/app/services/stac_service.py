import numpy as np
import rasterio
from rasterio.windows import from_bounds
from rasterio.warp import transform_bounds
from pystac_client import Client
import logging

logger = logging.getLogger(__name__)

class STACService:
    def __init__(self):
        # AWS Element84 Earth Search API (No authentication required)
        self.catalog_url = "https://earth-search.aws.element84.com/v1"
        self.client = Client.open(self.catalog_url)
        self.collection = "sentinel-2-l2a"

    def search_best_item(self, bbox: list[float], date_range: str, max_cloud_cover: int = 10, preferred_mgrs: str = None) -> any:
        """Find the best STAC item, optionally matching a specific MGRS tile."""
        search = self.client.search(
            collections=[self.collection],
            bbox=bbox,
            datetime=date_range,
            query={"eo:cloud_cover": {"lt": max_cloud_cover}},
            max_items=20
        )
        items = list(search.items())
        if not items:
            return None
            
        if preferred_mgrs:
            # Try to find the same tile
            matching = [it for it in items if it.properties.get("s2:mgrs_tile") == preferred_mgrs]
            if matching:
                items = matching
        
        # Sort by cloud cover
        items.sort(key=lambda x: x.properties.get("eo:cloud_cover", 100))
        return items[0]

    def fetch_tile_array(self, bbox: list[float], date_range: str, max_cloud_cover: int = 10, preferred_mgrs: str = None) -> tuple[np.ndarray, str, str, tuple[float, float, float, float]]:
        """
        Query the STAC catalog and stream 4 bands (R, G, B, NIR) for the given bounding box.
        Returns (array, mgrs_tile_id, crs_string, transformed_bbox)
        """
        best_item = self.search_best_item(bbox, date_range, max_cloud_cover, preferred_mgrs)
        
        if not best_item:
            raise ValueError(f"No Sentinel-2 tiles found for {date_range} under {max_cloud_cover}% clouds.")
        
        mgrs_id = best_item.properties.get("s2:mgrs_tile")
        logger.info(f"Using STAC Item: {best_item.id} (MGRS: {mgrs_id}) with {best_item.properties.get('eo:cloud_cover')}% clouds.")
        
        # Specify the 4 bands needed: Red (B04), Green (B03), Blue (B02), NIR (B08)
        # AWS Sentinel-2 assets use logical names: 'red', 'green', 'blue', 'nir'
        required_bands = ['red', 'green', 'blue', 'nir']
        for b in required_bands:
            if b not in best_item.assets:
                raise ValueError(f"Required band '{b}' is missing from STAC item {best_item.id}")

        band_arrays = []
        target_shape = None
        crs_string = None
        transformed_bbox = None
        
        logger.info(f"Streaming {required_bands} bands from AWS S3...")
        
        with rasterio.Env(AWS_NO_SIGN_REQUEST='YES'):
            # First, open one band to get the CRS and transform configuration
            sample_href = best_item.assets['red'].href
            try:
                with rasterio.open(sample_href) as sample_src:
                    src_crs = sample_src.crs

                    if src_crs.to_epsg():
                        crs_string = f"EPSG:{src_crs.to_epsg()}"
                    else:
                        crs_string = src_crs.to_wkt()
                    
                    transformed_bbox = transform_bounds('EPSG:4326', src_crs, *bbox)
                    window = from_bounds(*transformed_bbox, sample_src.transform)
                    
                    logger.info(f"STAC Item {best_item.id} - CRS: {crs_string}, BBox: {transformed_bbox}, Window: {window}")
                    
                    # Fetch all 4 bands using the same window
                    for band_name in required_bands:
                        href = best_item.assets[band_name].href
                        with rasterio.open(href) as src:
                            # Read the first band (index 1) from the TIFF
                            data = src.read(1, window=window)
                            
                            # Ensure all bands have the exact same pixel dimensions
                            if target_shape is None:
                                target_shape = data.shape
                            elif data.shape != target_shape:
                                logger.warning(f"Band {band_name} shape {data.shape} differs from target {target_shape}. Resizing...")
                                from PIL import Image as PILImage
                                temp_img = PILImage.fromarray(data)
                                temp_img = temp_img.resize((target_shape[1], target_shape[0]), PILImage.BILINEAR)
                                data = np.array(temp_img)

                            # Sentinel-2 L2A data is often uint16. Normalize to uint8 for consistency
                            if data.dtype == np.uint16:
                                 # Sentinel-2 DN values (reflectance * 10,000)
                                 # We clip at 4000 (roughly 0.4 reflectance) to keep contrast
                                 data = (np.clip(data / 4000.0, 0, 1) * 255.0).astype(np.uint8)
                            
                            band_arrays.append(data)
            except Exception as e:
                logger.error(f"Failed to stream bands from STAC: {e}")
                raise RuntimeError(f"STAC Streaming Error: {e}")
        
        # Stack into (4, H, W) then transpose to (H, W, 4)
        arr = np.stack(band_arrays, axis=0) # (4, H, W)
        arr = np.transpose(arr, (1, 2, 0)) # (H, W, 4)
        
        return arr, mgrs_id, crs_string, transformed_bbox
