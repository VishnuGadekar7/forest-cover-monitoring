from pystac_client import Client
import rasterio
import numpy as np

def diagnose_stac_access():
    catalog_url = "https://earth-search.aws.element84.com/v1"
    client = Client.open(catalog_url)
    collection = "sentinel-2-l2a"
    
    bbox = [-63.05, -9.05, -63.00, -9.00]
    date_range = "2023-06-01/2023-07-31"
    
    print(f"Searching {collection}...")
    search = client.search(
        collections=[collection],
        bbox=bbox,
        datetime=date_range,
        max_items=1
    )
    
    items = list(search.items())
    if not items:
        print("No items found.")
        return
    
    item = items[0]
    href = item.assets['red'].href
    print(f"Attempting to open: {href}")
    
    try:
        with rasterio.Env(AWS_NO_SIGN_REQUEST='YES'):
            with rasterio.open(href) as src:
                print(f"Success! Shape: {src.shape}, CRS: {src.crs}")
                # Try a small read
                data = src.read(1, window=((0, 10), (0, 10)))
                print(f"Read sample data: {data.flatten()[:5]}")
    except Exception as e:
        print(f"FAILED to open or read: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    diagnose_stac_access()
