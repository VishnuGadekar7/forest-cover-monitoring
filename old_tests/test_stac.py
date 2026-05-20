import requests
import json

def test_stac_automated():
    url = "http://127.0.0.1:8002/api/v1/detect-change-automated"
    
    # Small BBox in Rondonia, Brazil (Amazon Rainforest)
    # [min_lon, min_lat, max_lon, max_lat]
    payload = {
        "bbox": [-63.05, -9.05, -63.00, -9.00],
        "date_t1": "2021-06-01/2021-07-31",
        "date_t2": "2023-06-01/2023-07-31",
        "max_cloud_cover": 20
    }
    
    print(f"Sending STAC request for BBox: {payload['bbox']}...")
    try:
        response = requests.post(url, json=payload, timeout=180)
        print(f"Status Code: {response.status_code}")
        
        if response.status_code == 200:
            print("Success! Response JSON:")
            print(json.dumps(response.json(), indent=2))
        else:
            print("Error Response:")
            print(response.text)
    except Exception as e:
        print(f"Request failed: {e}")

if __name__ == "__main__":
    test_stac_automated()
