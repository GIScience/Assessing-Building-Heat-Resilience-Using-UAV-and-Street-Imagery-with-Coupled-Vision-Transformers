# preprocessing/utils/api_helpers.py
import os
import requests
import pandas as pd
from tqdm import tqdm
from config import MAPILLARY_DETECTIONS_DIR, ACCESS_TOKEN

def fetch_mapillary_detections(image_ids):
    """Fetches object detections for a list of Mapillary image IDs."""
    print("--> Fetching Mapillary object detections...")
    os.makedirs(MAPILLARY_DETECTIONS_DIR, exist_ok=True)
    
    unique_ids = [str(id) for id in image_ids if pd.notna(id)]
    
    for img_id in tqdm(unique_ids, desc="  Fetching detections"):
        csv_path = os.path.join(MAPILLARY_DETECTIONS_DIR, f"{img_id}.csv")
        if os.path.exists(csv_path):
            continue

        url = f"https://graph.mapillary.com/{img_id}/detections?access_token={ACCESS_TOKEN}&fields=image,value,geometry"
        try:
            response = requests.get(url, timeout=30)
            if response.status_code == 200:
                data = response.json().get('data', [])
                if data:
                    pd.DataFrame(data).to_csv(csv_path, index=False)
            else:
                print(f"    Warning: Failed for ID {img_id}: {response.status_code}")
        except requests.RequestException as e:
            print(f"    Error for ID {img_id}: {e}")