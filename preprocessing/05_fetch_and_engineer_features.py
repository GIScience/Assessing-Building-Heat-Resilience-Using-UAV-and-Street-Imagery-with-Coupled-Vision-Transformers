# preprocessing/05_fetch_and_engineer_features.py
import os
import sys
import pandas as pd
import geopandas as gpd

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils.api_helpers import fetch_mapillary_detections
from utils.feature_engineering import process_svi_for_wall_brightness, calculate_roof_brightness, calculate_vegetation_presence
import config

def run_feature_pipeline():
    """
    Fetches raw data and computes advanced features, saving an intermediate master file.
    """
    print("--- Stage 5: Fetching Detections and Engineering Features ---")
    
    # Load the visible buildings file (output from script 02)
    try:
        visible_buildings = gpd.read_file(config.VISIBLE_BUILDINGS_FILE)
    except Exception as e:
        print(f"❌ Error: Could not read visible buildings file. Run script 02 first. Details: {e}")
        return

    # Step 1: Fetch all required Mapillary detection data
    mapillary_ids = visible_buildings['mapillary_id'].unique()
    fetch_mapillary_detections(mapillary_ids)
    
    # Step 2: Calculate Wall Brightness from SVI detections
    # This requires precomputed centerline data (CENTERLINE_DATA_FILE).
    centerline_df = pd.read_csv(config.CENTERLINE_DATA_FILE)
    brightness_df = process_svi_for_wall_brightness(centerline_df)

    # Step 3: Calculate Roof Brightness
    features_df = calculate_roof_brightness(brightness_df)
    
    # Step 4: Calculate Vegetation Presence
    features_df = calculate_vegetation_presence(features_df)
    
    # Step 5: Save the consolidated features to a master file
    features_df.to_csv(config.FEATURES_MASTER_FILE, index=False)
    print(f"\n✅ Feature engineering complete. Master feature file saved to {config.FEATURES_MASTER_FILE}")

if __name__ == "__main__":
    run_feature_pipeline()