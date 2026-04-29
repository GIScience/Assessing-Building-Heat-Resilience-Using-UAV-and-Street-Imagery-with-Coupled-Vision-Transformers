# preprocessing/06b_generate_regression_data.py
import pandas as pd
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import config

def create_regression_dataset():
    """
    Creates the final CSV file for regression tasks.
    """
    print("\n--- Stage 6b: Generating Regression Dataset ---")
    
    try:
        features_df = pd.read_csv(config.FEATURES_MASTER_FILE)
    except FileNotFoundError as e:
        print(f"❌ Error: Missing master features file: {e}. Please run '05_fetch_and_engineer_features.py' first.")
        return

    # Select and format the required columns
    regression_df = features_df[['osm_id', 'wall_brightness', 'roof_brightness']].copy()
    regression_df['svi_path'] = regression_df['osm_id'].apply(
        lambda x: os.path.join(config.CROPPED_SVI_CHIPS_DIR, f"{x}.jpg").replace('\\', '/')
    )
    regression_df['uav_path'] = regression_df['osm_id'].apply(
        lambda x: os.path.join(config.CROPPED_UAV_CHIPS_DIR, f"{x}.tif").replace('\\', '/')
    )
    
    # Ensure correct column order
    final_cols = ['svi_path', 'uav_path', 'wall_brightness', 'roof_brightness']
    final_regression_df = regression_df[final_cols].dropna()

    final_regression_df.to_csv(config.FINAL_REGRESSION_FILE, index=False)
    print(f"Regression dataset creation complete. Saved to {config.FINAL_REGRESSION_FILE}")


if __name__ == "__main__":
    create_regression_dataset()