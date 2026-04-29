# preprocessing/06a_generate_classification_data.py
import os
import sys
import pandas as pd
import shutil
from tqdm import tqdm

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import config

def create_classification_dataset():
    """
    Creates the final directory structure and CSV file for classification tasks.
    """
    print("\n--- Stage 6a: Generating Classification Dataset ---")
    
    try:
        features_df = pd.read_csv(config.FEATURES_MASTER_FILE)
        ground_truth_df = pd.read_csv(config.CLASSIFICATION_LABELS_FILE)
    except FileNotFoundError as e:
        print(f"❌ Error: Missing required file: {e}. Please run previous steps or create the ground truth file.")
        return

    # Merge features with ground truth labels
    labeled_df = pd.merge(features_df, ground_truth_df, on="osm_id", how="inner")
    
    # Create the directory structure and copy files
    print("    Creating labeled directory structure and copying files...")
    all_records = []
    for _, row in tqdm(labeled_df.iterrows(), total=len(labeled_df), desc="  Processing labels"):
        osm_id = str(row['osm_id'])
        
        for task in config.LABEL_CATEGORIES:
            label = row.get(task)
            if pd.notna(label):
                # Create directories
                svi_dir = os.path.join(config.LABELS_DATA_DIR, 'svi', task, str(label))
                uav_dir = os.path.join(config.LABELS_DATA_DIR, 'uav', task, str(label))
                os.makedirs(svi_dir, exist_ok=True)
                os.makedirs(uav_dir, exist_ok=True)
                
                # Define source and destination paths
                src_svi_path = os.path.join(config.CROPPED_SVI_CHIPS_DIR, f"{osm_id}.jpg")
                src_uav_path = os.path.join(config.CROPPED_UAV_CHIPS_DIR, f"{osm_id}.tif")
                dest_svi_path = os.path.join(svi_dir, f"{osm_id}.png") # Saving as png as per your example
                dest_uav_path = os.path.join(uav_dir, f"{osm_id}.png")
                
                # Copy files
                if os.path.exists(src_svi_path): shutil.copy(src_svi_path, dest_svi_path)
                if os.path.exists(src_uav_path): shutil.copy(src_uav_path, dest_uav_path)
                
                # Add a record for the final CSV
                all_records.append({'svi_path': dest_svi_path.replace('\\', '/'), 'uav_path': dest_uav_path.replace('\\', '/')})

    # Create and save the final classification CSV
    final_class_df = pd.DataFrame(all_records)
    final_class_df.to_csv(config.FINAL_CLASSIFICATION_FILE, index=False)
    print(f"\nClassification dataset creation complete. Saved to {config.FINAL_CLASSIFICATION_FILE}")

if __name__ == "__main__":
    create_classification_dataset()