# preprocessing/04_create_svi_data.py
import os
import math
import sys
import requests
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point
from tqdm import tqdm
from PIL import Image, ImageDraw
from io import BytesIO

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import settings from the configuration file
import config

# --- Constants for SVI Processing ---
IMAGE_WIDTH = 4096
IMAGE_HEIGHT = 2048
CENTER_X = IMAGE_WIDTH / 2
PIXELS_PER_DEGREE = IMAGE_WIDTH / 360

# --- Helper Functions ---
def calculate_bearing(lat1, lon1, lat2, lon2):
    """Calculates bearing from point 1 to point 2 in degrees."""
    lat1, lat2, delta_lon = map(math.radians, [lat1, lat2, lon2 - lon1])
    y = math.sin(delta_lon) * math.cos(lat2)
    x = math.cos(lat1) * math.sin(lat2) - math.sin(lat1) * math.cos(lat2) * math.cos(delta_lon)
    return (math.degrees(math.atan2(y, x)) + 360) % 360

def haversine_distance(lat1, lon1, lat2, lon2):
    """Calculates distance between two lat/lon points in meters."""
    R = 6371000  # Earth radius in meters
    phi1, phi2, dphi, dlambda = map(math.radians, [lat1, lat2, lat2 - lat1, lon2 - lon1])
    a = math.sin(dphi / 2)**2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2)**2
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

def create_svi_chips():
    """
    Downloads, processes, and crops SVI panoramas for each visible building.
    """
    print("--- Stage 4: Creating Street View Image (SVI) Chips ---")

    # 1. Load required data
    try:
        buildings = gpd.read_file(config.VISIBLE_BUILDINGS_FILE)
        map_points = gpd.read_file(config.COMBINED_MAPILLARY_FILE)
    except Exception as e:
        print(f"❌ Error: Could not read input files. Ensure previous steps are complete.")
        print(f"   Details: {e}")
        return

    # 2. Ensure CRS is consistent for calculations (WGS 84)
    buildings = buildings.to_crs(epsg=4326)
    map_points = map_points.to_crs(epsg=4326)

    # 3. Create output directories
    os.makedirs(config.SVI_IMAGES_DIR, exist_ok=True)
    os.makedirs(config.CROPPED_SVI_CHIPS_DIR, exist_ok=True)
    print(f"    Full panoramas will be saved to: {config.SVI_IMAGES_DIR}")
    print(f"    Cropped SVI chips will be saved to: {config.CROPPED_SVI_CHIPS_DIR}")

    # 4. Create a lookup dictionary for Mapillary points for faster access
    map_points_dict = {str(row['id']): row for _, row in map_points.iterrows()}

    for _, row in tqdm(buildings.iterrows(), total=len(buildings), desc="  Processing buildings"):
        map_id = str(row["mapillary_id"])
        osm_id = str(row["osm_id"])
        building_centroid = row.building_geometry.centroid

        mrow = map_points_dict.get(map_id)
        if mrow is None:
            continue

        # 5. Download the full panoramic image if it doesn't exist
        img_path = os.path.join(config.SVI_IMAGES_DIR, f"{map_id}.jpg")
        if not os.path.exists(img_path):
            try:
                response = requests.get(mrow["thumb_original_url"], timeout=20)
                response.raise_for_status()
                img = Image.open(BytesIO(response.content))
                img.save(img_path)
            except requests.RequestException as e:
                print(f"    Warning: Failed to download image for {map_id}: {e}")
                continue
        else:
            img = Image.open(img_path)

        # 6. Calculate the position of the building in the panorama
        map_point_geom = mrow.geometry
        bearing = calculate_bearing(map_point_geom.y, map_point_geom.x, building_centroid.y, building_centroid.x)
        relative_angle = (bearing - mrow["computed_compass_angle"] + 360) % 360
        target_x = (CENTER_X + (relative_angle * PIXELS_PER_DEGREE)) % IMAGE_WIDTH

        # 7. Optionally draw a red line for debugging
        if config.DRAW_RED_LINE_ON_SVI:
            draw = ImageDraw.Draw(img)
            draw.line([(int(target_x), 0), (int(target_x), IMAGE_HEIGHT)], fill="red", width=5)
            img.save(img_path) # Overwrite with the version that has the line

        # 8. Crop the image to the correct half
        if target_x < CENTER_X:
            cropped = img.crop((0, 0, IMAGE_WIDTH / 2, IMAGE_HEIGHT))
        else:
            cropped = img.crop((IMAGE_WIDTH / 2, 0, IMAGE_WIDTH, IMAGE_HEIGHT))
        
        # 9. Save the cropped chip using the OSM ID for consistent naming
        output_filename = f"{osm_id}.jpg"
        save_path = os.path.join(config.CROPPED_SVI_CHIPS_DIR, output_filename)
        cropped.save(save_path)

    print("\nSVI image chip creation complete.")

if __name__ == "__main__":
    create_svi_chips()