# preprocessing/03_create_uav_image_chips.py
import os
import sys
import geopandas as gpd
import numpy as np
import rasterio
from rasterio.mask import mask
from rasterio.features import geometry_mask
from shapely.geometry import box, mapping
from tqdm import tqdm

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import settings from the configuration file
import config

def create_uav_chips():
    """
    Crops the main UAV orthomosaic raster into individual, masked image chips
    for each visible building.
    """
    print("--- Stage 3: Creating UAV Image Chips ---")

    # 1. Load the visible buildings GeoDataFrame (output from the previous script)
    try:
        buildings = gpd.read_file(config.VISIBLE_BUILDINGS_FILE)
    except Exception as e:
        print(f"❌ Error: Could not read the visible buildings file at '{config.VISIBLE_BUILDINGS_FILE}'.")
        print(f"   Please ensure '02_process_geospatial_data.py' has been run successfully.")
        print(f"   Details: {e}")
        return

    # 2. Ensure the output directory exists
    os.makedirs(config.CROPPED_UAV_CHIPS_DIR, exist_ok=True)
    print(f"    Outputting image chips to: {config.CROPPED_UAV_CHIPS_DIR}")

    # 3. Calculate the side length for a square crop area
    #    based on the largest building's area to ensure all buildings fit.
    buildings['area'] = buildings.geometry.area
    max_area = buildings['area'].max()
    side_length = np.sqrt(max_area) 

    # 4. Open the source UAV raster
    with rasterio.open(config.UAV_RASTER_FILE) as src:
        # Re-project buildings to match the raster's CRS
        buildings = buildings.to_crs(src.crs)

        for _, row in tqdm(buildings.iterrows(), total=len(buildings), desc="  Processing buildings"):
            geom = row.geometry
            osm_id = str(row.get("osm_id", "unknown_id"))
            output_filename = f"{osm_id}.tif"
            output_path = os.path.join(config.CROPPED_UAV_CHIPS_DIR, output_filename)

            if geom is None or geom.is_empty:
                continue

            # 5. Create a square bounding box around the building's centroid
            centroid = geom.centroid
            half_side = side_length / 2
            square_geom = box(
                centroid.x - half_side, centroid.y - half_side,
                centroid.x + half_side, centroid.y + half_side
            )

            try:
                # 6. Crop the main raster to the square area
                out_image, out_transform = mask(
                    src, [mapping(square_geom)], crop=True, all_touched=False
                )

                # 7. Create a boolean mask of the actual building geometry within the cropped area
                building_mask = geometry_mask(
                    [mapping(geom)],
                    out_shape=out_image.shape[1:], # (height, width)
                    transform=out_transform,
                    invert=True # True for pixels inside the geometry
                )

                # 8. Set all pixels *outside* the building mask to 0 (black)
                masked_image = out_image.copy()
                masked_image[:, ~building_mask] = 0

                # 9. Save the final masked image chip
                out_meta = src.meta.copy()
                out_meta.update({
                    "height": masked_image.shape[1],
                    "width": masked_image.shape[2],
                    "transform": out_transform,
                    "nodata": 0 # Set nodata value to 0
                })

                with rasterio.open(output_path, "w", **out_meta) as dest:
                    dest.write(masked_image)

            except Exception as e:
                print(f"    ❌ Skipping {osm_id}: {e}")

    print("\nUAV image chip creation complete.")

if __name__ == "__main__":
    create_uav_chips()