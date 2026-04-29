# preprocessing/01_fetch_mapillary_data.py
import os
import time
import sys
import requests
import geopandas as gpd
import pandas as pd
from shapely.geometry import box, Point
from tqdm import tqdm
import tempfile

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import settings from the configuration file
import config

def create_grid_tiles(bounds_gdf, tile_size_m):
    """Creates a grid of square tiles covering the bounds for mapillary data fetching."""
    print("--> Stage 1: Creating grid tiles...")
    minx, miny, maxx, maxy = bounds_gdf.total_bounds
    tiles = []
    x = minx
    while x < maxx:
        y = miny
        while y < maxy:
            tiles.append(box(x, y, x + tile_size_m, y + tile_size_m))
            y += tile_size_m
        x += tile_size_m

    tiles_gdf = gpd.GeoDataFrame(geometry=tiles, crs=bounds_gdf.crs)
    tiles_gdf['tile_id'] = range(len(tiles_gdf))
    
    intersecting_tiles = gpd.sjoin(tiles_gdf, bounds_gdf, predicate="intersects", how="inner")
    print(f"    Generated {len(intersecting_tiles)} tiles that intersect the Msimbazi extent.")
    return intersecting_tiles.drop(columns=["index_right"])

def fetch_and_combine_mapillary_data(tiles_gdf):
    """Fetches and combines Mapillary data using a temporary directory for tiles."""
    print("\n--> Stage 2: Fetching and combining Mapillary data...")
    
    with tempfile.TemporaryDirectory() as temp_dir:
        print(f"    Using temporary directory for tiles: {temp_dir}")
        for _, row in tqdm(tiles_gdf.to_crs(epsg=4326).iterrows(), total=len(tiles_gdf), desc="  Fetching tiles"):
            tile_id = row['tile_id']
            output_path = os.path.join(temp_dir, f"tile_{tile_id}.gpkg")

            minx, miny, maxx, maxy = row.geometry.bounds
            bbox_str = f"{minx},{miny},{maxx},{maxy}"
            
            url = (
                f"https://graph.mapillary.com/images?access_token={config.ACCESS_TOKEN}"
                f"&creator_username={config.CREATOR_USERNAME}&bbox={bbox_str}&fields={config.MAPILLARY_FIELDS}&limit=2000"
            )

            try:
                response = requests.get(url, timeout=30)
                if response.status_code == 200:
                    data = response.json().get("data", [])
                    if data:
                        images = [
                            {**item, 'geometry': Point(item.get("computed_geometry", {}).get("coordinates"))}
                            for item in data if item.get("computed_geometry", {}).get("coordinates")
                        ]
                        if images:
                            gpd.GeoDataFrame(images, crs="EPSG:4326").to_file(output_path, driver="GPKG")
                else:
                    print(f"    Warning: Non-200 response ({response.status_code}) for tile {tile_id}")
            except requests.RequestException as e:
                print(f"    Error on tile {tile_id}: {e}")
            
            time.sleep(0.6)

        # Merge all created GPKG files from the temporary directory
        gpkg_files = [os.path.join(temp_dir, f) for f in os.listdir(temp_dir) if f.endswith(".gpkg")]
        if not gpkg_files:
            print("    No Mapillary data found.")
            return None
            
        combined_gdf = pd.concat([gpd.read_file(f) for f in gpkg_files], ignore_index=True)
        combined_gdf = gpd.GeoDataFrame(combined_gdf.drop(columns=['geometry']), geometry=[Point(g['coordinates']) for g in combined_gdf['geometry']], crs="EPSG:4326")
    
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)
    combined_gdf.to_file(config.COMBINED_MAPILLARY_FILE, driver="GPKG")
    print(f"    Saved {len(combined_gdf)} total Mapillary points to {config.COMBINED_MAPILLARY_FILE}")
    return combined_gdf

if __name__ == "__main__":
    if not config.ACCESS_TOKEN or config.ACCESS_TOKEN == "YOUR_TOKEN_HERE":
        print("Error: Mapillary Access Token not set. Please update your .env file.")
    else:
        bounds_data = gpd.read_file(config.MSIMBAZI_EXTENT_FILE)
        tiles_to_process = create_grid_tiles(bounds_data, config.TILE_SIZE_METERS)
        fetch_and_combine_mapillary_data(tiles_to_process)
        print("\nMapillary data fetching complete.")