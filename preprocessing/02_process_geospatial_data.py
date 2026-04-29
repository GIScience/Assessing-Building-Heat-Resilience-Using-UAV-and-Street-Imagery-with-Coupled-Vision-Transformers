# preprocessing/02_process_geospatial_data.py
import os
import sys
import geopandas as gpd
import pandas as pd
import numpy as np
from shapely.geometry import LineString
from scipy.spatial import cKDTree
from tqdm import tqdm

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import settings from the configuration file
import config

def process_osm_data(bounds_gdf):
    """Buffers roads and finds intersecting buildings from OSM data."""
    print("--> Stage 1: Processing OSM data to find buildings near roads...")
    
    # Load the single, pre-filtered buildings file
    buildings = gpd.read_file(config.OSM_BUILDINGS_FILE).to_crs(bounds_gdf.crs)

    # Load roads and buffer
    roads = gpd.read_file(config.OSM_ROADS_FILE).to_crs(bounds_gdf.crs)
    buffered_roads = roads.buffer(config.ROAD_BUFFER_METERS)

    # Find intersecting buildings
    intersecting_buildings_idx = buildings.sindex.query(buffered_roads, predicate="intersects")
    intersecting_buildings = buildings.iloc[np.unique(intersecting_buildings_idx)]
    
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)
    intersecting_buildings.to_file(config.INTERSECTING_BUILDINGS_FILE, driver="GPKG")
    print(f"    Found {len(intersecting_buildings)} buildings within {config.ROAD_BUFFER_METERS}m of roads.")
    return intersecting_buildings

def analyze_visibility(buildings_gdf, points_gdf):
    """Performs nearest neighbor and line-of-sight visibility analysis."""
    print("\n--> Stage 2: Analyzing visibility between buildings and Mapillary points...")
    
    if points_gdf is None or buildings_gdf.empty:
        print("    Skipping visibility analysis due to missing data.")
        return
        
    # --- Nearest Neighbor Analysis ---
    points_gdf = points_gdf.to_crs(buildings_gdf.crs)
    buildings_gdf['centroid'] = buildings_gdf.geometry.centroid
    
    building_coords = np.array([pt.coords[0] for pt in buildings_gdf.centroid])
    point_coords = np.array([pt.coords[0] for pt in points_gdf.geometry])
    
    tree = cKDTree(point_coords)
    distances, indices = tree.query(building_coords, distance_upper_bound=config.NEIGHBOR_SEARCH_RADIUS_METERS)
    
    buildings_gdf['mapillary_id'] = [points_gdf.iloc[i]['id'] if np.isfinite(d) else None for i, d in zip(indices, distances)]
    buildings_gdf['point_geometry'] = [points_gdf.iloc[i]['geometry'] if np.isfinite(d) else None for i, d in zip(indices, distances)]
    buildings_gdf['distance_m'] = distances
    
    nearby_gdf = buildings_gdf[buildings_gdf['distance_m'] <= config.NEARBY_DISTANCE_METERS].copy()
    print(f"    Found {len(nearby_gdf)} buildings within {config.NEARBY_DISTANCE_METERS}m of a Mapillary point.")

    # --- Line-of-Sight Analysis ---
    # We can use the already loaded buildings_gdf as the obstruction layer
    all_buildings_sindex = buildings_gdf.sindex
    visibility = []
    for _, row in tqdm(nearby_gdf.iterrows(), total=len(nearby_gdf), desc="  Checking visibility"):
        line = LineString([row['point_geometry'], row['centroid']])
        possible_matches_idx = list(all_buildings_sindex.intersection(line.bounds))
        obstructors = buildings_gdf.iloc[possible_matches_idx]
        
        # Check if any building *other than the target building itself* intersects the line
        is_obstructed = obstructors[obstructors['osm_id'] != row['osm_id']].intersects(line).any()
        visibility.append(not is_obstructed)

    nearby_gdf['is_visible'] = visibility
    visible_buildings = nearby_gdf[nearby_gdf['is_visible']].copy()
    
    # Clean up final columns
    visible_buildings = visible_buildings[['osm_id', 'mapillary_id', 'distance_m', 'geometry']].rename(columns={'geometry': 'building_geometry'})
    visible_buildings.to_file(config.VISIBLE_BUILDINGS_FILE, driver="GPKG")
    print(f"    Found {len(visible_buildings)} visible buildings. Final results saved to {config.VISIBLE_BUILDINGS_FILE}")


if __name__ == "__main__":
    if not os.path.exists(config.COMBINED_MAPILLARY_FILE):
        print("Error: Combined Mapillary data not found. Please run '01_fetch_mapillary_data.py' first.")
    else:
        bounds_data = gpd.read_file(config.MSIMBAZI_EXTENT_FILE)
        mapillary_points = gpd.read_file(config.COMBINED_MAPILLARY_FILE)
        
        buildings_near_roads = process_osm_data(bounds_data)
        analyze_visibility(buildings_near_roads, mapillary_points)
        
        print("\nGeospatial processing complete.")