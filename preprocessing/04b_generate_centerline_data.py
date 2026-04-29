#!/usr/bin/env python3
"""
Generate centerline_data.csv

This script computes per-building viewing geometry by calculating where each
building's centerline projects onto the corresponding Mapillary panorama image.

The output CSV contains:
- osm_id: OpenStreetMap building identifier
- mapillary_id: Mapillary image identifier where the building is visible
- line_x_pixel: x-coordinate in the panorama where the building centerline appears
- half_cropped_image: 'left' or 'right' to indicate which half of the panorama is used

Output: output/centerline_data.csv
"""

import os
import pandas as pd
import geopandas as gpd
import numpy as np
from shapely.geometry import Point, LineString
from tqdm import tqdm
import config


def compute_building_to_panorama_projection(building_geom, mapillary_point, image_width=config.IMAGE_WIDTH):
    """
    Project a building's centerline onto a panorama's pixel space.
    
    The projection maps the building's direction (relative to the street-view point)
    to a horizontal pixel coordinate in the panorama image.
    
    Returns:
    - line_x_pixel: x-coordinate in panorama (0 to image_width)
    - half: 'left' if x < image_width/2, else 'right'
    """
    # Get building centroid
    building_center = building_geom.centroid
    
    # Vector from viewpoint to building center
    direction = np.array([
        building_center.x - mapillary_point.x,
        building_center.y - mapillary_point.y
    ])
    
    # Avoid division by zero
    if np.linalg.norm(direction) < 1e-6:
        return None, None
    
    # Normalize to unit vector
    direction = direction / np.linalg.norm(direction)
    
    # Compute bearing (angle from north, in degrees)
    # In equirectangular panoramas, bearing direction maps to pixel x-coordinate
    bearing_rad = np.arctan2(direction[0], direction[1])
    bearing_deg = np.degrees(bearing_rad)
    
    # Normalize bearing to [0, 360)
    bearing_deg = bearing_deg % 360
    
    # Map bearing to panorama pixel coordinate
    # Assuming standard Mapillary panorama: 360° mapped to panorama width
    line_x_pixel = (bearing_deg / 360.0) * image_width
    
    # Determine if building is in left or right half
    half = 'right' if line_x_pixel >= image_width / 2 else 'left'
    
    return line_x_pixel, half


def generate_centerline_data():
    """
    Generate centerline_data.csv by computing viewing geometry for each building-panorama pair.
    """
    print("\n=== Generating Building Centerline Data ===\n")
    
    # Load input data
    print("Loading visible buildings and Mapillary points...")
    buildings_gdf = gpd.read_file(config.VISIBLE_BUILDINGS_FILE)
    mapillary_gdf = gpd.read_file(config.MAPILLARY_POINTS_FILE)
    
    # Ensure both GeoDataFrames have matching CRS
    if buildings_gdf.crs != mapillary_gdf.crs:
        mapillary_gdf = mapillary_gdf.to_crs(buildings_gdf.crs)
    
    print(f"  Buildings: {len(buildings_gdf)}")
    print(f"  Mapillary points: {len(mapillary_gdf)}")
    
    # Build a list of building-to-panorama associations
    centerline_records = []
    
    print("\nComputing projections...")
    for building_idx, building_row in tqdm(buildings_gdf.iterrows(), total=len(buildings_gdf), desc="  Buildings"):
        building_geom = building_row.geometry
        osm_id = building_row.get('osm_id', building_row.get('id', None))
        mapillary_id = building_row.get('mapillary_id', None)
        
        if pd.isna(mapillary_id) or osm_id is None:
            continue
        
        # Find the corresponding Mapillary point
        mapillary_row = mapillary_gdf[mapillary_gdf['id'] == mapillary_id]
        if mapillary_row.empty:
            continue
        
        mapillary_point = mapillary_row.iloc[0].geometry
        
        # Compute projection
        line_x_pixel, half_cropped_image = compute_building_to_panorama_projection(building_geom, mapillary_point)
        
        if line_x_pixel is None:
            continue
        
        centerline_records.append({
            'osm_id': osm_id,
            'mapillary_id': mapillary_id,
            'line_x_pixel': line_x_pixel,
            'half_cropped_image': half_cropped_image
        })
    
    # Create DataFrame and save
    print(f"\nGenerated {len(centerline_records)} building-panorama associations.")
    
    centerline_df = pd.DataFrame(centerline_records)
    centerline_df.to_csv(config.CENTERLINE_DATA_FILE, index=False)
    print(f"✅ Centerline data saved to {config.CENTERLINE_DATA_FILE}")
    
    return centerline_df


if __name__ == "__main__":
    generate_centerline_data()
