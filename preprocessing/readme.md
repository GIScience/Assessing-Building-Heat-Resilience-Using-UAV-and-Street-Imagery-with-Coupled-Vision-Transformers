# Preprocessing

Numbered scripts that transform raw geospatial data into model-ready image chips and CSV datasets. Run sequentially from the project root. All paths are configured in [`config.py`](../config.py).

See the [root README](../README.md) for overall pipeline context.

---

## Directory Structure

- `01_fetch_mapillary_data.py`: Fetches Mapillary image metadata for the AOI.
- `02_process_geospatial_data.py`: Filters and selects visible buildings.
- `03_create_uav_data.py`: Generates UAV image chips.
- `04_create_svi_data.py`: Downloads panoramas and generates SVI image chips.
- `04b_generate_centerline_data.py`: Computes building-to-panorama viewing geometry.
- `05_fetch_and_engineer_features.py`: Fetches detections and engineers features.
- `06a_generate_classification_data.py`: Builds classification dataset from labels.
- `06b_generate_regression_data.py`: Builds regression dataset from features.
- `utils/`: Helper modules (`api_helpers.py`, `feature_engineering.py`).

---

## Scripts

### `01_fetch_mapillary_data.py`

- **Purpose:** Queries the Mapillary images API to retrieve panoramic image metadata (location, compass angle, ID) for all images intersecting the AOI.
- **Inputs:**
  - `data/Msimbazi_extent.geojson`
  - `.env` (must define `MAPILLARY_ACCESS_TOKEN`)
- **Outputs:**
  - `output/mapillary_points_combined.gpkg`
- **Usage:** `python preprocessing/01_fetch_mapillary_data.py`

### `02_process_geospatial_data.py`

- **Purpose:** Identifies buildings within a configurable distance of roads, then retains only those visible from at least one Mapillary camera position using line-of-sight analysis.
- **Inputs:**
  - `data/OSM_residential_buildings.gpkg`
  - `data/OSM_roads.gpkg`
  - `data/Msimbazi_extent.geojson`
  - `output/mapillary_points_combined.gpkg`
- **Outputs:**
  - `output/buildings_near_roads.gpkg`
  - `output/visible_buildings_final.gpkg`
- **Usage:** `python preprocessing/02_process_geospatial_data.py`

### `03_create_uav_data.py`

- **Purpose:** Crops the UAV orthomosaic to each visible building footprint, applying a polygon mask to isolate the roof area. Produces one GeoTIFF chip per building.
- **Inputs:**
  - `output/visible_buildings_final.gpkg`
  - `data/Msimbazi_image.tif`
- **Outputs:**
  - `output/cropped_uav/*.tif` 
- **Usage:** `python preprocessing/03_create_uav_data.py`

### `04_create_svi_data.py`

- **Purpose:** Downloads equirectangular panoramas from Mapillary for each visible building, determines which half of the panorama faces the building, and crops it into a single JPEG chip.
- **Inputs:**
  - `output/visible_buildings_final.gpkg`
  - `output/mapillary_points_combined.gpkg`
  - Mapillary image download API
- **Outputs:**
  - `output/svi_panoramas/*.jpg` (full panoramas, intermediate)
  - `output/cropped_svi/*.jpg` 
- **Usage:** `python preprocessing/04_create_svi_data.py`

### `04b_generate_centerline_data.py`

- **Purpose:** Projects each building's road-facing centerline onto corresponding Mapillary panorama pixel coordinates. Outputs the panorama ID, x-pixel position, and left/right half assignment per building.
- **Inputs:**
  - `output/visible_buildings_final.gpkg`
  - `output/mapillary_points_combined.gpkg`
- **Outputs:**
  - `output/centerline_data.csv`
- **Usage:** `python preprocessing/04b_generate_centerline_data.py`

### `05_fetch_and_engineer_features.py`

- **Purpose:** Retrieves Mapillary object detections for each building's associated image, then computes engineered features: wall brightness (from masked SVI region), roof brightness (from UAV chip), vegetation presence, and per-channel color statistics.
- **Inputs:**
  - `output/visible_buildings_final.gpkg`
  - `output/centerline_data.csv`
  - `output/cropped_uav/*.tif`
  - `output/cropped_svi/*.jpg`
  - Mapillary detections API
- **Outputs:**
  - `output/mapillary_detections/*.csv`
  - `output/features_master.csv`
- **Usage:** `python preprocessing/05_fetch_and_engineer_features.py`

### `06a_generate_classification_data.py`

- **Purpose:** Joins `features_master.csv` with manually curated `data/osm_labels.csv`, copies corresponding SVI and UAV chips into a label-organized directory tree (`labels_data/<modality>/<task>/<label>/`), and writes the final classification CSV.
- **Inputs:**
  - `output/features_master.csv`
  - `data/osm_labels.csv`
  - `output/cropped_svi/*.jpg`
  - `output/cropped_uav/*.tif`
- **Outputs:**
  - `output/labels_data/svi/<task>/<label>/*.jpg`
  - `output/labels_data/uav/<task>/<label>/*.tif`
  - `output/CV_classdata.csv`
- **Usage:** `python preprocessing/06a_generate_classification_data.py`

### `06b_generate_regression_data.py`

- **Purpose:** Filters `features_master.csv` to buildings with valid brightness values and existing image chips on disk, then writes the regression CSV.
- **Inputs:**
  - `output/features_master.csv`
  - `output/cropped_svi/*.jpg`
  - `output/cropped_uav/*.tif`
- **Outputs:**
  - `output/CV_regression.csv`
- **Usage:** `python preprocessing/06b_generate_regression_data.py`
    ### Manual Labeling (for Step 06a)

    Create `data/osm_labels.csv` before running step 06a. Each row should correspond to a unique building (by `osm_id`) and its associated SVI and UAV image chips, with the following example columns:

    | fid | osm_id    | structural_openness | number_of_floors | vegetation | material_rooftop | material_wall |
    |-----|-----------|---------------------|------------------|------------|------------------|--------------|
    | 1   | 338449341 | closed_structure    | one              | yes        | metal            | concrete     |
    ... | ... | ...       | ...                 | ...              | ...        | ...              | ...          |
    ---

    For each building, the corresponding SVI and UAV image chips are named as `{osm_id}.jpg` and `{osm_id}.tif` and are located in the appropriate folders (e.g., `output/cropped_svi/`, `output/cropped_uav/`, or under `labels_data/`).

    Refer to `data/osm_labels.csv` for the full set of label values used in the pipeline.

---

###  `06b_generate_regression_data.py` 
python preprocessing/06b_generate_regression_data.py
- **Purpose:** Filters `features_master.csv` to buildings with valid brightness values and existing image chips on disk, then writes the regression CSV.
- **Inputs:**
  - `output/features_master.csv`
  - `output/cropped_svi/*.jpg`
  - `output/cropped_uav/*.tif`
- **Outputs:**
  - `output/CV_regression.csv`
- **Usage:** `python preprocessing/06b_generate_regression_data.py`




> **Note:** While the pipeline originates from raw GeoTIFFs (`.tif`) and JPEGs (`.jpg`), all chips in the `output/` subfolders are standardized to **`.png`** during earlier preprocessing steps to ensure consistency for the modeling stage.