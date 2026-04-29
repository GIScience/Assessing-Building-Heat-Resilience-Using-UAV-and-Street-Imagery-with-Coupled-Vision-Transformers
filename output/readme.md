# Output




## Directory Structure

- `cropped_uav/`: UAV image chips.
- `cropped_svi/`: Cropped SVI chips.
- `svi_panoramas/`: Full equirectangular panoramas from mapillary.
- `labels_data/`: Label-organized image tree for classification tasks .
- `mapillary_detections/`: Per-building object-detection CSVs from Mapillary API.
- `training_results/`: Model checkpoints, metrics, and training plots.
- `visualizations/`: Grad-CAM attention overlays.

---

## Generated Files

### `mapillary_points_combined.gpkg`

- **Produced by:** `01_fetch_mapillary_data.py`
- **Contents:** Geospatial metadata (location, compass angle, image ID) for all Mapillary images within the AOI.

### `buildings_near_roads.gpkg`

- **Produced by:** `02_process_geospatial_data.py`
- **Contents:** Building footprints filtered to those within configurable distance of a road.

### `visible_buildings_final.gpkg`

- **Produced by:** `02_process_geospatial_data.py`
- **Contents:** Subset of road-adjacent buildings confirmed visible from at least one Mapillary camera position.

### `centerline_data.csv`

- **Produced by:** `04b_generate_centerline_data.py`
- **Contents:** Per-building viewing geometry — panorama ID, x-pixel position, and left/right half assignment.

### `features_master.csv`

- **Produced by:** `05_fetch_and_engineer_features.py`
- **Contents:** Consolidated engineered-feature table. Key columns:

  | Column | Type | Description |
  |--------|------|-------------|
  | `osm_id` | int | OpenStreetMap building ID |
  | `mapillary_id` | str | Associated Mapillary image ID |
  | `roof_brightness` | float | Mean pixel intensity of masked UAV chip |
  | `wall_brightness` | float | Mean pixel intensity of wall region in SVI chip |
  | `vegetation_presence` | float | Vegetation detection score (0–1) |
  <!-- | `uav_mean_r/g/b` | float | Per-channel mean of UAV chip | -->

### `CV_classdata.csv`

- **Produced by:** `06a_generate_classification_data.py`
- **Contents:** Model-ready classification dataset with columns `svi_path` and `uav_path` pointing to label-organized chips under `labels_data/`.

### `CV_regression.csv`

- **Produced by:** `06b_generate_regression_data.py`
- **Contents:** Model-ready regression dataset with columns `svi_path`, `uav_path`, `wall_brightness`, and `roof_brightness`.

---
