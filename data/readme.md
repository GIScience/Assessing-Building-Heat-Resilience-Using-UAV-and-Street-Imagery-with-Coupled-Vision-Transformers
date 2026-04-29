# Data

Raw input files used by the [preprocessing pipeline](../preprocessing/readme.md). 
---

## Files

### `Msimbazi_extent.geojson`

- **Purpose:** Defines the area of interest (AOI) boundary polygon.
- **Source:** Prepared in QGIS from the UAV orthomosaic extent
- **Used by:** `01_fetch_mapillary_data.py`, `02_process_geospatial_data.py`

### `Msimbazi_image.tif`

- **Purpose:** UAV orthomosaic raster covering the AOI, used to generate per-building UAV chips.
- **Source:** [OpenAerialMap](https://openaerialmap.org) API.
- **Used by:** `03_create_uav_data.py`

### `OSM_residential_buildings.gpkg`

- **Purpose:** Residential building footprints within the AOI (polygon geometries with `osm_id`).
- **Source:** [Geofabrik OSM extract](https://download.geofabrik.de/africa/tanzania.html), filtered and exported to GeoPackage.
- **Used by:** `02_process_geospatial_data.py`, `06a_generate_classification_data.py`

### `OSM_roads.gpkg`

- **Purpose:** Road geometries used for road-proximity filtering of buildings.
- **Source:** Geofabrik OSM extract, filtered and exported to GeoPackage.
- **Used by:** `02_process_geospatial_data.py`

### `osm_labels.csv`

- **Purpose:** Manually curated ground-truth labels for classification dataset generation.
- **Source:** Manual annotation (~2 000 building–task pairs recommended).
- **Format:** See [labeling specification](../preprocessing/readme.md#manual-labeling-for-step-06a).
- **Used by:** `06a_generate_classification_data.py` , Required only for classification; regression workflows run without it.

