# Training

Model training scripts for k-fold cross-validated classification and regression with GCViT backbones. Run from the project root inside an [Enroot](https://github.com/NVIDIA/enroot) GPU container.

See the [root README](../README.md) for container setup and full pipeline context.

---

## Directory Structure

- `train_classifier_dual_input.py`: Dual-input (SVI + UAV) classifier — primary CGCViT model.
- `train_classifier_single_input.py`: Single-input (SVI or UAV) classifier — ablation baseline.
- `train_regressor_dual_input.py`: Dual-input regressor for continuous brightness targets.

---

## Scripts

### `train_classifier_dual_input.py`

- **Purpose:** Trains a Coupled GCViT (CGCViT) classifier that fuses SVI and UAV image streams to predict categorical building attributes.
- **Arguments:**
  - `--task` (required): Classification task key (e.g., `material_rooftop`, `structural_openness`, `vegetation`)
  - `--backbone` (default: `Tiny`): GCViT variant (`XXTiny` | `XTiny` | `Tiny` | `Small` | `Base` | `Large`)
  - `--epochs` (default: `50`): Training epochs per fold
  - `--batch_size` (default: `8`): Batch size before distribution strategy scaling
  - `--k_folds` (default: `5`): Number of cross-validation folds
  - `--test_size` (default: `0.15`): Held-out test split fraction
  - `--min_samples_per_class` (default: `5`): Minimum samples to retain a class
  - `--augmentation_strength` (default: `0.3`): Augmentation intensity (0.0–1.0)
- **Inputs:**
  - `output/CV_classdata.csv`
  - SVI and UAV image chips referenced by the CSV
- **Outputs:** `output/training_results/dual_input_classifier_<uuid>/`
  - `<task>_fold_<N>_best.h5` — best validation checkpoint per fold
  - `<task>_cv_results.json` — aggregated metrics (accuracy, F1, confusion matrices)
  - `<task>_fold_<N>_history.png` — per-fold training curves
- **Usage:**
  ```bash
  enroot start --root --rw --mount "$HOME:/mnt/host" pyxis_tensorflow \
    sh -c "cd /mnt/host/heat_cv && python training/train_classifier_dual_input.py \
      --task material_rooftop --backbone Tiny --epochs 100 --batch_size 8 \
      --k_folds 5 --test_size 0.15 --augmentation_strength 0.3"
  ```

### `train_classifier_single_input.py`

- **Purpose:** Trains a single-stream GCViT classifier on either SVI or UAV images alone. Used as an ablation baseline against the dual-input model.
- **Arguments:**
  - `--task` (required): Classification task key
  - `--input_type` (required): `svi` or `uav`
  - `--backbone` (default: `Tiny`): GCViT variant
  - `--epochs` (default: `50`): Training epochs per fold
  - `--batch_size` (default: `8`): Batch size
  - `--k_folds` (default: `5`): Number of CV folds
  - `--test_size` (default: `0.15`): Held-out test fraction
  - `--min_samples_per_class` (default: `5`): Minimum class samples
  - `--augmentation_strength` (default: `0.3`): Augmentation intensity
- **Inputs:**
  - `output/CV_classdata.csv`
  - Image chips for the selected input type
- **Outputs:** `output/training_results/single_input_classifier_<uuid>/`
  - `<task>_<input_type>_fold_<N>_best.h5`
  - `<task>_<input_type>_cv_results.json`
- **Usage:**
  ```bash
  enroot start --root --rw --mount "$HOME:/mnt/host" pyxis_tensorflow \
    sh -c "cd /mnt/host/heat_cv && python training/train_classifier_single_input.py \
      --task vegetation --input_type svi --backbone Tiny --epochs 100 \
      --batch_size 8 --k_folds 5 --test_size 0.15"
  ```

### `train_regressor_dual_input.py`

- **Purpose:** Trains a dual-input CGCViT regressor to predict continuous surface brightness values (wall or roof).
- **Arguments:**
  - `--target_column` (required): `wall_brightness` or `roof_brightness`
  - `--backbone` (default: `Tiny`): GCViT variant
  - `--epochs` (default: `50`): Training epochs per fold
  - `--batch_size` (default: `8`): Batch size
  - `--k_folds` (default: `5`): Number of CV folds
  - `--test_size` (default: `0.15`): Held-out test fraction
  - `--augmentation_strength` (default: `0.3`): Augmentation intensity
- **Inputs:**
  - `output/CV_regression.csv`
  - SVI and UAV image chips referenced by the CSV
- **Outputs:** `output/training_results/regressor_<uuid>/`
  - `<target_column>_fold_<N>_best.h5`
  - `<target_column>_kfold_regression_results.json` — MAE, RMSE, R² per fold
- **Usage:**
  ```bash
  enroot start --root --rw --mount "$HOME:/mnt/host" pyxis_tensorflow \
    sh -c "cd /mnt/host/heat_cv && python training/train_regressor_dual_input.py \
      --target_column wall_brightness --epochs 100 --batch_size 8 \
      --k_folds 5 --test_size 0.15 --augmentation_strength 0.3"
  ```

---

<!-- ## Cross-Validation Strategy

| Script | Splitter | Stratification |
|--------|----------|----------------|
| `train_classifier_dual_input.py` | `StratifiedKFold` | By class label |
| `train_classifier_single_input.py` | `StratifiedKFold` | By class label |
| `train_regressor_dual_input.py` | `KFold` | None (continuous target) |

--- -->

