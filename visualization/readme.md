# Visualization

Grad-CAM attention-map scripts for interpreting trained GCViT models. Run from the project root after training is complete.

See the [root README](../README.md) for pipeline context.

---

## Directory Structure

- `gradcam_dual_input.py`: Generates per-branch (SVI + UAV) Grad-CAM overlays from a dual-input checkpoint.
- `gradcam_single_input.py`: Generates a single Grad-CAM overlay from a single-input checkpoint.

---

## Scripts

### `gradcam_dual_input.py`

- **Purpose:** Computes Grad-CAM heatmaps for both the SVI and UAV branches of a trained dual-input model, producing a 2×2 visualization (original image + attention overlay per branch) with the predicted class and confidence.
- **Arguments:**
  - `--weights_path` (required): Path to a trained `.h5` checkpoint
  - `--svi_path` (required): Path to an SVI image chip
  - `--uav_path` (required): Path to a UAV image chip
  - `--class_names` (required): Space-separated class names matching the model output layer
  - `--output_name` (required): Filename for the saved visualization
- **Inputs:**
  - Trained dual-input checkpoint from `output/training_results/`
  - One SVI chip and one UAV chip for the same building
- **Outputs:**
  - `output/visualizations/<output_name>`
- **Usage:**
  ```bash
  python visualization/gradcam_dual_input.py \
    --weights_path output/training_results/<run_dir>/material_rooftop_fold_1_best.h5 \
    --svi_path output/cropped_svi/12345.jpg \
    --uav_path output/cropped_uav/12345.tif \
    --class_names concrete metal tarpaulin \
    --output_name dual_material_rooftop_gradcam.png
  ```

### `gradcam_single_input.py`

- **Purpose:** Computes a Grad-CAM heatmap for a single-input GCViT model, producing a side-by-side visualization (original + attention overlay) with per-class confidence scores.
- **Arguments:**
  - `--weights_path` (required): Path to a trained `.h5` checkpoint
  - `--image_path` (required): Path to the input image chip (SVI or UAV)
  - `--class_names` (required): Space-separated class names matching the model output layer
  - `--output_name` (required): Filename for the saved visualization
- **Inputs:**
  - Trained single-input checkpoint from `output/training_results/`
  - One image chip
- **Outputs:**
  - `output/visualizations/<output_name>`
- **Usage:**
  ```bash
  python visualization/gradcam_single_input.py \
    --weights_path output/training_results/<run_dir>/vegetation_svi_fold_1_best.h5 \
    --image_path output/cropped_svi/12345.jpg \
    --class_names yes no \
    --output_name single_vegetation_gradcam.png
  ```

---
