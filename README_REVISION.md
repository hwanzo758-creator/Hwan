# UAV Multispectral Super-Resolution Benchmark

This repository provides the PyTorch code used for the revised manuscript **"Spectral-Consistency-Aware Evaluation of Deep Super-Resolution Methods for UAV Five-Band Multispectral Crop Imagery"** submitted to *Remote Sensing*.

The code evaluates super-resolution (SR) methods for five-band UAV multispectral crop imagery with explicit attention to spatial reconstruction quality, spectral consistency, vegetation-index preservation, computational efficiency, and reviewer-requested robustness checks.

Repository URL: https://github.com/hwanzo758-creator/UAV-Multispectral-Super-Resolution

## What This Release Contains

This release is the **revision benchmark and figure-generation code**. It assumes that the five-band HR/LR cube dataset has already been prepared as `.npy` files. Raw AI Hub TIFF parsing and cube-building scripts are not included in this release.

Main capabilities:

- Joint five-channel SR and band-wise SR evaluation
- SR scales x2, x3, and x4
- Bicubic, SRCNN, EDSR, RCAN, SwinIR-based SR, ESRGAN-based SR, HAT-based SR, and DAT-based SR
- Revision experiments for loss-confounding, parameter capacity, degradation robustness, noise robustness, standard PSNR, excluded-scene reporting, and clipped vegetation-index errors
- Figure generation for RMSE boxplots, RGB/false-color grids, and NDVI/GNDVI/NDRE error maps

## Data

The study uses the AI Hub dataset:

**Field Crop (Cabbage, etc.) Growth Condition Data**  
Dataset code: **71484**  
URL: https://aihub.or.kr/aihubdata/data/view.do?dataSetSn=71484

The AI Hub dataset is **not redistributed** in this repository. Users must download the data directly from AI Hub and follow the AI Hub data-use policy. Preprocessed cubes, model checkpoints, and generated outputs are also excluded because they are large derived artifacts.

## Expected Preprocessed Dataset

`Final_version5_revision.py` expects a preprocessed dataset directory like this:

```text
light_cabbage_training_dataset/
  index/
    scene_index.csv
  HR_npy/
  LR_x2_npy/
  LR_x3_npy/
  LR_x4_npy/
```

Each HR cube is expected to have shape:

```text
(H, W, 5)
```

The channel order is:

```python
cube[:, :, 0] = Blue
cube[:, :, 1] = Green
cube[:, :, 2] = Red
cube[:, :, 3] = RedEdge
cube[:, :, 4] = NIR
```

A template for the scene-index format is provided as `scene_index_template.csv`. The main pipeline reads the actual index from:

```text
DATASET_DIR/index/scene_index.csv
```

If the exact manuscript split can be redistributed, place the final `scene_index.csv` in the dataset `index/` folder and, optionally, add a copy to the repository. If it cannot be redistributed, keep the template and document how the split was regenerated from the AI Hub-derived preprocessed dataset using the fixed seed and filtering rules in the pipeline.

## Repository Structure

```text
.
|-- Final_version5_revision.py
|-- Make_NDVI_hat_dat.py
|-- make_rgb_grids_hat_dat.py
|-- make_fig2_boxplot.py
|-- requirements.txt
|-- scene_index_template.csv
|-- CITATION.cff
|-- LICENSE
|-- README.md
|-- README_REVISION.md
|-- RELEASE_CHECKLIST.md
|-- .gitignore
`-- .gitattributes
```

## Main Scripts

| File | Purpose |
|---|---|
| `Final_version5_revision.py` | Main training/evaluation pipeline for the manuscript revision experiments |
| `Make_NDVI_hat_dat.py` | Generates single-scene NDVI, GNDVI, and NDRE error-map outputs with HAT/DAT |
| `make_rgb_grids_hat_dat.py` | Generates RGB, false-color, and zoom comparison grids with HAT/DAT columns |
| `make_fig2_boxplot.py` | Generates the x4 band-wise per-scene RMSE boxplot from saved metric CSV files |
| `scene_index_template.csv` | Example scene-index schema; not the manuscript split itself |
| `RELEASE_CHECKLIST.md` | Checklist for making the GitHub repository public |
| `CITATION.cff` | Citation metadata template |

## Installation

Create a Python environment and install dependencies:

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

Recommended environment:

```text
Python >= 3.10
PyTorch >= 2.0
CUDA-enabled GPU recommended for training and inference
```

Main dependencies are listed in `requirements.txt`:

```text
numpy
pandas
opencv-python
torch
torchvision
scikit-image
matplotlib
openpyxl
Pillow
tifffile
tqdm
```

## Configure Paths

Before running, edit the path settings near the top of `Final_version5_revision.py`:

```python
DATASET_DIR = Path("/path/to/data/light_cabbage_training_dataset")
OUTPUT_ROOT = Path("/path/to/results")
RUN_NAME = "your_run_name"
CHECKPOINT_RUN_NAME = "existing_checkpoint_run_if_reusing_models"
```

The current script defaults are placeholders or internal run names. They should be changed for a clean reproduction run.

For the figure scripts, also edit the path settings near the top of each file:

```python
# Make_NDVI_hat_dat.py and make_rgb_grids_hat_dat.py
DATASET_DIR = Path("/path/to/data/light_cabbage_training_dataset")
CHECKPOINT_DIR = Path("/path/to/checkpoints")
OUTPUT_DIR = Path("/path/to/single_scene_outputs")

# make_fig2_boxplot.py
METRICS_DIR = "/path/to/results/<run_name>/sr_logs"
```

## Running the Main Benchmark

Choose an experiment preset and SR scales in `Final_version5_revision.py`:

```python
EXPERIMENT_PRESET = "revision_experiments"
USER_SCALES = [2, 3, 4]
```

Then run:

```bash
python Final_version5_revision.py
```

The file currently uses:

```python
EXPERIMENT_PRESET = "full_eval"
USER_SCALES = [4]
```

Those defaults are useful for a narrower evaluation run, but the manuscript revision experiments should be run with the presets below.

## Revision Experiment Presets

| Experiment | Setting |
|---|---|
| Main revision comparison with HAT/DAT, EDSR-wide, A1 loss-confounding check, B3 noise evaluation, and D1/D2/D3 outputs | `EXPERIMENT_PRESET = "revision_experiments"`, `USER_SCALES = [2, 3, 4]` |
| Gaussian-blur plus bicubic degradation robustness | `EXPERIMENT_PRESET = "degradation_b2"`, `DEGRADATION_MODE = "blur_bicubic"`, `USER_SCALES = [4]` |
| HAT/DAT loss-term ablation at x4 | `EXPERIMENT_PRESET = "loss_ablation_hatdat"`, `USER_SCALES = [4]` |
| HAT-only noise robustness rows at x4 | `EXPERIMENT_PRESET = "noise_hat"`, `USER_SCALES = [4]` |

Implemented revision components:

- `hat` and `dat`: recent attention-based SR baselines
- `spatial_no_spectral`: joint model trained without SAM and VI losses to separate reconstruction mode from spectral-loss effects
- `edsr_wide`: widened joint EDSR matched to the band-wise parameter budget
- `DEGRADATION_MODE = "blur_bicubic"`: Gaussian blur followed by bicubic downsampling
- `EVAL_NOISE_SIGMAS`: additive Gaussian noise on x4 LR test inputs
- `PSNR_std`: standard PSNR with peak value 1.0, reported alongside scene-dependent PSNR
- `excluded_scenes_report.csv`: records excluded by the RMSE >= 1 rule
- `*_MAE_clip01`: vegetation-index errors after clipping reflectance to [0, 1]

## Supported SR Models

| Method | Description |
|---|---|
| `bicubic` | Non-learning interpolation baseline |
| `srcnn` | Shallow CNN baseline |
| `edsr` | Enhanced Deep Super-Resolution model |
| `rcan` | Residual Channel Attention Network |
| `swinir` | Compact SwinIR-based SR baseline |
| `esrgan` | ESRGAN-based adversarial SR baseline |
| `hat` | HAT-based attention SR baseline added in the revision |
| `dat` | DAT-based attention SR baseline added in the revision |
| `edsr_wide` | Parameter-matched widened joint EDSR for the capacity check |

## Training and Evaluation Modes

| Mode | Description |
|---|---|
| `joint5ch` | One model processes all five spectral bands jointly |
| `bandwise` | One independent single-channel model is trained per spectral band |

For x3, HR is center-cropped to the natural `LR * 3` grid because 800 pixels is not exactly divisible by 3. The x3 condition is treated as diagnostic; x2 and x4 are the cleaner scales for model ranking.

## Figure Generation

After the metric CSVs, checkpoints, and single-scene SR outputs exist, run:

```bash
python make_fig2_boxplot.py
python Make_NDVI_hat_dat.py
python make_rgb_grids_hat_dat.py
```

Figure scripts are intentionally separated from the main benchmark so that manuscript figures can be regenerated without rerunning the full training pipeline.

## Outputs

Typical outputs under `OUTPUT_ROOT/RUN_NAME/` include:

```text
sr_checkpoints/
sr_logs/
sr_preview/
sr_diff_map/
```

Important log/result files include:

```text
train_history_*.csv
train_history_all.csv
train_history_all.xlsx
run_config*.json
metrics_*_x*.csv
excluded_scenes_report.csv
bad_scene_quality_filter.csv
skipped_nan_inf_records.csv
```

Large generated artifacts are intentionally ignored by Git, including `.npy`, `.pth`, `.pt`, `.ckpt`, `.png`, `.tif`, `sr_checkpoints/`, `sr_logs/`, and `*_SR_RESULTS/`.

## Reproducibility Notes

- Fixed random seed: 42
- LR cubes are generated from HR cubes using OpenCV area interpolation unless a different degradation mode is selected
- Invalid NaN/Inf records and quality-filtered scenes are excluded before splitting
- The fixed scene-quality scale set is controlled by `SCENE_QUALITY_SCALES = [2, 3, 4]` so different revision runs use the same valid-record pool
- Test evaluation uses the held-out test split by default
- Records with `RMSE >= 1` are excluded from paper summaries and written to `excluded_scenes_report.csv`
- Vegetation-index metrics include NDVI, GNDVI, and NDRE

## Citation and Acknowledgement

If you use this code, please cite the associated manuscript and acknowledge AI Hub dataset code 71484 according to the AI Hub usage policy.

The code is released under the MIT License. The AI Hub dataset is not covered by this license and must be obtained directly from AI Hub.