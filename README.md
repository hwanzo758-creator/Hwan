# UAV Multispectral Super-Resolution Benchmark

This repository provides the PyTorch code used for the revised manuscript **"Spectral-Consistency-Aware Evaluation of Deep Super-Resolution Methods for UAV Five-Band Multispectral Crop Imagery"** submitted to *Remote Sensing*.

The code evaluates super-resolution (SR) methods for five-band UAV multispectral crop imagery with explicit attention to spatial reconstruction quality, spectral consistency, vegetation-index preservation, computational efficiency, and reviewer-requested robustness checks.

Repository URL: https://github.com/hwanzo758-creator/UAV-Multispectral-Super-Resolution

## What This Release Contains

This release contains the **benchmark, the cube-building script and the figure-generation code**, together with the scene index used in the manuscript. The AI Hub imagery itself is not redistributed.

Main capabilities:

- Joint five-channel SR and band-wise SR evaluation
- SR scales x2, x3, and x4
- Bicubic, SRCNN, EDSR, RCAN, SwinIR-based SR, ESRGAN-based SR, HAT-based SR, DAT-based SR, and MambaIR-based SR
- Revision experiments for loss-confounding, parameter capacity, degradation robustness, noise robustness, standard PSNR, excluded-scene reporting, and clipped vegetation-index errors
- Figure generation for RMSE boxplots, RGB/false-color grids, and NDVI/GNDVI/NDRE error maps

## Data

The study uses the AI Hub dataset:

**Field Crop (Cabbage, etc.) Growth Condition Data**  
Dataset code: **71484**  
URL: https://aihub.or.kr/aihubdata/data/view.do?dataSetSn=71484

The AI Hub dataset is **not redistributed** in this repository. Users must download the data directly from AI Hub and follow the AI Hub data-use policy. Preprocessed cubes, model checkpoints, and generated outputs are also excluded because they are large derived artifacts.

## Expected Preprocessed Dataset

`Final_version6_mambair.py` expects a preprocessed dataset directory like this:

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

The pipeline reads the index from:

```text
DATASET_DIR/index/scene_index.csv
```

The index used in the manuscript is included in this repository as `scene_index.csv` (17,204 records). Copy it to `DATASET_DIR/index/` after building the cubes. It lists, for every scene, the AI Hub source file names, the acquisition date, region and field codes, and the paths of the HR and LR cubes relative to the dataset root; the AI Hub imagery itself is not redistributed.

**Row order matters.** The split is not stored in the file: `split_records()` reshuffles the records with `RANDOM_SEED = 42` at run time and takes a 70/15/15 split. The reported 10,124 / 2,169 / 2,170 split is therefore only reproduced if the rows are read in the order given here. `scene_index_template.csv` shows the same schema in minimal form.

## Repository Structure

```text
.
|-- Final_version6_mambair.py      main training / evaluation pipeline
|-- selftest_mambair.py           pre-flight check for the MambaIR baseline
|-- check_env.py                  library and GPU check
|-- build_uav_5band_dataset.py    builds the 5-band HR/LR cubes from the AI Hub TIFFs
|-- Make_NDVI_maps.py             single-scene error maps
|-- make_rgb_grids.py             RGB / false-colour / zoom comparison grids
|-- make_fig2_boxplot.py          per-scene RMSE boxplot
|-- scene_index.csv               scene index used in the manuscript (17,204 records)
|-- scene_index_template.csv      scene-index schema
|-- requirements.txt
|-- CITATION.cff
|-- LICENSE
|-- README.md
|-- README_REVISION.md
|-- .gitignore
```

## Main Scripts

| File | Purpose |
|---|---|
| `Final_version6_mambair.py` | Main training/evaluation pipeline for all nine methods and every revision experiment. Contains the data-integrity filters, patch-quality thresholds, valid-pixel masking, all metric implementations, and the fixed random seed. |
| `selftest_mambair.py` | Pre-flight check for the MambaIR baseline: exactness of the chunked selective scan against a sequential reference, parameter counts, forward/backward finiteness, peak memory and step time |
| `check_env.py` | Verifies the required libraries and reports CUDA / GPU availability |
| `Make_NDVI_maps.py` | Generates the single-scene per-band and vegetation-index error-map outputs |
| `make_rgb_grids.py` | Generates the RGB, false-color, and zoom comparison grids |
| `make_fig2_boxplot.py` | Generates the x4 band-wise per-scene RMSE boxplot from saved metric CSV files |
| `build_uav_5band_dataset.py` | Builds the five-band HR/LR `.npy` cubes and the scene index from the raw AI Hub TIFF and JSON files |
| `scene_index.csv` | The scene index used in the manuscript: 17,204 records with their AI Hub source file names and cube paths |
| `scene_index_template.csv` | Minimal example of the same schema |
| `CITATION.cff` | Citation metadata |

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

No source edit is required: every path and preset is read from the environment.

```bash
export SR_DATASET_DIR=/path/to/light_cabbage_training_dataset
export SR_OUTPUT_ROOT=/path/to/results
export SR_RUN_NAME=my_run
```

The figure scripts still carry path settings near the top of each file, which should be edited before use:

```python
# Make_NDVI_maps.py and make_rgb_grids.py
DATASET_DIR = Path("/path/to/light_cabbage_training_dataset")
CHECKPOINT_DIR = Path("/path/to/checkpoints")
OUTPUT_DIR = Path("/path/to/single_scene_outputs")

# make_fig2_boxplot.py
METRICS_DIR = "/path/to/results/<run_name>/sr_logs"
```

## Running the Main Benchmark

```bash
SR_PRESET=revision_experiments SR_SCALES=2,3,4 python -u Final_version6_mambair.py
```

Without `SR_PRESET` and `SR_SCALES` the file falls back to `full_eval` at x4 only, which is a narrower evaluation run than the manuscript uses.

## Revision Experiment Presets

| Experiment | Setting |
|---|---|
| Main comparison, all nine methods, plus EDSR-wide, the loss-confounding control, the noise evaluation and the D1/D2/D3 outputs | `SR_PRESET=revision_experiments SR_SCALES=2,3,4` |
| Gaussian-blur plus bicubic degradation robustness | `SR_DEGRADATION=blur_bicubic SR_PRESET=degradation_b2 SR_SCALES=4` |
| HAT/DAT loss-term ablation at x4 | `SR_PRESET=loss_ablation_hatdat SR_SCALES=4` |
| HAT-only noise robustness rows at x4 | `SR_PRESET=noise_hat SR_SCALES=4` |
| MambaIR baseline only, all scales and both modes | `SR_PRESET=mambair_only SR_SCALES=2,3,4` |
| MambaIR loss-term ablation at x4 | `SR_PRESET=loss_ablation_mambair SR_SCALES=4` |
| MambaIR noise robustness at x4 | `SR_PRESET=noise_mambair SR_SCALES=4 SR_NOISE_METHODS=mambair` |

Implemented revision components:

- `hat`, `dat` and `mambair`: recent SR baselines added during revision (HAT CVPR 2023, DAT ICCV 2023, MambaIR ECCV 2024)
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
| `mambair` | MambaIR-based state-space SR baseline added in the revision |
| `edsr_wide` | Parameter-matched widened joint EDSR for the capacity check (joint mode, x4 only) |

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
python Make_NDVI_maps.py
python make_rgb_grids.py
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

## Reproducing the Reported Numbers

Every path and preset can be set from the shell, so no source edit is required:

```bash
export SR_DATASET_DIR=/path/to/light_cabbage_training_dataset
export SR_OUTPUT_ROOT=/path/to/results
export SR_RUN_NAME=my_run

# main comparison, all nine methods
SR_PRESET=revision_experiments SR_SCALES=2,3,4 python -u Final_version6_mambair.py

# a single method (example: the MambaIR baseline)
SR_PRESET=mambair_only SR_SCALES=2,3,4 python -u Final_version6_mambair.py

# loss-term ablation (Table 7)
SR_PRESET=loss_ablation_hatdat  SR_SCALES=4 python -u Final_version6_mambair.py
SR_PRESET=loss_ablation_mambair SR_SCALES=4 python -u Final_version6_mambair.py

# alternative degradation (Table 8)
SR_DEGRADATION=blur_bicubic SR_PRESET=revision_experiments SR_SCALES=4 python -u Final_version6_mambair.py

# noise robustness (Table 9), evaluation only - reuses the x4 checkpoints
SR_PRESET=noise_hat      SR_SCALES=4 python -u Final_version6_mambair.py
SR_PRESET=noise_mambair  SR_SCALES=4 SR_NOISE_METHODS=mambair python -u Final_version6_mambair.py
```

Recognised environment variables: `SR_DATASET_DIR`, `SR_OUTPUT_ROOT`, `SR_RUN_NAME`, `SR_PRESET`,
`SR_SCALES`, `SR_DEGRADATION`, `SR_NOISE_METHODS`, and the MambaIR shape knobs
`SR_MAMBA_DIM / _GROUPS / _DEPTH / _D_STATE / _EXPAND`.

## Where the Reviewer-Requested Items Live in the Code

| Item | Location in `Final_version6_mambair.py` |
|---|---|
| Data-integrity filters (file existence, shape, NaN/Inf) | `read_scene_index()`, `is_clean_npy()`, `CHECK_SCENE_QUALITY_ALL_SCALES`, `SCENE_QUALITY_SCALES` |
| Patch-quality thresholds (valid ratio, std, dynamic range, LR-up-to-HR RMSE) | `MIN_PATCH_VALID_RATIO`, `MIN_PATCH_STD`, `MIN_PATCH_DYNAMIC_RANGE`, `MAX_PATCH_LRUP_HR_RMSE` |
| Valid-pixel masking | `USE_MASKED_LOSS`, `masked_l1_loss` and the masked metric helpers |
| Metric implementations (RMSE, PSNR, standard PSNR, SSIM, SAM, VI errors, band-wise) | metrics section; `REPORT_VI_CLIP01` adds the clipped VI columns |
| Fixed random seed | `RANDOM_SEED = 42` |
| Excluded-scene report (RMSE >= 1) | `excluded_scenes_report.csv` written to `sr_logs/` |
| Model definitions for all nine methods | `build_model()` |
