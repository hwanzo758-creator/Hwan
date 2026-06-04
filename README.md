# UAV Multispectral Super-Resolution

This repository provides a PyTorch-based pipeline for constructing 5-band multispectral UAV image cubes and training super-resolution (SR) models for agricultural remote sensing imagery.

The workflow consists of two main steps:

1. **Dataset construction**
   Stack raw Blue, Green, Red, RedEdge, and NIR TIFF images into 5-band HR cubes and generate LR cubes for ×2, ×3, and ×4 SR experiments.

2. **Model training**
   Train multispectral super-resolution models using either joint 5-channel learning or band-wise learning.

---

## Overview

This project was designed for UAV-based agricultural multispectral imagery.

The input data consist of five spectral bands:

```text
Blue, Green, Red, RedEdge, NIR
```

The generated cube format is:

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

---

## Repository Structure

```text
.
├── README.md
├── requirements.txt
├── .gitignore
├── build_light_cabbage_gangwon_dataset.py
└── train_sr_all_crops_Joint_5ch_x234_SR.py
```

---

## Main Scripts

| File                                      | Description                                                                    |
| ----------------------------------------- | ------------------------------------------------------------------------------ |
| `build_light_cabbage_gangwon_dataset.py`  | Builds the 5-band HR/LR cube dataset from raw TIFF files and JSON annotations. |
| `train_sr_all_crops_Joint_5ch_x234_SR.py` | Trains multispectral SR models using the generated HR/LR cube dataset.         |

---

## Dataset

This project uses the AIHub open dataset:

**Dataset name:** 노지작물(배추 등) 작황 데이터
**Provider:** AIHub / NIA
**Dataset URL:** https://aihub.or.kr/aihubdata/data/view.do?dataSetSn=71484

The original dataset is not included in this repository because of file size and dataset license restrictions.
Users should download the dataset directly from AIHub and follow the AIHub data usage policy.

After downloading, the expected source data structure is:

```text
107.노지작물(배추 등) 작황 데이터/
└── 01-1.정식개방데이터/
    └── Training/
        ├── 01.원천데이터/
        │   └── TS_1.배추_1.강원도/
        └── 02.라벨링데이터/
            └── TL_1.배추_1.강원도/
```

---

## Installation

Create a Python environment and install the required packages:

```bash
pip install -r requirements.txt
```

Recommended environment:

```text
Python >= 3.10
PyTorch >= 2.0
CUDA-enabled GPU recommended
```

---

## Requirements

The main dependencies are:

```text
numpy
pandas
opencv-python
torch
torchvision
matplotlib
openpyxl
scikit-image
tqdm
Pillow
tifffile
```

---

## Step 1. Build the 5-Band Cube Dataset

Before training SR models, generate HR and LR cube datasets from the raw TIFF files.

Open `build_light_cabbage_gangwon_dataset.py` and modify the following paths:

```python
RAW_ROOT = Path("path/to/Training/01.원천데이터/TS_1.배추_1.강원도")
LABEL_ROOT = Path("path/to/Training/02.라벨링데이터/TL_1.배추_1.강원도")
OUT_ROOT = Path("./light_cabbage_gangwon_dataset")
```

Then run:

```bash
python build_light_cabbage_gangwon_dataset.py
```

---

## Dataset Builder Output

The builder generates the following dataset structure:

```text
light_cabbage_gangwon_dataset/
├── index/
│   └── scene_index.csv
├── HR_npy/
├── LR_x2_npy/
├── LR_x3_npy/
├── LR_x4_npy/
├── masks/
├── rgb_preview/
├── false_color/
├── overlay/
├── metadata/
└── logs/
```

### Generated Files

| Folder                  | Description                                                 |
| ----------------------- | ----------------------------------------------------------- |
| `HR_npy/`               | Original 5-band HR cubes.                                   |
| `LR_x2_npy/`            | ×2 low-resolution cubes.                                    |
| `LR_x3_npy/`            | ×3 low-resolution cubes.                                    |
| `LR_x4_npy/`            | ×4 low-resolution cubes.                                    |
| `masks/`                | Binary masks generated from JSON polygon annotations.       |
| `rgb_preview/`          | RGB preview images for visualization.                       |
| `false_color/`          | False-color preview images using NIR, Red, and Green bands. |
| `overlay/`              | Annotation overlay preview images.                          |
| `metadata/`             | Scene-level metadata files.                                 |
| `index/scene_index.csv` | Main index file used by the training script.                |

---

## Step 2. Train Super-Resolution Models

After building the dataset, open `train_sr_all_crops_Joint_5ch_x234_SR.py` and modify the dataset/output paths:

```python
DATASET_DIR = Path("./light_cabbage_gangwon_dataset")
OUTPUT_ROOT = Path("./outputs")
```

Choose the SR scale:

```python
USER_SCALES = [2]        # Train x2 only
USER_SCALES = [3]        # Train x3 only
USER_SCALES = [4]        # Train x4 only
USER_SCALES = [2, 3, 4]  # Train all scales
```

Run training:

```bash
python train_sr_all_crops_Joint_5ch_x234_SR.py
```

---

## Supported SR Models

The training script supports the following SR methods:

| Method      | Description                             |
| ----------- | --------------------------------------- |
| Bicubic     | Non-learning interpolation baseline.    |
| SRCNN       | CNN-based baseline SR model.            |
| EDSR        | Enhanced Deep Super-Resolution model.   |
| RCAN        | Residual Channel Attention Network.     |
| SwinIR-like | Lightweight Transformer-based SR model. |
| ESRGAN-like | GAN-based SR model.                     |

---

## Training Modes

Two training modes are supported:

| Mode       | Description                                              |
| ---------- | -------------------------------------------------------- |
| `joint5ch` | Trains one model using all five spectral bands together. |
| `bandwise` | Trains separate models for each spectral band.           |

---

## Output Structure

Training results are saved under:

```text
OUTPUT_ROOT/RUN_NAME/
├── sr_checkpoints/
└── sr_logs/
    ├── train_history_*.csv
    ├── train_history_all.csv
    ├── train_history_all.xlsx
    ├── run_config*.json
    └── train_loss_plots/
```

### Main Outputs

| Output                   | Description                              |
| ------------------------ | ---------------------------------------- |
| `sr_checkpoints/`        | Trained model checkpoints.               |
| `train_history_*.csv`    | Per-model training history.              |
| `train_history_all.csv`  | Merged training history.                 |
| `train_history_all.xlsx` | Excel summary of all training histories. |
| `train_loss_plots/`      | Training loss curve images.              |
| `run_config*.json`       | Run configuration log.                   |

---

## Notes on Large Files

The following files and folders are intentionally excluded from this repository:

```text
*.npy
*.tif
*.tiff
*.raw
*.hdr
*.bsq
*.bil
*.bip
*.pth
*.pt
sr_checkpoints/
sr_logs/
outputs/
results/
```

Please use `.gitignore` to prevent accidental upload of large datasets, generated cubes, checkpoints, and result files.

---

## Reproducibility Notes

* The dataset builder crops all bands to a common minimum size before stacking.
* The band order is fixed as `Blue, Green, Red, RedEdge, NIR`.
* LR cubes are generated from HR cubes using OpenCV area interpolation.
* The training script supports ×2, ×3, and ×4 SR.
* For ×3 SR, HR and LR grids may not be perfectly divisible, so HR images are center-cropped to match the natural `LR × scale` size during training.
* Preview images are percentile-stretched only for visualization. The `.npy` cube values are not normalized by the preview process.

---

## Citation / Acknowledgement

This project uses the AIHub dataset **노지작물(배추 등) 작황 데이터**.

Users should cite or acknowledge AIHub according to the dataset usage policy.

If you use this code in your research, please cite the corresponding paper or repository when available.

---

## License

This repository is intended for research and educational purposes.

The dataset license follows the AIHub data usage policy.
Please check the original AIHub dataset page before redistributing or using the data commercially.
