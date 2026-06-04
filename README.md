# 5-Band Multispectral UAV Image Super-Resolution

This repository provides a PyTorch-based training pipeline for 5-band multispectral UAV image super-resolution using Blue, Green, Red, RedEdge, and NIR bands.

## Features

- Supports x2, x3, and x4 super-resolution
- Joint 5-channel SR training
- Band-wise SR ablation
- Models:
  - SRCNN
  - EDSR
  - RCAN
  - SwinIR-like
  - ESRGAN-like
- Residual SR learning
- Masked loss for nodata/background pixels
- Patch quality filtering
- Training history logging
- Training loss curve generation

## Dataset Structure

Expected dataset structure:

```text
light_cabbage_training_dataset/
├── index/
│   └── scene_index.csv
├── HR_npy/
├── LR_x2_npy/
├── LR_x3_npy/
└── LR_x4_npy/
