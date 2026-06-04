# train_sr_final_x234_joint5ch_bandwise_6models_v21_ndvi_10scene_evalonly.py
# V15 x2/x3/x4 update:
# - Scales expanded to x2, x3, and x4 while keeping quality filtering and error-map export.
#
# V14 error-map update:
# - Mean absolute error map, NDVI error map, NDRE error map, and panel image save option added.
# - CSV rows now include error-map paths.
#
# V13 quality filtering update:
# - Scene-level bad image filtering before train/val/test split.
# - Patch-level bad patch filtering during random LR-HR patch sampling.
# - Optional HR vs LR-up mismatch filtering to avoid misaligned or abnormal samples.
# - Bad scene list is saved to sr_logs/bad_scene_quality_filter.csv.
# - Patch quality settings are saved in run_config.json and run_time_summary.csv.
#
# V7 final update:
# - Input cubes are assumed to be already normalized reflectance-scale arrays.
# - Per-patch robust normalization is disabled.
# - -10000/nodata pixels are excluded by mask and replaced with 0.0 for model input.
# - Masked loss is enabled so nodata/background does not affect training.
# - Worker-local npy LRU cache reduces repeated disk I/O.
# - Validation batch_size uses BATCH_SIZE.
#
# ============================================================
# 5-band Multispectral Super-Resolution Pipeline
#
# Features:
# - NaN/Inf 포함 scene 자동 제외
# - 이미지 크기 기반 HR_PATCH_SIZE 자동 선택
# - Joint 5-channel SR
# - Band-wise SR ablation
# - 6 methods:
#   1. Bicubic
#   2. SRCNN
#   3. EDSR
#   4. RCAN
#   5. SwinIR-like
#   6. ESRGAN-like
#
# Added:
# - Epoch별 train / val / total time 측정
# - Scene별 load / inference / save / eval total time 측정
# - MPixels/sec, sec/MPixel 계산
# - model parameter 수 저장
# - 추가 성능지표:
#   MSE, R2, SCC, UQI, ERGAS, RASE
# - Band-wise:
#   RMSE, MAE, PSNR, SSIM
# - 모든 train_history_*.csv 파일을 train_history_all.csv/xlsx로 자동 통합 저장
#
# Input:
#   light_cabbage_gangwon_dataset/index/scene_index.csv
#   HR_npy/*.npy
#   LR_x2_npy/*.npy
#
# Output:
#   SR_x2_joint5ch_bicubic_npy/
#   SR_x2_joint5ch_srcnn_npy/
#   SR_x2_bandwise_srcnn_npy/
#   sr_logs/
#   sr_preview/
#   sr_diff_map/
#   sr_checkpoints/
# ============================================================

from pathlib import Path
from collections import OrderedDict
from torch.utils.data import Dataset, DataLoader

import csv
import json
import math
import random
import time
import re
from datetime import datetime

import cv2
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================
# 1. 사용자 설정
# ============================================================

# ============================================================
# Dataset directory setting
# ============================================================
# build_light_all_crops_training_dataset.py에서 생성한 결과 폴더를 지정합니다.
# 기본 빌드 설정:
#   SAVE_HR_NPY = True
#   SAVE_LR_NPY = True
#   SAVE_MASK_NPY = False
#   SAVE_OVERLAY = False
#   SAVE_METADATA = False
#
# 예상 구조:
#   light_all_crops_training_dataset/
#   ├── index/scene_index.csv
#   ├── HR_npy/
#   ├── LR_x2_npy/

DATASET_DIR = Path(
    "/home/whanjo/datasets/Python_file/SR_HR_LR_TEST/light_cabbage_training_dataset"
)

SCENE_INDEX_PATH = DATASET_DIR / "index" / "scene_index.csv"


# ============================================================
# Output directory setting
# ============================================================
# 입력 데이터(DATASET_DIR)는 그대로 두고, 결과 저장 위치만 별도로 분리합니다.
# 원하는 저장 위치로 OUTPUT_ROOT만 바꾸면 됩니다.
OUTPUT_ROOT = Path(
    "/home/whanjo/datasets/Python_file/SR_HR_LR_TEST/light_cabbage_training_dataset_SR_RESULTS"
)

# 실행별 결과를 분리하고 싶으면 RUN_NAME을 바꾸세요.
# 예: "all_crop_model_compare", "all_crop_x4_scale_compare", "test_run_001"
# NDVI 확인용 결과 폴더입니다.
# 기존 전체 결과/metrics를 덮어쓰지 않기 위해 별도 RUN_NAME을 사용합니다.
RUN_NAME = "final_train_only_loss_curves"

# 기존 훈련 checkpoint가 저장되어 있는 원본 RUN_NAME입니다.
# 모델은 여기서 불러오고, 새 NDVI/metric 결과는 위 RUN_NAME 폴더에 저장합니다.
CHECKPOINT_RUN_NAME = "final_x234_joint5ch_bandwise_6models_v19_resume_amp_earlystop_rank"

# 최종 결과 저장 폴더
OUTPUT_DIR = OUTPUT_ROOT / RUN_NAME

# checkpoint 로드 전용 폴더
CHECKPOINT_DIR = OUTPUT_ROOT / CHECKPOINT_RUN_NAME / "sr_checkpoints"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
RANDOM_SEED = 42
# ============================================================
# Fast startup scan setting
# ============================================================
# CSV가 있어도 초기 단계에서 HR/LR npy 전체 값을 훑으면 매우 오래 걸립니다.
# True이면 파일 존재, load 가능 여부, shape, channel 수만 확인합니다.
FAST_STARTUP_SCAN = True

# 전체 배추 데이터셋을 사용합니다.
# 일부만 테스트하고 싶으면 예: 2000으로 변경하세요.
MAX_RECORDS_FOR_TEST = None

# ============================================================
USE_QUICK_TEST_PRESET = False


# ============================================================
# Experiment preset switch
# ============================================================
# 비교 실험은 유지하되, 시간/용량을 줄이기 위한 preset 방식입니다.
#
# 1) "model_compare"  : 추천 기본값
#    - x2에서 joint5ch 기반 bicubic/SRCNN/EDSR/RCAN/SwinIR 비교
#    - 동시에 대표 모델에 대해 bandwise ablation 포함
#    - 비교성은 유지하면서 전체 시간을 줄이는 balanced setting
#
# 2) "scale_compare"
#    - x2/x3/x4 scale별 성능 비교
#    - method는 대표 모델만 사용
#
# 3) "bandwise_ablation"
#    - joint5ch vs bandwise 비교
#    - 매우 오래 걸리므로 대표 모델만 사용
#
# 4) "full_paper"
#    - 전체 조합. 시간이 매우 오래 걸리고 저장량도 큼
#
# 5) "full_eval"
#    - 전체 모델/모드 평가 preset
#    - 실제 scale은 USER_SCALES에서만 제어
#
EXPERIMENT_PRESET = "full_eval"

# ============================================================
# User scale selection
# ============================================================
# 원하는 scale만 선택해서 실행합니다.
# 예시:
#   USER_SCALES = [2]        # x2만
#   USER_SCALES = [3]        # x3만
#   USER_SCALES = [4]        # x4만
#   USER_SCALES = [2, 4]     # x2, x4만
#   USER_SCALES = [2, 3, 4]  # 전체
#
# 주의: [3] [2] [4]처럼 연속 리스트를 쓰는 것은 Python 문법상 의도와 다르게 동작할 수 있습니다.
# 반드시 아래처럼 하나의 리스트 안에 원하는 scale을 넣으세요.
USER_SCALES = [3]

# ============================================================
# Preset definitions
# ============================================================
# 중복 방지 원칙:
# - scale은 USER_SCALES에서만 결정합니다.
# - preset은 모델 목록, joint/bandwise 여부, epoch/save 옵션만 결정합니다.
FULL_METHODS = ["bicubic", "srcnn", "edsr", "rcan", "swinir", "esrgan"]
CORE_METHODS = ["bicubic", "edsr", "rcan", "swinir"]
REPRESENTATIVE_METHODS = ["edsr", "swinir"]

PRESETS = {
    # 논문용/최종 평가 기본 preset입니다.
    # 실제 scale은 USER_SCALES=[3], [2, 4], [2, 3, 4] 등으로 따로 선택합니다.
    "full_eval": {
        "FAST_MODE": False,
        "JOINT_METHODS_TO_RUN": FULL_METHODS,
        "BANDWISE_METHODS_TO_RUN": FULL_METHODS,
        "RUN_JOINT_5CH": True,
        "RUN_BANDWISE_ABLATION": True,
        "EPOCHS": 50,
        "PATCHES_PER_EPOCH": 1000,
        "SAVE_CHECKPOINT": True,
    },

    # 모델 비교 중심: joint5ch 위주, bandwise는 비활성화
    "model_compare": {
        "FAST_MODE": False,
        "JOINT_METHODS_TO_RUN": CORE_METHODS,
        "BANDWISE_METHODS_TO_RUN": REPRESENTATIVE_METHODS,
        "RUN_JOINT_5CH": True,
        "RUN_BANDWISE_ABLATION": False,
        "EPOCHS": 20,
        "PATCHES_PER_EPOCH": 500,
        "SAVE_CHECKPOINT": True,
    },

    # scale 비교 중심: 대표 모델만 실행
    "scale_compare": {
        "FAST_MODE": False,
        "JOINT_METHODS_TO_RUN": ["bicubic", "srcnn", "edsr", "swinir"],
        "BANDWISE_METHODS_TO_RUN": ["bicubic", "edsr"],
        "RUN_JOINT_5CH": True,
        "RUN_BANDWISE_ABLATION": False,
        "EPOCHS": 20,
        "PATCHES_PER_EPOCH": 400,
        "SAVE_CHECKPOINT": True,
    },

    # joint5ch vs bandwise 비교
    "bandwise_ablation": {
        "FAST_MODE": False,
        "JOINT_METHODS_TO_RUN": ["bicubic", "edsr", "swinir"],
        "BANDWISE_METHODS_TO_RUN": ["bicubic", "edsr", "swinir"],
        "RUN_JOINT_5CH": True,
        "RUN_BANDWISE_ABLATION": True,
        "EPOCHS": 20,
        "PATCHES_PER_EPOCH": 400,
        "SAVE_CHECKPOINT": True,
    },
}

# 기존 이름 호환용 alias입니다. 내부 설정은 full_eval과 동일합니다.
PRESET_ALIASES = {
    "final_full_x234": "full_eval",
    "full_paper": "full_eval",
}

preset_key = PRESET_ALIASES.get(EXPERIMENT_PRESET, EXPERIMENT_PRESET)
if preset_key not in PRESETS:
    raise ValueError(
        f"Unsupported EXPERIMENT_PRESET: {EXPERIMENT_PRESET}. "
        f"Valid presets: {sorted(PRESETS.keys()) + sorted(PRESET_ALIASES.keys())}"
    )

# 선택된 preset 값을 전역 설정으로 반영합니다.
for _key, _value in PRESETS[preset_key].items():
    globals()[_key] = list(_value) if isinstance(_value, list) else _value

# ============================================================
# Final scale selection from USER_SCALES
# ============================================================
# 모든 preset은 method/mode/epoch/save 옵션만 정하고,
# 실제 평가 scale은 USER_SCALES 한 곳에서만 결정합니다.
VALID_SCALES = [2, 3, 4]

if not isinstance(USER_SCALES, list) or len(USER_SCALES) == 0:
    raise ValueError(
        "USER_SCALES must be a non-empty list, e.g. [3], [2], [4], [2, 4], or [2, 3, 4]."
    )

invalid_scales = [s for s in USER_SCALES if s not in VALID_SCALES]
if len(invalid_scales) > 0:
    raise ValueError(f"Invalid scales in USER_SCALES: {invalid_scales}. Valid scales: {VALID_SCALES}")

# 중복 제거 + 입력 순서 유지
SCALES = list(dict.fromkeys(USER_SCALES))

print("\n====================================")
print("User-selected evaluation scales")
print("====================================")
print(f"EXPERIMENT_PRESET = {EXPERIMENT_PRESET}")
print(f"USER_SCALES       = {USER_SCALES}")
print(f"Final SCALES      = {SCALES}")
print("====================================")

# 전체 summary/폴더 생성을 위한 union list입니다.
# 실제 실행은 mode별 method list를 따로 사용합니다.
METHODS_TO_RUN = list(dict.fromkeys(JOINT_METHODS_TO_RUN + BANDWISE_METHODS_TO_RUN))

# Final version: SCALES is defined only from USER_SCALES above.
# Do not force x2/x3/x4 anywhere else.

EXPERIMENT_MODES = []
if RUN_JOINT_5CH:
    EXPERIMENT_MODES.append("joint5ch")
if RUN_BANDWISE_ABLATION:
    EXPERIMENT_MODES.append("bandwise")

TRAIN_MODE = {
    "srcnn": "scratch",
    "edsr": "scratch",
    "rcan": "scratch",
    "swinir": "scratch",
    "esrgan": "scratch",
}

# ============================================================
# V19 long-run stability / resume settings
# ============================================================
POSTPROCESS_ONLY = False

# checkpoint가 있으면 기존 모델을 사용하고, 없으면 해당 모델만 재훈련합니다.
# True이면 checkpoint 없는 모델은 skip, False이면 checkpoint 없는 모델은 재훈련.

SKIP_TRAIN_IF_CHECKPOINT_EXISTS = True
# NDVI 이미지를 새로 만들기 위해 기존 metrics가 있어도 평가를 다시 수행합니다.

EARLY_STOPPING = True
PATIENCE = 10
MIN_DELTA = 1e-5

USE_AMP = True
CHECK_SCENE_QUALITY_ALL_SCALES = True
ADD_RANK_COLUMNS = True

PRETRAINED_PATHS = {
    "srcnn": "",
    "edsr": "",
    "rcan": "",
    "swinir": "",
    "esrgan": "",
}

RESUME_CHECKPOINTS = {
    "srcnn": "",
    "edsr": "",
    "rcan": "",
    "swinir": "",
    "esrgan": "",
}

# ============================================================
# DataLoader / GPU utilization settings
# ============================================================
# RTX PRO 6000 98GB VRAM 환경 기준으로 GPU 활용률을 높이기 위한 기본값입니다.
# OOM이 발생하면 BATCH_SIZE를 8로 낮추세요.
BATCH_SIZE = 16
NUM_WORKERS = 8
PIN_MEMORY = True
PERSISTENT_WORKERS = True
PREFETCH_FACTOR = 4

# 이미 생성된 HR/LR npy cube가 반사율 스케일로 정리되어 있다는 가정하에
# robust_norm_factor/normalize_cube는 생략합니다.
# 단, -10000/nodata는 학습에서 제외하기 위해 mask로 유지합니다.
INPUT_ALREADY_NORMALIZED = True
USE_NPY_CACHE = True
NPY_CACHE_MAX_ITEMS = 16

LR = 1e-4
WEIGHT_DECAY = 1e-6

USE_DYNAMIC_PATCH_SIZE = True

PATCH_SIZE_CANDIDATES = [192, 144, 120, 96, 72, 48, 36, 24]
MIN_HR_PATCH_SIZE = 24


HR_PATCH_SIZE = 96
VAL_PATCHES_PER_SCENE = 5

# V11: SwinIR-like uses global MultiheadAttention over all pixels in each tile.
# Large inference tiles can make attention extremely slow because cost grows roughly with token_count^2.
# Therefore SwinIR evaluation uses a smaller tile than CNN-based models.

TRAIN_RATIO = 0.7
VAL_RATIO = 0.15
TEST_RATIO = 0.15

NORM_PERCENTILE = 99.5
NORM_EPS = 1e-6

# AIHub/GeoTIFF 계열에서 자주 쓰이는 background/nodata 값 처리
# 예: -10000은 실제 반사율이 아니라 invalid/background pixel임
NODATA_THRESHOLD = -9999.0
MIN_VALID_PIXEL_RATIO_SCENE = 0.05
MIN_VALID_PIXEL_RATIO_PATCH = 0.10
PATCH_RANDOM_TRIES = 30
USE_MASKED_LOSS = True

# ============================================================
# V13 Quality filtering settings
# ============================================================
# 이상한 scene/patch가 학습에 들어가지 않도록 하는 필터입니다.
# - scene-level: train/val/test split 전에 비정상 이미지 전체 제외
# - patch-level: 학습 중 비정상 patch 재샘플링
# - LRup-HR mismatch: LR을 bicubic으로 HR 크기까지 올렸을 때 HR과 지나치게 다르면 제외
# 평가 전용에서도 scene 품질 필터를 반드시 적용합니다.
# read_scene_index() 단계에서 bad scene은 records에 들어가지 않으므로
# joint5ch/bandwise 평가, preview, error map 생성에 모두 적용되지 않습니다.
ENABLE_SCENE_QUALITY_FILTER = True
ENABLE_PATCH_QUALITY_FILTER = True
ENABLE_PATCH_LRUP_HR_FILTER = True

# scene-level filter
SCENE_FILTER_SAMPLE_STRIDE = 16
# Evaluation clean-only setting
# 이상한 이미지/정합 불량 scene은 전체 평가에서도 제외합니다.
# - invalid/nodata가 많은 scene 제외
# - 값 변화가 거의 없는 scene 제외
# - LR을 bicubic으로 올렸을 때 HR과 차이가 큰 scene 제외
MAX_SCENE_INVALID_RATIO = 0.50
MIN_SCENE_VALID_RATIO = 1.0 - MAX_SCENE_INVALID_RATIO
MIN_SCENE_STD = 0.003
MIN_SCENE_DYNAMIC_RANGE = 0.015
MAX_SCENE_LRUP_HR_RMSE = 0.20

# patch-level filter
PATCH_QUALITY_RANDOM_TRIES = 60
MAX_PATCH_INVALID_RATIO = 0.50
MIN_PATCH_VALID_RATIO = 1.0 - MAX_PATCH_INVALID_RATIO
MIN_PATCH_STD = 0.003
MIN_PATCH_DYNAMIC_RANGE = 0.010
MAX_PATCH_LRUP_HR_RMSE = 0.20

# False이면 bad scene은 로그만 남기고 제외하지 않습니다.
DROP_BAD_SCENES = True


# ============================================================
# Sharpness / detail restoration settings
# ============================================================
# 선명도 개선을 위해 다음 두 가지를 추가합니다.
# 1) Residual SR: SR = bicubic(LR) + learned_detail
# 2) Sharpness loss: gradient + Laplacian edge loss
USE_RESIDUAL_SR = True
# SRCNN also uses the residual base path in forward(), so bicubic acts
# as the stable baseline and the network learns detail correction only.
USE_SHARPNESS_LOSS = True

# HR-SR 시각적 차이를 줄이기 위한 detail-focused loss 설정
# - L1/Charbonnier: 전체 반사율 값 정합
# - SSIM: 구조/밝기/대비 정합
# - Gradient/Laplacian/Frequency: 경계와 고주파 detail 보존
# - SAM/VI: 5-band 스펙트럼 및 식생지수 보존
L1_WEIGHT = 1.00
CHARBONNIER_WEIGHT = 0.50
SSIM_WEIGHT = 0.20
GRAD_WEIGHT = 0.15
EDGE_WEIGHT = 0.08
FREQ_WEIGHT = 0.03
SAM_WEIGHT = 0.10
VI_WEIGHT = 0.10

ESRGAN_PRETRAIN_EPOCHS = 20
ESRGAN_ADV_WEIGHT = 1e-3
ESRGAN_L1_WEIGHT = 1.0
ESRGAN_SAM_WEIGHT = 0.05
ESRGAN_VI_WEIGHT = 0.05

# 여기서 다시 덮어쓰면 model_compare의 5장이 30장으로 바뀌므로 유지해야 함.

BANDS = ["B", "G", "R", "E", "N"]
BAND_NAMES = ["Blue", "Green", "Red", "RedEdge", "NIR"]
CHANNELS = 5

# Preview / error-map / excel summary options
SAVE_BAND_ERROR_MAPS = False

# Preview 저장 방식
# True  : HR, LR-up, SR 이미지를 각각 따로 저장
# False : 기존처럼 HR|LR-up|SR 3장을 한 이미지로 합쳐 저장
SAVE_SEPARATE_PREVIEW = True

RUN_CONFIG_CACHE = {}


# ============================================================
# 2. 기본 유틸
# ============================================================

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def now_string():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def format_seconds(sec):
    sec = float(sec)
    h = int(sec // 3600)
    m = int((sec % 3600) // 60)
    s = sec % 60
    return f"{h:02d}:{m:02d}:{s:05.2f}"


def count_model_parameters(model):
    if model is None:
        return {
            "params_total": 0,
            "params_trainable": 0,
        }

    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)

    return {
        "params_total": int(total),
        "params_trainable": int(trainable),
    }


def ensure_dirs():
    (OUTPUT_DIR / "sr_logs").mkdir(parents=True, exist_ok=True)
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)


def write_dict_csv(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    if len(rows) == 0:
        print(f"[WARN] No rows to save: {path}")
        return

    fieldnames = list(rows[0].keys())

    with open(path, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for row in rows:
            writer.writerow(row)

    print(f"[INFO] Saved CSV: {path}")


def is_clean_npy(path):
    """
    빠른 npy 검사 함수.

    FAST_STARTUP_SCAN=True이면:
      - 파일 존재 여부
      - np.load 가능 여부
      - 3D cube 여부
      - channel 수 확인
    만 수행합니다.

    배추 데이터셋은 빌더 단계에서 NaN/Inf scene을 이미 제외했으므로
    학습 시작 전에 68,000개 이상의 npy 내부값을 다시 훑지 않습니다.
    """
    path = Path(path)

    if not path.exists():
        return False, "missing_file"

    try:
        arr = np.load(path, mmap_mode="r")
    except Exception as e:
        return False, f"load_error: {e}"

    if arr.ndim != 3:
        return False, f"invalid_ndim: {arr.shape}"

    if arr.shape[-1] != CHANNELS:
        return False, f"invalid_channels: {arr.shape}"

    if FAST_STARTUP_SCAN:
        return True, "clean_fast_shape_only"

    if np.isnan(arr).any():
        return False, "contains_nan"

    if np.isinf(arr).any():
        return False, "contains_inf"

    valid = arr > NODATA_THRESHOLD
    valid_ratio = float(valid.sum() / max(arr.size, 1))

    if valid_ratio < MIN_VALID_PIXEL_RATIO_SCENE:
        return False, f"too_few_valid_pixels: {valid_ratio:.4f}"

    return True, "clean"


def get_record_value(record, keys, default=""):
    """scene_index.csv 컬럼명이 달라도 호환되도록 여러 후보 key를 순서대로 탐색."""
    for key in keys:
        value = record.get(key, "")
        if value is not None and str(value).strip() != "":
            return str(value).strip()
    return default


def resolve_dataset_path(path_value):
    """
    CSV에 저장된 경로가 절대경로/상대경로/데이터셋 폴더 포함 상대경로 중
    어떤 형태여도 최대한 실제 파일 경로로 변환.
    """
    if path_value is None:
        return Path("")

    path_value = str(path_value).strip()
    if path_value == "":
        return Path("")

    p = Path(path_value)

    if p.is_absolute():
        return p

    candidates = [
        Path.cwd() / p,
        DATASET_DIR.parent / p,
        DATASET_DIR / p,
    ]

    # 예: light_all_crops_training_dataset/HR_npy/xxx.npy 형태인 경우
    parts = list(p.parts)
    if DATASET_DIR.name in parts:
        idx = parts.index(DATASET_DIR.name)
        rel_inside = Path(*parts[idx + 1:]) if idx + 1 < len(parts) else Path("")
        candidates.append(DATASET_DIR / rel_inside)

    for cand in candidates:
        if cand.exists():
            return cand

    # 파일이 아직 없더라도 가장 가능성 높은 경로 반환
    if DATASET_DIR.name in parts:
        idx = parts.index(DATASET_DIR.name)
        rel_inside = Path(*parts[idx + 1:]) if idx + 1 < len(parts) else Path("")
        return DATASET_DIR / rel_inside

    return DATASET_DIR / p


def read_scene_index():
    if not SCENE_INDEX_PATH.exists():
        raise FileNotFoundError(f"scene_index.csv not found: {SCENE_INDEX_PATH}")

    records = []

    with open(SCENE_INDEX_PATH, "r", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for row in reader:
            records.append(row)

    if MAX_RECORDS_FOR_TEST is not None:
        records = records[:MAX_RECORDS_FOR_TEST]
        print(f"[INFO] MAX_RECORDS_FOR_TEST enabled: {len(records)} records")

    valid_records = []
    skip_rows = []

    for r in records:
        scene_id = r.get("scene_id", "unknown")

        try:
            hr_path = get_hr_path(r)
        except Exception as e:
            skip_rows.append({
                "scene_id": scene_id,
                "data_type": "HR",
                "reason": f"missing_or_invalid_HR_path: {e}",
                "path": "",
            })
            continue

        ok, reason = is_clean_npy(hr_path)

        if not ok:
            skip_rows.append({
                "scene_id": scene_id,
                "data_type": "HR",
                "reason": reason,
                "path": str(hr_path),
            })
            continue

        has_problem = False

        for scale in SCALES:
            try:
                lr_path = get_lr_path(r, scale)
            except Exception as e:
                skip_rows.append({
                    "scene_id": scene_id,
                    "data_type": f"LR_x{scale}",
                    "reason": f"missing_or_invalid_LR_path: {e}",
                    "path": "",
                })
                has_problem = True
                break

            ok, reason = is_clean_npy(lr_path)

            if not ok:
                skip_rows.append({
                    "scene_id": scene_id,
                    "data_type": f"LR_x{scale}",
                    "reason": reason,
                    "path": str(lr_path),
                })
                has_problem = True
                break

        if has_problem:
            continue

        if ENABLE_SCENE_QUALITY_FILTER:
            scene_is_bad = False
            scene_bad_rows = []
            quality_scales = SCALES if CHECK_SCENE_QUALITY_ALL_SCALES else [SCALES[0]]

            for q_scale in quality_scales:
                try:
                    hr_q = np.load(hr_path, mmap_mode="r")
                    lr_q = np.load(get_lr_path(r, q_scale), mmap_mode="r")
                    is_bad, reason, qinfo = is_bad_scene_quality(hr_q, lr_q, scale=q_scale)
                except Exception as e:
                    is_bad = True
                    reason = f"scene_quality_check_error_x{q_scale}:{e}"
                    qinfo = {}

                if is_bad:
                    scene_is_bad = True
                    qrow = {
                        "scene_id": scene_id,
                        "data_type": "SCENE_QUALITY",
                        "scale": q_scale,
                        "reason": reason,
                        "path": str(hr_path),
                    }
                    qrow.update(qinfo)
                    scene_bad_rows.append(qrow)

            if scene_is_bad:
                skip_rows.extend(scene_bad_rows)
                if DROP_BAD_SCENES:
                    continue

        valid_records.append(r)

    print("\n====================================")
    print("Scene index filtering result")
    print("====================================")
    print(f"[INFO] Total records       : {len(records)}")
    print(f"[INFO] Valid clean records : {len(valid_records)}")
    print(f"[INFO] Skipped records     : {len(skip_rows)}")
    print("====================================")

    if len(skip_rows) > 0:
        skip_log_path = OUTPUT_DIR / "sr_logs" / "skipped_nan_inf_records.csv"
        write_dict_csv(skip_log_path, skip_rows)

        quality_rows = [row for row in skip_rows if row.get("data_type") == "SCENE_QUALITY"]
        if len(quality_rows) > 0:
            quality_log_path = OUTPUT_DIR / "sr_logs" / "bad_scene_quality_filter.csv"
            write_dict_csv(quality_log_path, quality_rows)

    if len(valid_records) == 0:
        raise RuntimeError(
            "No valid clean records found. Check scene_index.csv column names: "
            "HR_path/hr_npy and LR_x2_path/lr_x2_npy etc."
        )

    return valid_records

def split_records(records):
    records = records.copy()
    random.shuffle(records)

    n = len(records)
    n_train = int(n * TRAIN_RATIO)
    n_val = int(n * VAL_RATIO)

    train_records = records[:n_train]
    val_records = records[n_train:n_train + n_val]
    test_records = records[n_train + n_val:]

    if len(val_records) == 0:
        val_records = train_records[:max(1, min(5, len(train_records)))]

    if len(test_records) == 0:
        test_records = val_records

    print("\n====================================")
    print("Dataset split")
    print("====================================")
    print(f"Train: {len(train_records)}")
    print(f"Val  : {len(val_records)}")
    print(f"Test : {len(test_records)}")
    print("====================================")

    return train_records, val_records, test_records




def get_hr_path(record):
    # 기존 SR 코드 컬럼명과 새 데이터셋 빌더 컬럼명을 모두 지원
    p = get_record_value(record, [
        "HR_path", "hr_npy", "HR_npy", "hr_path",
    ])

    if p is None or str(p).strip() == "":
        raise FileNotFoundError(f"HR path is empty for scene {record.get('scene_id')}")

    p = resolve_dataset_path(p)

    if not p.exists():
        raise FileNotFoundError(f"HR file not found: {p}")

    return p


def get_lr_path(record, scale):
    # 기존 SR 코드 컬럼명과 새 데이터셋 빌더 컬럼명을 모두 지원
    p = get_record_value(record, [
        f"LR_x{scale}_path",
        f"lr_x{scale}_npy",
        f"LR_x{scale}_npy",
        f"lr_x{scale}_path",
    ])

    if p is None or str(p).strip() == "":
        raise FileNotFoundError(f"LR x{scale} path is empty for scene {record.get('scene_id')}")

    p = resolve_dataset_path(p)

    if not p.exists():
        raise FileNotFoundError(f"LR file not found: {p}")

    return p

def load_npy(path):
    arr = np.load(path).astype(np.float32)

    if arr.ndim != 3 or arr.shape[-1] != CHANNELS:
        raise ValueError(f"Invalid cube shape: {path}, shape={arr.shape}")

    if np.isnan(arr).any() or np.isinf(arr).any():
        raise ValueError(f"NaN/Inf detected in clean dataset: {path}")

    return arr


def make_valid_mask_np(cube):
    """True = 실제 반사율 유효 영역, False = nodata/background."""
    return np.isfinite(cube) & (cube > NODATA_THRESHOLD)


def clean_nodata_to_zero(cube, valid_mask=None):
    cube = cube.astype(np.float32).copy()

    if valid_mask is None:
        valid_mask = make_valid_mask_np(cube)

    cube[~valid_mask] = 0.0

    return cube.astype(np.float32)



def center_crop_to_shape(cube, target_h, target_w):
    """
    Center-crop a cube to the requested target size.

    Purpose:
    - For x3 SR with 800x800 HR and 266x266 LR, the natural SR size is 798x798.
    - Cropping HR to LR*scale prevents forced 798->800 resizing and preserves
      pixel-wise HR/SR alignment for quantitative evaluation.
    """
    cube = np.asarray(cube)
    h, w = cube.shape[:2]
    target_h = int(target_h)
    target_w = int(target_w)

    if target_h <= 0 or target_w <= 0:
        raise ValueError(f"Invalid target crop size: ({target_h}, {target_w})")

    if target_h > h or target_w > w:
        raise ValueError(
            f"Target crop size is larger than input: "
            f"input={cube.shape}, target=({target_h}, {target_w})"
        )

    y1 = (h - target_h) // 2
    x1 = (w - target_w) // 2
    y2 = y1 + target_h
    x2 = x1 + target_w

    return cube[y1:y2, x1:x2, :]


def align_hr_to_lr_scale(hr, lr, scale):
    """
    Align HR to the natural SR size determined by LR size × scale.

    Example:
      HR=(800, 800, 5), LR_x3=(266, 266, 5), scale=3
      -> aligned HR=(798, 798, 5)

    For x2/x4 with 800x800 HR, no crop occurs:
      LR_x2=(400, 400) -> target=(800, 800)
      LR_x4=(200, 200) -> target=(800, 800)
    """
    hr = np.asarray(hr)
    lr = np.asarray(lr)

    if hr.ndim != 3 or lr.ndim != 3:
        raise ValueError(f"HR/LR must be 3D cubes. HR={hr.shape}, LR={lr.shape}")

    target_h = int(lr.shape[0] * scale)
    target_w = int(lr.shape[1] * scale)

    if hr.shape[0] == target_h and hr.shape[1] == target_w:
        return hr

    if hr.shape[0] >= target_h and hr.shape[1] >= target_w:
        return center_crop_to_shape(hr, target_h, target_w)

    # Fallback: if HR is unexpectedly smaller, use the largest common centered area.
    common_h = min(hr.shape[0], target_h)
    common_w = min(hr.shape[1], target_w)
    return center_crop_to_shape(hr, common_h, common_w)


def align_hr_mask_to_lr_scale(hr, hr_mask, lr, scale):
    """
    Align HR cube and its valid mask using exactly the same crop.
    Used by SRPatchDataset so x3 training/validation patches are geometrically
    consistent with LR_x3.
    """
    target_h = int(lr.shape[0] * scale)
    target_w = int(lr.shape[1] * scale)

    if hr.shape[0] == target_h and hr.shape[1] == target_w:
        return hr, hr_mask

    if hr.shape[0] < target_h or hr.shape[1] < target_w:
        target_h = min(hr.shape[0], target_h)
        target_w = min(hr.shape[1], target_w)

    h, w = hr.shape[:2]
    y1 = (h - target_h) // 2
    x1 = (w - target_w) // 2
    y2 = y1 + target_h
    x2 = x1 + target_w

    return hr[y1:y2, x1:x2, :], hr_mask[y1:y2, x1:x2, :]


def mask_nodata_for_metrics(hr, sr):
    """Metric 계산 전 HR nodata 위치를 NaN으로 바꿔 모든 지표에서 제외."""
    hr, sr = crop_common_np(hr, sr)
    valid = make_valid_mask_np(hr) & np.isfinite(sr)

    hr_m = hr.astype(np.float32).copy()
    sr_m = sr.astype(np.float32).copy()

    hr_m[~valid] = np.nan
    sr_m[~valid] = np.nan

    valid_ratio = float(valid.sum() / max(valid.size, 1))

    return hr_m, sr_m, valid, valid_ratio



def safe_sample_cube_for_quality(cube, stride=16):
    """Scene-level 품질 검사용으로 cube를 stride 간격으로 샘플링."""
    cube = np.asarray(cube)
    stride = max(1, int(stride))
    if cube.ndim != 3:
        return cube
    return cube[::stride, ::stride, :]


def get_valid_values_for_quality(cube):
    """품질 필터 계산용 유효 pixel 값 추출."""
    cube = np.asarray(cube, dtype=np.float32)
    valid = np.isfinite(cube) & (cube > NODATA_THRESHOLD)
    if valid.sum() == 0:
        return None, valid
    return cube[valid], valid


def calc_lrup_hr_rmse_np(hr_patch, lr_patch):
    """LR patch를 HR patch 크기로 bicubic upsample한 뒤 HR과 RMSE 계산."""
    hr_patch = np.asarray(hr_patch, dtype=np.float32)
    lr_patch = np.asarray(lr_patch, dtype=np.float32)

    if hr_patch.ndim != 3 or lr_patch.ndim != 3:
        return np.nan

    if hr_patch.shape[-1] != lr_patch.shape[-1]:
        c = min(hr_patch.shape[-1], lr_patch.shape[-1])
        hr_patch = hr_patch[..., :c]
        lr_patch = lr_patch[..., :c]

    lr_up = cv2.resize(
        lr_patch,
        (hr_patch.shape[1], hr_patch.shape[0]),
        interpolation=cv2.INTER_CUBIC
    )

    if lr_up.ndim == 2:
        lr_up = lr_up[..., None]

    hr_m, lr_m, valid, valid_ratio = mask_nodata_for_metrics(hr_patch, lr_up)

    if valid_ratio <= 0:
        return np.nan

    diff = hr_m - lr_m
    valid2 = np.isfinite(diff)

    if valid2.sum() == 0:
        return np.nan

    return float(np.sqrt(np.mean(diff[valid2] ** 2)))


def is_bad_scene_quality(hr, lr, scale):
    """
    Scene-level 이상 이미지 필터.
    너무 invalid가 많거나, 값 변화가 거의 없거나, HR-LRup mismatch가 큰 scene을 제외합니다.
    """
    if not ENABLE_SCENE_QUALITY_FILTER:
        return False, "disabled", {}

    info = {}

    # Align HR to LR*scale before quality checks.
    # This prevents x3 from comparing 800x800 HR with a 798x798 natural SR grid.
    try:
        hr = align_hr_to_lr_scale(hr, lr, scale)
    except Exception as e:
        return True, f"scene_alignment_error_x{scale}:{e}", info

    hr_s = safe_sample_cube_for_quality(hr, SCENE_FILTER_SAMPLE_STRIDE)
    lr_s = safe_sample_cube_for_quality(lr, max(1, SCENE_FILTER_SAMPLE_STRIDE // max(int(scale), 1)))

    hr_values, hr_valid = get_valid_values_for_quality(hr_s)
    lr_values, lr_valid = get_valid_values_for_quality(lr_s)

    hr_valid_ratio = float(hr_valid.mean()) if hr_valid.size > 0 else 0.0
    lr_valid_ratio = float(lr_valid.mean()) if lr_valid.size > 0 else 0.0

    info["hr_valid_ratio"] = hr_valid_ratio
    info["lr_valid_ratio"] = lr_valid_ratio

    if hr_values is None:
        return True, "empty_valid_hr", info

    if lr_values is None:
        return True, "empty_valid_lr", info

    if hr_valid_ratio < MIN_SCENE_VALID_RATIO:
        return True, f"low_hr_valid_ratio:{hr_valid_ratio:.4f}", info

    if lr_valid_ratio < MIN_SCENE_VALID_RATIO:
        return True, f"low_lr_valid_ratio:{lr_valid_ratio:.4f}", info

    hr_std = float(np.std(hr_values))
    lr_std = float(np.std(lr_values))
    hr_dyn = float(np.percentile(hr_values, 99) - np.percentile(hr_values, 1))
    lr_dyn = float(np.percentile(lr_values, 99) - np.percentile(lr_values, 1))

    info["hr_std"] = hr_std
    info["lr_std"] = lr_std
    info["hr_dynamic_range_p99_p1"] = hr_dyn
    info["lr_dynamic_range_p99_p1"] = lr_dyn

    if hr_std < MIN_SCENE_STD:
        return True, f"low_hr_std:{hr_std:.6f}", info

    if lr_std < MIN_SCENE_STD:
        return True, f"low_lr_std:{lr_std:.6f}", info

    if hr_dyn < MIN_SCENE_DYNAMIC_RANGE:
        return True, f"low_hr_dynamic_range:{hr_dyn:.6f}", info

    if lr_dyn < MIN_SCENE_DYNAMIC_RANGE:
        return True, f"low_lr_dynamic_range:{lr_dyn:.6f}", info

    # HR-LRup mismatch 확인. 큰 scene을 stride로 줄여서 계산하므로 비용이 낮습니다.
    try:
        lr_for_rmse = cv2.resize(
            lr_s.astype(np.float32),
            (hr_s.shape[1], hr_s.shape[0]),
            interpolation=cv2.INTER_CUBIC
        )
        if lr_for_rmse.ndim == 2:
            lr_for_rmse = lr_for_rmse[..., None]
        hr_m, lr_m, _, valid_ratio = mask_nodata_for_metrics(hr_s, lr_for_rmse)
        diff = hr_m - lr_m
        valid = np.isfinite(diff)
        rmse = float(np.sqrt(np.mean(diff[valid] ** 2))) if valid.sum() > 0 else np.nan
    except Exception:
        rmse = np.nan
        valid_ratio = 0.0

    info["scene_lrup_hr_rmse"] = rmse
    info["scene_lrup_hr_valid_ratio"] = float(valid_ratio)

    if np.isfinite(rmse) and rmse > MAX_SCENE_LRUP_HR_RMSE:
        return True, f"scene_lrup_hr_rmse_too_large:{rmse:.6f}", info

    return False, "ok", info


def is_bad_patch_quality(hr_patch, lr_patch, mask_patch=None, scale=2):
    """
    Patch-level 이상 patch 필터.
    bad=True이면 학습용 patch로 쓰지 않고 다른 위치를 다시 샘플링합니다.
    """
    if not ENABLE_PATCH_QUALITY_FILTER:
        return False, "disabled", {"score": 1.0}

    info = {}

    hr_patch = np.asarray(hr_patch, dtype=np.float32)
    lr_patch = np.asarray(lr_patch, dtype=np.float32)

    hr_values, hr_valid = get_valid_values_for_quality(hr_patch)
    lr_values, lr_valid = get_valid_values_for_quality(lr_patch)

    hr_valid_ratio = float(hr_valid.mean()) if hr_valid.size > 0 else 0.0
    lr_valid_ratio = float(lr_valid.mean()) if lr_valid.size > 0 else 0.0

    info["hr_valid_ratio"] = hr_valid_ratio
    info["lr_valid_ratio"] = lr_valid_ratio

    if mask_patch is not None:
        mask_ratio = float(np.mean(mask_patch > 0.5))
    else:
        mask_ratio = min(hr_valid_ratio, lr_valid_ratio)

    info["mask_valid_ratio"] = mask_ratio

    if hr_values is None:
        info["score"] = -999.0
        return True, "empty_valid_hr_patch", info

    if lr_values is None:
        info["score"] = -999.0
        return True, "empty_valid_lr_patch", info

    if hr_valid_ratio < MIN_PATCH_VALID_RATIO:
        info["score"] = hr_valid_ratio
        return True, f"low_hr_valid_ratio:{hr_valid_ratio:.4f}", info

    if lr_valid_ratio < MIN_PATCH_VALID_RATIO:
        info["score"] = lr_valid_ratio
        return True, f"low_lr_valid_ratio:{lr_valid_ratio:.4f}", info

    if mask_ratio < MIN_VALID_PIXEL_RATIO_PATCH:
        info["score"] = mask_ratio
        return True, f"low_mask_valid_ratio:{mask_ratio:.4f}", info

    hr_std = float(np.std(hr_values))
    lr_std = float(np.std(lr_values))
    hr_dyn = float(np.percentile(hr_values, 99) - np.percentile(hr_values, 1))
    lr_dyn = float(np.percentile(lr_values, 99) - np.percentile(lr_values, 1))

    info["hr_std"] = hr_std
    info["lr_std"] = lr_std
    info["hr_dynamic_range_p99_p1"] = hr_dyn
    info["lr_dynamic_range_p99_p1"] = lr_dyn

    if hr_std < MIN_PATCH_STD:
        info["score"] = hr_std
        return True, f"low_hr_patch_std:{hr_std:.6f}", info

    if lr_std < MIN_PATCH_STD:
        info["score"] = lr_std
        return True, f"low_lr_patch_std:{lr_std:.6f}", info

    if hr_dyn < MIN_PATCH_DYNAMIC_RANGE:
        info["score"] = hr_dyn
        return True, f"low_hr_patch_dynamic_range:{hr_dyn:.6f}", info

    if lr_dyn < MIN_PATCH_DYNAMIC_RANGE:
        info["score"] = lr_dyn
        return True, f"low_lr_patch_dynamic_range:{lr_dyn:.6f}", info

    rmse = np.nan
    if ENABLE_PATCH_LRUP_HR_FILTER:
        rmse = calc_lrup_hr_rmse_np(hr_patch, lr_patch)
        info["patch_lrup_hr_rmse"] = rmse
        if np.isfinite(rmse) and rmse > MAX_PATCH_LRUP_HR_RMSE:
            info["score"] = -float(rmse)
            return True, f"patch_lrup_hr_rmse_too_large:{rmse:.6f}", info

    # fallback 후보 선택용 점수. 높을수록 좋은 patch.
    score = (
        2.0 * min(hr_valid_ratio, lr_valid_ratio, mask_ratio) +
        0.5 * min(hr_std, lr_std) +
        0.5 * min(hr_dyn, lr_dyn)
    )
    if np.isfinite(rmse):
        score -= 0.5 * rmse

    info["score"] = float(score)

    return False, "ok", info


def robust_norm_factor(hr):
    if INPUT_ALREADY_NORMALIZED:
        return 1.0

    valid = make_valid_mask_np(hr)

    if valid.sum() == 0:
        return 1.0

    values = hr[valid]
    values = values[values > 0]

    if len(values) == 0:
        return 1.0

    v = np.percentile(values, NORM_PERCENTILE)

    if not np.isfinite(v) or v <= NORM_EPS:
        v = np.max(values)

    if not np.isfinite(v) or v <= NORM_EPS:
        v = 1.0

    return float(v)


def normalize_cube(cube, norm_factor, valid_mask=None):
    cube = cube.astype(np.float32)

    if INPUT_ALREADY_NORMALIZED:
        # 이미 정규화된 입력은 추가 정규화를 하지 않습니다.
        # 다만 -10000/nodata 및 NaN/Inf는 모델 입력에 들어가지 않도록 0으로 치환합니다.
        if valid_mask is None:
            valid_mask = make_valid_mask_np(cube)
        cube = clean_nodata_to_zero(cube, valid_mask)
        cube = np.nan_to_num(cube, nan=0.0, posinf=1.5, neginf=0.0)
        return cube.astype(np.float32)

    if valid_mask is None:
        valid_mask = make_valid_mask_np(cube)

    cube = clean_nodata_to_zero(cube, valid_mask)

    norm_factor = float(norm_factor)

    if not np.isfinite(norm_factor) or norm_factor <= NORM_EPS:
        norm_factor = 1.0

    cube = cube / max(norm_factor, NORM_EPS)
    cube = np.clip(cube, 0.0, 1.5)

    return cube.astype(np.float32)


def np_to_tensor(cube):
    return torch.from_numpy(cube.transpose(2, 0, 1)).float()


def make_dataloader(dataset, batch_size, shuffle, num_workers, pin_memory=True):
    """
    DataLoader 생성 헬퍼.
    - num_workers > 0일 때만 persistent_workers / prefetch_factor 적용
    - CPU patch loading 병목 완화 목적
    """
    kwargs = {
        "dataset": dataset,
        "batch_size": batch_size,
        "shuffle": shuffle,
        "num_workers": num_workers,
        "pin_memory": pin_memory,
    }

    if num_workers > 0:
        kwargs["persistent_workers"] = PERSISTENT_WORKERS
        kwargs["prefetch_factor"] = PREFETCH_FACTOR

    return DataLoader(**kwargs)


# ============================================================
# 3. Dynamic patch / tile size
# ============================================================

def get_dataset_shapes(records):
    hr_shapes = []
    lr_shapes_by_scale = {scale: [] for scale in SCALES}

    for r in records:
        try:
            hr = np.load(get_hr_path(r), mmap_mode="r")
            hr_shapes.append((hr.shape[0], hr.shape[1]))

            for scale in SCALES:
                lr = np.load(get_lr_path(r, scale), mmap_mode="r")
                lr_shapes_by_scale[scale].append((lr.shape[0], lr.shape[1]))

        except Exception as e:
            print(f"[WARN] Failed to read shape for scene={r.get('scene_id', 'unknown')}: {e}")

    return hr_shapes, lr_shapes_by_scale


def choose_dynamic_hr_patch_size(records):
    hr_shapes, _ = get_dataset_shapes(records)

    if len(hr_shapes) == 0:
        print(f"[WARN] Cannot determine HR shapes. Use default HR_PATCH_SIZE={HR_PATCH_SIZE}")
        return HR_PATCH_SIZE

    min_hr_h = min([s[0] for s in hr_shapes])
    min_hr_w = min([s[1] for s in hr_shapes])
    min_hr_side = min(min_hr_h, min_hr_w)

    print("\n====================================")
    print("Dynamic HR patch size selection")
    print("====================================")
    print(f"Min HR height : {min_hr_h}")
    print(f"Min HR width  : {min_hr_w}")
    print(f"Min HR side   : {min_hr_side}")

    for candidate in PATCH_SIZE_CANDIDATES:
        if candidate > min_hr_side:
            continue

        divisible = True

        for scale in SCALES:
            if candidate % scale != 0:
                divisible = False
                break

        if not divisible:
            continue

        if candidate < MIN_HR_PATCH_SIZE:
            continue

        print(f"[INFO] Selected HR_PATCH_SIZE = {candidate}")
        print("====================================")
        return int(candidate)

    lcm_scale = 1
    for scale in SCALES:
        lcm_scale = math.lcm(lcm_scale, scale)

    fallback = (min_hr_side // lcm_scale) * lcm_scale

    if fallback < MIN_HR_PATCH_SIZE:
        fallback = MIN_HR_PATCH_SIZE

    print(f"[WARN] No candidate matched. Use fallback HR_PATCH_SIZE = {fallback}")
    print("====================================")

    return int(fallback)


def update_dynamic_sizes(records):
    global HR_PATCH_SIZE

    if USE_DYNAMIC_PATCH_SIZE:
        HR_PATCH_SIZE = choose_dynamic_hr_patch_size(records)



    print("\n====================================")
    print("Final dynamic size settings")
    print("====================================")
    print(f"HR_PATCH_SIZE = {HR_PATCH_SIZE}")
    print("====================================")


# ============================================================
# 4. Pretrained / Resume
# ============================================================

def extract_state_dict(checkpoint):
    if isinstance(checkpoint, dict):
        candidate_keys = [
            "state_dict",
            "model_state_dict",
            "generator_state_dict",
            "params",
            "params_ema",
            "net",
        ]

        for key in candidate_keys:
            if key in checkpoint and isinstance(checkpoint[key], dict):
                return checkpoint[key]

    if isinstance(checkpoint, dict):
        return checkpoint

    raise ValueError("Unsupported checkpoint format.")


def clean_state_dict_keys(state_dict):
    new_state = {}

    for k, v in state_dict.items():
        nk = k

        prefixes = [
            "module.",
            "model.",
            "net_g.",
            "generator.",
        ]

        for p in prefixes:
            if nk.startswith(p):
                nk = nk[len(p):]

        new_state[nk] = v

    return new_state


def load_partial_pretrained(model, pretrained_path, strict=False):
    pretrained_path = str(pretrained_path)

    if pretrained_path is None or pretrained_path.strip() == "":
        print("[INFO] No pretrained path provided.")
        return model

    pretrained_path = Path(pretrained_path)

    if not pretrained_path.exists():
        print(f"[WARN] Pretrained file not found: {pretrained_path}")
        return model

    print(f"[INFO] Loading pretrained weights: {pretrained_path}")

    checkpoint = torch.load(pretrained_path, map_location="cpu")
    pretrained_state = extract_state_dict(checkpoint)
    pretrained_state = clean_state_dict_keys(pretrained_state)

    model_state = model.state_dict()

    matched = {}
    skipped = []

    for k, v in pretrained_state.items():
        if k in model_state and model_state[k].shape == v.shape:
            matched[k] = v
        else:
            skipped.append(k)

    model_state.update(matched)
    model.load_state_dict(model_state, strict=strict)

    print("[INFO] Partial pretrained loading result")
    print(f"  Loaded layers : {len(matched)}")
    print(f"  Skipped layers: {len(skipped)}")

    return model


def load_resume_checkpoint(model, checkpoint_path, method=None):
    checkpoint_path = str(checkpoint_path)

    if checkpoint_path is None or checkpoint_path.strip() == "":
        print("[INFO] No resume checkpoint provided.")
        return model, 0

    checkpoint_path = Path(checkpoint_path)

    if not checkpoint_path.exists():
        print(f"[WARN] Resume checkpoint not found: {checkpoint_path}")
        return model, 0

    print(f"[INFO] Resuming from checkpoint: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=DEVICE)

    if method == "esrgan":
        if "generator_state_dict" in checkpoint:
            model.load_state_dict(checkpoint["generator_state_dict"], strict=True)
        else:
            state = extract_state_dict(checkpoint)
            model.load_state_dict(state, strict=False)
    else:
        if "model_state_dict" in checkpoint:
            model.load_state_dict(checkpoint["model_state_dict"], strict=True)
        else:
            state = extract_state_dict(checkpoint)
            model.load_state_dict(state, strict=False)

    start_epoch = checkpoint.get("epoch", 0) if isinstance(checkpoint, dict) else 0

    print(f"[INFO] Resume start epoch: {start_epoch}")

    return model, start_epoch


# ============================================================
# 5. Dataset
# ============================================================

class SRPatchDataset(Dataset):
    def __init__(
        self,
        records,
        scale,
        channels=5,
        band_index=None,
        patches_per_epoch=1000,
        train=True,
    ):
        self.records = records
        self.scale = scale
        self.channels = channels
        self.band_index = band_index
        self.patches_per_epoch = patches_per_epoch
        self.train = train

        self.hr_patch_size = HR_PATCH_SIZE

        if self.hr_patch_size % scale != 0:
            raise ValueError(
                f"HR_PATCH_SIZE={self.hr_patch_size} must be divisible by scale={scale}"
            )

        self.lr_patch_size = self.hr_patch_size // scale
        self.cache = OrderedDict()

        if self.lr_patch_size < 8:
            raise ValueError("LR patch size is too small.")

    def load_cached_npy(self, path):
        path = str(path)

        if not USE_NPY_CACHE:
            return load_npy(path)

        if path in self.cache:
            arr = self.cache.pop(path)
            self.cache[path] = arr
            return arr

        arr = load_npy(path)
        self.cache[path] = arr

        while len(self.cache) > NPY_CACHE_MAX_ITEMS:
            self.cache.popitem(last=False)

        return arr

    def __len__(self):
        if self.train:
            return self.patches_per_epoch

        return max(self.patches_per_epoch, 1)

    def get_eval_patch_xy(self, idx, max_lr_y, max_lr_x):
        positions = [
            (0.50, 0.50),
            (0.25, 0.25),
            (0.25, 0.75),
            (0.75, 0.25),
            (0.75, 0.75),
        ]
        record_count = max(len(self.records), 1)
        slot = (idx // record_count) % max(VAL_PATCHES_PER_SCENE, 1)
        y_ratio, x_ratio = positions[slot % len(positions)]

        y_lr = int(round(max_lr_y * y_ratio))
        x_lr = int(round(max_lr_x * x_ratio))

        y_lr = min(max(y_lr, 0), max_lr_y)
        x_lr = min(max(x_lr, 0), max_lr_x)

        return y_lr, x_lr

    def select_channels(self, cube):
        if self.channels == 5:
            return cube

        if self.channels == 1:
            if self.band_index is None:
                raise ValueError("band_index must be provided when channels=1")

            return cube[..., self.band_index:self.band_index + 1]

        raise ValueError(f"Unsupported channels: {self.channels}")

    def __getitem__(self, idx):
        if self.train:
            record = random.choice(self.records)
        else:
            record = self.records[idx % len(self.records)]

        hr_raw = self.load_cached_npy(get_hr_path(record))
        lr_raw = self.load_cached_npy(get_lr_path(record, self.scale))

        if INPUT_ALREADY_NORMALIZED:
            # 이미 반사율 스케일로 정리된 npy라고 가정하므로 robust norm/normalize는 생략합니다.
            # 단, -10000/nodata는 모델 입력에서 0으로 치환하고 loss에서는 제외합니다.
            hr_valid_mask = make_valid_mask_np(hr_raw)
            lr_valid_mask = make_valid_mask_np(lr_raw)

            # Align HR and HR mask to LR*scale.
            # For x3, 800x800 HR with 266x266 LR becomes 798x798 HR.
            hr_raw_aligned, hr_valid_mask = align_hr_mask_to_lr_scale(
                hr_raw, hr_valid_mask.astype(np.float32), lr_raw, self.scale
            )
            hr_valid_mask = hr_valid_mask.astype(bool)

            hr = clean_nodata_to_zero(hr_raw_aligned, hr_valid_mask)
            lr = clean_nodata_to_zero(lr_raw, lr_valid_mask)

            hr = np.nan_to_num(hr.astype(np.float32), nan=0.0, posinf=1.5, neginf=0.0)
            lr = np.nan_to_num(lr.astype(np.float32), nan=0.0, posinf=1.5, neginf=0.0)

            hr_mask = hr_valid_mask.astype(np.float32)

            hr = self.select_channels(hr)
            lr = self.select_channels(lr)
            hr_mask = self.select_channels(hr_mask)
        else:
            hr_valid_mask = make_valid_mask_np(hr_raw)
            lr_valid_mask = make_valid_mask_np(lr_raw)

            # Align HR and HR mask to LR*scale for non-integer scale compatibility.
            hr_raw_aligned, hr_valid_mask = align_hr_mask_to_lr_scale(
                hr_raw, hr_valid_mask.astype(np.float32), lr_raw, self.scale
            )
            hr_valid_mask = hr_valid_mask.astype(bool)

            norm_factor = robust_norm_factor(hr_raw_aligned)

            hr = normalize_cube(hr_raw_aligned, norm_factor, hr_valid_mask)
            lr = normalize_cube(lr_raw, norm_factor, lr_valid_mask)

            hr_mask = hr_valid_mask.astype(np.float32)

            hr = self.select_channels(hr)
            lr = self.select_channels(lr)
            hr_mask = self.select_channels(hr_mask)

        h_lr, w_lr = lr.shape[:2]
        h_hr, w_hr = hr.shape[:2]

        max_lr_y = min(
            h_lr - self.lr_patch_size,
            h_hr // self.scale - self.lr_patch_size
        )
        max_lr_x = min(
            w_lr - self.lr_patch_size,
            w_hr // self.scale - self.lr_patch_size
        )

        if max_lr_y <= 0 or max_lr_x <= 0:
            lr_patch = cv2.resize(
                lr,
                (self.lr_patch_size, self.lr_patch_size),
                interpolation=cv2.INTER_CUBIC
            )

            hr_patch = cv2.resize(
                hr,
                (self.hr_patch_size, self.hr_patch_size),
                interpolation=cv2.INTER_CUBIC
            )

            if lr_patch.ndim == 2:
                lr_patch = lr_patch[..., None]

            if hr_patch.ndim == 2:
                hr_patch = hr_patch[..., None]

            hr_mask_patch = cv2.resize(
                hr_mask,
                (self.hr_patch_size, self.hr_patch_size),
                interpolation=cv2.INTER_NEAREST
            )

            if hr_mask_patch.ndim == 2:
                hr_mask_patch = hr_mask_patch[..., None]

            return np_to_tensor(lr_patch), np_to_tensor(hr_patch), np_to_tensor(hr_mask_patch)

        if self.train:
            # V13: nodata가 많거나, 값 변화가 거의 없거나, HR-LRup mismatch가 큰 patch는 훈련에서 제외
            y_lr = random.randint(0, max_lr_y)
            x_lr = random.randint(0, max_lr_x)
            best_score = -1e18
            best_y_lr = y_lr
            best_x_lr = x_lr

            tries = PATCH_QUALITY_RANDOM_TRIES if ENABLE_PATCH_QUALITY_FILTER else PATCH_RANDOM_TRIES

            for _ in range(tries):
                cand_y_lr = random.randint(0, max_lr_y)
                cand_x_lr = random.randint(0, max_lr_x)
                cand_y_hr = cand_y_lr * self.scale
                cand_x_hr = cand_x_lr * self.scale

                cand_lr_patch = lr[
                    cand_y_lr:cand_y_lr + self.lr_patch_size,
                    cand_x_lr:cand_x_lr + self.lr_patch_size,
                    :
                ]

                cand_hr_patch = hr[
                    cand_y_hr:cand_y_hr + self.hr_patch_size,
                    cand_x_hr:cand_x_hr + self.hr_patch_size,
                    :
                ]

                cand_mask = hr_mask[
                    cand_y_hr:cand_y_hr + self.hr_patch_size,
                    cand_x_hr:cand_x_hr + self.hr_patch_size,
                    :
                ]

                if cand_mask.size == 0:
                    continue

                bad_patch, patch_reason, patch_info = is_bad_patch_quality(
                    cand_hr_patch,
                    cand_lr_patch,
                    cand_mask,
                    scale=self.scale
                )

                score = float(patch_info.get("score", -1e18))

                if score > best_score:
                    best_score = score
                    best_y_lr = cand_y_lr
                    best_x_lr = cand_x_lr

                if not bad_patch:
                    y_lr = cand_y_lr
                    x_lr = cand_x_lr
                    break
            else:
                # 모든 후보가 bad이면 가장 덜 나쁜 patch 사용
                y_lr = best_y_lr
                x_lr = best_x_lr
        else:
            y_lr, x_lr = self.get_eval_patch_xy(idx, max_lr_y, max_lr_x)

        y_hr = y_lr * self.scale
        x_hr = x_lr * self.scale

        lr_patch = lr[
            y_lr:y_lr + self.lr_patch_size,
            x_lr:x_lr + self.lr_patch_size,
            :
        ]

        hr_patch = hr[
            y_hr:y_hr + self.hr_patch_size,
            x_hr:x_hr + self.hr_patch_size,
            :
        ]

        hr_mask_patch = hr_mask[
            y_hr:y_hr + self.hr_patch_size,
            x_hr:x_hr + self.hr_patch_size,
            :
        ]

        if lr_patch.shape[0] != self.lr_patch_size or lr_patch.shape[1] != self.lr_patch_size:
            lr_patch = cv2.resize(
                lr_patch,
                (self.lr_patch_size, self.lr_patch_size),
                interpolation=cv2.INTER_CUBIC
            )

        if hr_patch.shape[0] != self.hr_patch_size or hr_patch.shape[1] != self.hr_patch_size:
            hr_patch = cv2.resize(
                hr_patch,
                (self.hr_patch_size, self.hr_patch_size),
                interpolation=cv2.INTER_CUBIC
            )
            hr_mask_patch = cv2.resize(
                hr_mask_patch,
                (self.hr_patch_size, self.hr_patch_size),
                interpolation=cv2.INTER_NEAREST
            )

        if lr_patch.ndim == 2:
            lr_patch = lr_patch[..., None]

        if hr_patch.ndim == 2:
            hr_patch = hr_patch[..., None]

        if hr_mask_patch.ndim == 2:
            hr_mask_patch = hr_mask_patch[..., None]

        return np_to_tensor(lr_patch), np_to_tensor(hr_patch), np_to_tensor(hr_mask_patch)


# ============================================================
# 6. Models
# ============================================================

def make_activation(name):
    if name == "relu":
        return nn.ReLU(inplace=True)

    if name == "gelu":
        return nn.GELU()

    if name == "lrelu":
        return nn.LeakyReLU(0.2, inplace=True)

    return nn.ReLU(inplace=True)


def zero_init_conv(conv):
    nn.init.zeros_(conv.weight)
    if conv.bias is not None:
        nn.init.zeros_(conv.bias)


def make_upsampler(scale, nf, activation="relu"):
    layers = []

    if scale == 2:
        layers += [
            nn.Conv2d(nf, nf * 4, 3, padding=1),
            nn.PixelShuffle(2),
            make_activation(activation),
        ]

    elif scale == 3:
        layers += [
            nn.Conv2d(nf, nf * 9, 3, padding=1),
            nn.PixelShuffle(3),
            make_activation(activation),
        ]

    elif scale == 4:
        layers += [
            nn.Conv2d(nf, nf * 4, 3, padding=1),
            nn.PixelShuffle(2),
            make_activation(activation),
            nn.Conv2d(nf, nf * 4, 3, padding=1),
            nn.PixelShuffle(2),
            make_activation(activation),
        ]

    else:
        raise ValueError(f"Unsupported scale: {scale}")

    return nn.Sequential(*layers)


class SRCNN(nn.Module):
    def __init__(self, scale, channels=5):
        super().__init__()

        self.scale = scale

        # Keep the original SRCNN layer shapes so previously trained checkpoints
        # can still be loaded. The practical improvement is in forward():
        # use bicubic upsampling as a stable base image and let SRCNN predict
        # only the residual/detail component.
        self.net = nn.Sequential(
            nn.Conv2d(channels, 64, kernel_size=9, padding=4),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 32, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, channels, kernel_size=5, padding=2),
        )
        zero_init_conv(self.net[-1])

    def forward(self, x):
        base = F.interpolate(
            x,
            scale_factor=self.scale,
            mode="bicubic",
            align_corners=False
        )

        detail = self.net(base)

        # Improved SRCNN:
        #   SR = bicubic base + predicted detail
        # This generally stabilizes output and helps avoid overly dark/flat
        # predictions compared with directly predicting the full SR image.
        if USE_RESIDUAL_SR:
            out = base + detail
        else:
            out = detail

        return out


class ResidualBlock(nn.Module):
    def __init__(self, nf=64):
        super().__init__()

        self.conv1 = nn.Conv2d(nf, nf, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(nf, nf, kernel_size=3, padding=1)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        res = self.relu(self.conv1(x))
        res = self.conv2(res)

        return x + res * 0.1


class EDSR(nn.Module):
    def __init__(self, scale, channels=5, nf=64, n_blocks=8):
        super().__init__()

        self.scale = scale
        self.head = nn.Conv2d(channels, nf, kernel_size=3, padding=1)

        self.body = nn.Sequential(
            *[ResidualBlock(nf) for _ in range(n_blocks)]
        )

        self.body_conv = nn.Conv2d(nf, nf, kernel_size=3, padding=1)
        self.up = make_upsampler(scale, nf, activation="relu")
        self.tail = nn.Conv2d(nf, channels, kernel_size=3, padding=1)
        zero_init_conv(self.tail)

    def forward(self, x):
        # Residual SR: bicubic upsample을 base로 두고 모델은 detail만 학습
        base = F.interpolate(
            x,
            scale_factor=self.scale,
            mode="bicubic",
            align_corners=False
        )

        feat = self.head(x)

        res = self.body(feat)
        res = self.body_conv(res)

        feat = feat + res
        feat = self.up(feat)
        detail = self.tail(feat)

        if USE_RESIDUAL_SR:
            out = base + detail
        else:
            out = detail

        return out


class ChannelAttention(nn.Module):
    def __init__(self, nf=64, reduction=16):
        super().__init__()

        hidden = max(nf // reduction, 4)

        self.pool = nn.AdaptiveAvgPool2d(1)

        self.ca = nn.Sequential(
            nn.Conv2d(nf, hidden, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, nf, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        weight = self.ca(self.pool(x))
        return x * weight


class RCAB(nn.Module):
    def __init__(self, nf=64):
        super().__init__()

        self.conv1 = nn.Conv2d(nf, nf, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(nf, nf, kernel_size=3, padding=1)
        self.relu = nn.ReLU(inplace=True)
        self.ca = ChannelAttention(nf)

    def forward(self, x):
        res = self.relu(self.conv1(x))
        res = self.conv2(res)
        res = self.ca(res)

        return x + res * 0.1


class ResidualGroup(nn.Module):
    def __init__(self, nf=64, n_rcab=4):
        super().__init__()

        self.blocks = nn.Sequential(
            *[RCAB(nf) for _ in range(n_rcab)]
        )

        self.conv = nn.Conv2d(nf, nf, kernel_size=3, padding=1)

    def forward(self, x):
        res = self.blocks(x)
        res = self.conv(res)

        return x + res


class RCAN(nn.Module):
    def __init__(self, scale, channels=5, nf=64, n_groups=4, n_rcab=4):
        super().__init__()

        self.scale = scale
        self.head = nn.Conv2d(channels, nf, kernel_size=3, padding=1)

        self.body = nn.Sequential(
            *[ResidualGroup(nf, n_rcab) for _ in range(n_groups)]
        )

        self.body_conv = nn.Conv2d(nf, nf, kernel_size=3, padding=1)
        self.up = make_upsampler(scale, nf, activation="relu")
        self.tail = nn.Conv2d(nf, channels, kernel_size=3, padding=1)
        zero_init_conv(self.tail)

    def forward(self, x):
        base = F.interpolate(
            x,
            scale_factor=self.scale,
            mode="bicubic",
            align_corners=False
        )

        feat = self.head(x)

        res = self.body(feat)
        res = self.body_conv(res)

        feat = feat + res
        feat = self.up(feat)
        detail = self.tail(feat)

        if USE_RESIDUAL_SR:
            out = base + detail
        else:
            out = detail

        return out


class SwinLikeBlock(nn.Module):
    def __init__(self, dim=64, heads=4, mlp_ratio=2.0):
        super().__init__()

        self.norm1 = nn.LayerNorm(dim)

        self.attn = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=heads,
            batch_first=True
        )

        self.norm2 = nn.LayerNorm(dim)

        hidden = int(dim * mlp_ratio)

        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, dim)
        )

    def forward(self, x):
        b, c, h, w = x.shape

        tokens = x.flatten(2).transpose(1, 2)

        y = self.norm1(tokens)
        attn_out, _ = self.attn(y, y, y, need_weights=False)
        tokens = tokens + attn_out

        y = self.norm2(tokens)
        tokens = tokens + self.mlp(y)

        out = tokens.transpose(1, 2).reshape(b, c, h, w)

        return out


class SwinIRLike(nn.Module):
    def __init__(self, scale, channels=5, dim=64, depth=4, heads=4):
        super().__init__()

        self.scale = scale
        self.head = nn.Conv2d(channels, dim, kernel_size=3, padding=1)

        self.blocks = nn.Sequential(
            *[SwinLikeBlock(dim=dim, heads=heads) for _ in range(depth)]
        )

        self.body_conv = nn.Conv2d(dim, dim, kernel_size=3, padding=1)
        self.up = make_upsampler(scale, dim, activation="gelu")
        self.tail = nn.Conv2d(dim, channels, kernel_size=3, padding=1)
        zero_init_conv(self.tail)

    def forward(self, x):
        base = F.interpolate(
            x,
            scale_factor=self.scale,
            mode="bicubic",
            align_corners=False
        )

        feat = self.head(x)

        res = self.blocks(feat)
        res = self.body_conv(res)

        feat = feat + res
        feat = self.up(feat)
        detail = self.tail(feat)

        if USE_RESIDUAL_SR:
            out = base + detail
        else:
            out = detail

        return out


class DenseResidualBlock(nn.Module):
    def __init__(self, nf=64, gc=32):
        super().__init__()

        self.conv1 = nn.Conv2d(nf, gc, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(nf + gc, gc, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(nf + gc * 2, gc, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(nf + gc * 3, gc, kernel_size=3, padding=1)
        self.conv5 = nn.Conv2d(nf + gc * 4, nf, kernel_size=3, padding=1)

        self.lrelu = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x):
        x1 = self.lrelu(self.conv1(x))
        x2 = self.lrelu(self.conv2(torch.cat([x, x1], dim=1)))
        x3 = self.lrelu(self.conv3(torch.cat([x, x1, x2], dim=1)))
        x4 = self.lrelu(self.conv4(torch.cat([x, x1, x2, x3], dim=1)))
        x5 = self.conv5(torch.cat([x, x1, x2, x3, x4], dim=1))

        return x + x5 * 0.2


class RRDB(nn.Module):
    def __init__(self, nf=64, gc=32):
        super().__init__()

        self.rdb1 = DenseResidualBlock(nf, gc)
        self.rdb2 = DenseResidualBlock(nf, gc)
        self.rdb3 = DenseResidualBlock(nf, gc)

    def forward(self, x):
        res = self.rdb1(x)
        res = self.rdb2(res)
        res = self.rdb3(res)

        return x + res * 0.2


class ESRGANGenerator(nn.Module):
    def __init__(self, scale, channels=5, nf=64, n_rrdb=6):
        super().__init__()

        self.scale = scale
        self.conv_first = nn.Conv2d(channels, nf, kernel_size=3, padding=1)

        self.trunk = nn.Sequential(
            *[RRDB(nf) for _ in range(n_rrdb)]
        )

        self.trunk_conv = nn.Conv2d(nf, nf, kernel_size=3, padding=1)
        self.up = make_upsampler(scale, nf, activation="lrelu")

        self.conv_hr = nn.Conv2d(nf, nf, kernel_size=3, padding=1)
        self.conv_last = nn.Conv2d(nf, channels, kernel_size=3, padding=1)
        zero_init_conv(self.conv_last)

        self.lrelu = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x):
        base = F.interpolate(
            x,
            scale_factor=self.scale,
            mode="bicubic",
            align_corners=False
        )

        fea = self.conv_first(x)

        trunk = self.trunk(fea)
        trunk = self.trunk_conv(trunk)

        fea = fea + trunk

        out = self.up(fea)
        out = self.lrelu(self.conv_hr(out))
        detail = self.conv_last(out)

        if USE_RESIDUAL_SR:
            out = base + detail
        else:
            out = detail

        return out


class ESRGANDiscriminator(nn.Module):
    def __init__(self, channels=5, base=64):
        super().__init__()

        def block(in_ch, out_ch, stride):
            return nn.Sequential(
                nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=stride, padding=1),
                nn.BatchNorm2d(out_ch),
                nn.LeakyReLU(0.2, inplace=True)
            )

        self.features = nn.Sequential(
            nn.Conv2d(channels, base, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),

            block(base, base, 2),
            block(base, base * 2, 1),
            block(base * 2, base * 2, 2),
            block(base * 2, base * 4, 1),
            block(base * 4, base * 4, 2),
            block(base * 4, base * 8, 1),
            block(base * 8, base * 8, 2),
        )

        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(base * 8, 1)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)

        return x


def build_model(method, scale, channels=5):
    if method == "srcnn":
        return SRCNN(scale=scale, channels=channels)

    if method == "edsr":
        return EDSR(scale=scale, channels=channels, nf=96, n_blocks=16)

    if method == "rcan":
        return RCAN(scale=scale, channels=channels, nf=96, n_groups=6, n_rcab=6)

    if method == "swinir":
        return SwinIRLike(scale=scale, channels=channels, dim=96, depth=6, heads=4)

    if method == "esrgan":
        return ESRGANGenerator(scale=scale, channels=channels, nf=64, n_rrdb=6)

    raise ValueError(f"Unsupported method: {method}")


# ============================================================
# 7. Loss
# ============================================================

def masked_l1_loss(sr, hr, mask=None):
    if mask is None or not USE_MASKED_LOSS:
        return F.l1_loss(sr, hr)

    mask = mask.float()
    loss = torch.abs(sr - hr) * mask
    return loss.sum() / (mask.sum() + 1e-8)


def sam_loss(sr, hr, mask=None):
    sr_f = sr.permute(0, 2, 3, 1).reshape(-1, sr.shape[1])
    hr_f = hr.permute(0, 2, 3, 1).reshape(-1, hr.shape[1])

    if mask is not None and USE_MASKED_LOSS:
        mask_f = mask.permute(0, 2, 3, 1).reshape(-1, mask.shape[1])
        valid = mask_f.mean(dim=1) > 0.5
        if valid.sum() == 0:
            return torch.tensor(0.0, device=sr.device)
        sr_f = sr_f[valid]
        hr_f = hr_f[valid]

    dot = torch.sum(sr_f * hr_f, dim=1)
    sr_norm = torch.norm(sr_f, dim=1)
    hr_norm = torch.norm(hr_f, dim=1)

    denom = sr_norm * hr_norm + 1e-8

    cos = dot / denom
    cos = torch.clamp(cos, -1.0 + 1e-6, 1.0 - 1e-6)

    angle = torch.acos(cos)

    return torch.mean(angle)


def vegetation_indices_torch(x):
    green = x[:, 1:2, :, :]
    red = x[:, 2:3, :, :]
    rededge = x[:, 3:4, :, :]
    nir = x[:, 4:5, :, :]

    ndvi = (nir - red) / (nir + red + 1e-6)
    gndvi = (nir - green) / (nir + green + 1e-6)
    ndre = (nir - rededge) / (nir + rededge + 1e-6)

    return ndvi, gndvi, ndre


def vi_loss(sr, hr, mask=None):
    if sr.shape[1] != 5 or hr.shape[1] != 5:
        return torch.tensor(0.0, device=sr.device)

    sr_ndvi, sr_gndvi, sr_ndre = vegetation_indices_torch(sr)
    hr_ndvi, hr_gndvi, hr_ndre = vegetation_indices_torch(hr)

    if mask is not None and USE_MASKED_LOSS:
        # 식생지수는 5밴드가 모두 유효한 픽셀만 사용
        pix_mask = (mask.mean(dim=1, keepdim=True) > 0.5).float()
        loss = (
            masked_l1_loss(sr_ndvi, hr_ndvi, pix_mask) +
            masked_l1_loss(sr_gndvi, hr_gndvi, pix_mask) +
            masked_l1_loss(sr_ndre, hr_ndre, pix_mask)
        ) / 3.0
    else:
        loss = (
            F.l1_loss(sr_ndvi, hr_ndvi) +
            F.l1_loss(sr_gndvi, hr_gndvi) +
            F.l1_loss(sr_ndre, hr_ndre)
        ) / 3.0

    return loss


def gradient_loss(sr, hr, mask=None):
    """1차 gradient 차이를 줄여 경계와 줄무늬 구조를 더 선명하게 유지."""
    sr_dx = sr[:, :, :, 1:] - sr[:, :, :, :-1]
    hr_dx = hr[:, :, :, 1:] - hr[:, :, :, :-1]

    sr_dy = sr[:, :, 1:, :] - sr[:, :, :-1, :]
    hr_dy = hr[:, :, 1:, :] - hr[:, :, :-1, :]

    if mask is not None and USE_MASKED_LOSS:
        mask_dx = mask[:, :, :, 1:] * mask[:, :, :, :-1]
        mask_dy = mask[:, :, 1:, :] * mask[:, :, :-1, :]

        loss_dx = torch.abs(sr_dx - hr_dx) * mask_dx
        loss_dy = torch.abs(sr_dy - hr_dy) * mask_dy

        loss_dx = loss_dx.sum() / (mask_dx.sum() + 1e-8)
        loss_dy = loss_dy.sum() / (mask_dy.sum() + 1e-8)
    else:
        loss_dx = F.l1_loss(sr_dx, hr_dx)
        loss_dy = F.l1_loss(sr_dy, hr_dy)

    return loss_dx + loss_dy


def laplacian_edge_loss(sr, hr, mask=None):
    """Laplacian edge 차이를 줄여 고주파 detail을 보존."""
    channels = sr.shape[1]

    kernel = torch.tensor(
        [[0.0, -1.0, 0.0],
         [-1.0, 4.0, -1.0],
         [0.0, -1.0, 0.0]],
        dtype=sr.dtype,
        device=sr.device
    ).view(1, 1, 3, 3)

    kernel = kernel.repeat(channels, 1, 1, 1)

    sr_edge = F.conv2d(sr, kernel, padding=1, groups=channels)
    hr_edge = F.conv2d(hr, kernel, padding=1, groups=channels)

    if mask is not None and USE_MASKED_LOSS:
        loss = torch.abs(sr_edge - hr_edge) * mask
        return loss.sum() / (mask.sum() + 1e-8)

    return F.l1_loss(sr_edge, hr_edge)



def masked_charbonnier_loss(sr, hr, mask=None, eps=1e-3):
    """L1보다 부드럽지만 outlier에 강한 Charbonnier loss."""
    diff = sr - hr
    loss = torch.sqrt(diff * diff + eps * eps)

    if mask is not None and USE_MASKED_LOSS:
        mask = mask.float()
        loss = loss * mask
        return loss.sum() / (mask.sum() + 1e-8)

    return loss.mean()


def masked_ssim_loss(sr, hr, mask=None, window_size=7):
    """채널별 differentiable SSIM loss. 값이 낮을수록 HR 구조와 가까움."""
    pad = window_size // 2
    channels = sr.shape[1]

    # 반사율 데이터는 scene마다 range가 다를 수 있으므로 안정적인 상수 사용
    c1 = 0.01 ** 2
    c2 = 0.03 ** 2

    mu_x = F.avg_pool2d(sr, window_size, stride=1, padding=pad)
    mu_y = F.avg_pool2d(hr, window_size, stride=1, padding=pad)

    sigma_x = F.avg_pool2d(sr * sr, window_size, stride=1, padding=pad) - mu_x * mu_x
    sigma_y = F.avg_pool2d(hr * hr, window_size, stride=1, padding=pad) - mu_y * mu_y
    sigma_xy = F.avg_pool2d(sr * hr, window_size, stride=1, padding=pad) - mu_x * mu_y

    ssim_map = ((2 * mu_x * mu_y + c1) * (2 * sigma_xy + c2)) / (
        (mu_x * mu_x + mu_y * mu_y + c1) * (sigma_x + sigma_y + c2) + 1e-8
    )

    loss_map = 1.0 - torch.clamp(ssim_map, -1.0, 1.0)

    if mask is not None and USE_MASKED_LOSS:
        mask = mask.float()
        return (loss_map * mask).sum() / (mask.sum() + 1e-8)

    return loss_map.mean()


def frequency_loss(sr, hr, mask=None):
    """FFT amplitude 차이를 줄여 흐릿한 SR을 억제하는 고주파 보존 loss."""
    if mask is not None and USE_MASKED_LOSS:
        sr = sr * mask.float()
        hr = hr * mask.float()

    sr_fft = torch.fft.rfft2(sr, norm="ortho")
    hr_fft = torch.fft.rfft2(hr, norm="ortho")

    sr_amp = torch.log1p(torch.abs(sr_fft))
    hr_amp = torch.log1p(torch.abs(hr_fft))

    return F.l1_loss(sr_amp, hr_amp)


def standard_sr_loss(sr, hr, mask=None):
    l1 = masked_l1_loss(sr, hr, mask)
    charb = masked_charbonnier_loss(sr, hr, mask)
    ssim_l = masked_ssim_loss(sr, hr, mask)
    freq = frequency_loss(sr, hr, mask)

    if USE_SHARPNESS_LOSS:
        grad = gradient_loss(sr, hr, mask)
        edge = laplacian_edge_loss(sr, hr, mask)
    else:
        grad = torch.tensor(0.0, device=sr.device)
        edge = torch.tensor(0.0, device=sr.device)

    if sr.shape[1] == 5:
        sam = sam_loss(sr, hr, mask)
        vi = vi_loss(sr, hr, mask)
        loss = (
            L1_WEIGHT * l1 +
            CHARBONNIER_WEIGHT * charb +
            SSIM_WEIGHT * ssim_l +
            GRAD_WEIGHT * grad +
            EDGE_WEIGHT * edge +
            FREQ_WEIGHT * freq +
            SAM_WEIGHT * sam +
            VI_WEIGHT * vi
        )
    else:
        loss = (
            L1_WEIGHT * l1 +
            CHARBONNIER_WEIGHT * charb +
            SSIM_WEIGHT * ssim_l +
            GRAD_WEIGHT * grad +
            EDGE_WEIGHT * edge +
            FREQ_WEIGHT * freq
        )

    return loss


# ============================================================
# 11. Training
# ============================================================

def get_checkpoint_name(mode, method, scale, band_name=None):
    if mode == "joint5ch":
        return f"{mode}_{method}_x{scale}_best.pth"

    if mode == "bandwise":
        return f"{mode}_{method}_{band_name}_x{scale}_best.pth"

    raise ValueError(f"Unsupported mode: {mode}")


def get_checkpoint_path(mode, method, scale, band_name=None):
    """
    기존 훈련 모델은 CHECKPOINT_DIR에서 불러옵니다.
    새 NDVI/평가 결과는 OUTPUT_DIR에 저장하므로 기존 metrics를 덮어쓰지 않습니다.
    """
    return CHECKPOINT_DIR / get_checkpoint_name(
        mode=mode,
        method=method,
        scale=scale,
        band_name=band_name,
    )


def load_trained_model_from_checkpoint(method, scale, mode="joint5ch", channels=5, band_index=None, band_name=None):
    ckpt_path = get_checkpoint_path(mode=mode, method=method, scale=scale, band_name=band_name)
    if not ckpt_path.exists():
        return None
    if method == "esrgan":
        model = ESRGANGenerator(scale=scale, channels=channels, nf=64, n_rrdb=6).to(DEVICE)
        checkpoint = torch.load(ckpt_path, map_location=DEVICE)
        if "generator_state_dict" in checkpoint:
            model.load_state_dict(checkpoint["generator_state_dict"], strict=True)
        else:
            state = extract_state_dict(checkpoint)
            model.load_state_dict(state, strict=False)
    else:
        model = build_model(method, scale, channels=channels).to(DEVICE)
        checkpoint = torch.load(ckpt_path, map_location=DEVICE)
        if "model_state_dict" in checkpoint:
            model.load_state_dict(checkpoint["model_state_dict"], strict=True)
        else:
            state = extract_state_dict(checkpoint)
            model.load_state_dict(state, strict=False)
    model.eval()
    print(f"[INFO] Loaded existing checkpoint and skipped training: {ckpt_path}")
    return model


def train_standard_model(
    method,
    scale,
    train_records,
    val_records,
    mode="joint5ch",
    channels=5,
    band_index=None,
    band_name=None,
):
    print("\n====================================")
    print(f"Train {mode} {method} x{scale}")
    if mode == "bandwise":
        print(f"Band: {band_name}")
    print("====================================")

    if SKIP_TRAIN_IF_CHECKPOINT_EXISTS:
        existing_model = load_trained_model_from_checkpoint(
            method=method,
            scale=scale,
            mode=mode,
            channels=channels,
            band_index=band_index,
            band_name=band_name,
        )
        if existing_model is not None:
            return existing_model

    model = build_model(method, scale, channels=channels).to(DEVICE)

    model_info = count_model_parameters(model)
    print(f"[INFO] Params total    : {model_info['params_total']:,}")
    print(f"[INFO] Params trainable: {model_info['params_trainable']:,}")

    train_mode = TRAIN_MODE.get(method, "scratch")
    start_epoch = 0

    if train_mode == "finetune":
        print("[INFO] Train mode: finetune")
        model = load_partial_pretrained(
            model=model,
            pretrained_path=PRETRAINED_PATHS.get(method, ""),
            strict=False
        ).to(DEVICE)

    elif train_mode == "resume":
        print("[INFO] Train mode: resume")
        model, start_epoch = load_resume_checkpoint(
            model=model,
            checkpoint_path=RESUME_CHECKPOINTS.get(method, ""),
            method=method
        )
        model = model.to(DEVICE)

    else:
        print("[INFO] Train mode: scratch")

    train_dataset = SRPatchDataset(
        records=train_records,
        scale=scale,
        channels=channels,
        band_index=band_index,
        patches_per_epoch=PATCHES_PER_EPOCH,
        train=True
    )

    val_dataset = SRPatchDataset(
        records=val_records,
        scale=scale,
        channels=channels,
        band_index=band_index,
        patches_per_epoch=max(len(val_records) * VAL_PATCHES_PER_SCENE, 1),
        train=False
    )

    train_loader = make_dataloader(
        dataset=train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY
    )

    val_loader = make_dataloader(
        dataset=val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LR,
        weight_decay=WEIGHT_DECAY
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=EPOCHS,
        eta_min=LR * 0.01
    )

    amp_enabled = bool(USE_AMP and DEVICE == "cuda" and torch.cuda.is_available())
    scaler = torch.cuda.amp.GradScaler(enabled=amp_enabled)

    best_val = float("inf")
    epochs_no_improve = 0

    ckpt_name = get_checkpoint_name(
        mode=mode,
        method=method,
        scale=scale,
        band_name=band_name
    )

    ckpt_path = get_checkpoint_path(mode=mode, method=method, scale=scale, band_name=band_name)

    history = []
    train_start_total = time.perf_counter()

    for epoch in range(start_epoch + 1, EPOCHS + 1):
        epoch_start_time = time.perf_counter()

        model.train()

        train_losses = []
        train_start_time = time.perf_counter()

        for lr_patch, hr_patch, mask_patch in train_loader:
            lr_patch = lr_patch.to(DEVICE, non_blocking=True)
            hr_patch = hr_patch.to(DEVICE, non_blocking=True)
            mask_patch = mask_patch.to(DEVICE, non_blocking=True)

            if not torch.isfinite(lr_patch).all() or not torch.isfinite(hr_patch).all():
                print("[WARN] Non-finite input patch detected. Skip this batch.")
                continue

            optimizer.zero_grad(set_to_none=True)

            with torch.cuda.amp.autocast(enabled=amp_enabled):
                sr_patch = model(lr_patch)
                loss = standard_sr_loss(sr_patch, hr_patch, mask_patch)

            if not torch.isfinite(loss):
                print("[WARN] Non-finite loss detected. Skip this batch.")
                continue

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()

            train_losses.append(float(loss.detach().cpu().item()))

        train_time_sec = time.perf_counter() - train_start_time

        scheduler.step()

        model.eval()

        val_losses = []
        val_start_time = time.perf_counter()

        with torch.no_grad():
            for lr_patch, hr_patch, mask_patch in val_loader:
                lr_patch = lr_patch.to(DEVICE, non_blocking=True)
                hr_patch = hr_patch.to(DEVICE, non_blocking=True)
                mask_patch = mask_patch.to(DEVICE, non_blocking=True)

                if not torch.isfinite(lr_patch).all() or not torch.isfinite(hr_patch).all():
                    print("[WARN] Non-finite val input patch detected. Skip this batch.")
                    continue

                sr_patch = model(lr_patch)
                loss = standard_sr_loss(sr_patch, hr_patch, mask_patch)

                if not torch.isfinite(loss):
                    print("[WARN] Non-finite val loss detected. Skip this batch.")
                    continue

                val_losses.append(loss.item())

        val_time_sec = time.perf_counter() - val_start_time
        epoch_time_sec = time.perf_counter() - epoch_start_time

        train_loss = float(np.mean(train_losses)) if len(train_losses) > 0 else float("nan")
        val_loss = float(np.mean(val_losses)) if len(val_losses) > 0 else train_loss

        history.append({
            "epoch": epoch,
            "mode": mode,
            "method": method,
            "scale": scale,
            "band": band_name if band_name is not None else "all",
            "train_loss": train_loss,
            "val_loss": val_loss,
            "lr": optimizer.param_groups[0]["lr"],
            "train_time_sec": train_time_sec,
            "val_time_sec": val_time_sec,
            "epoch_time_sec": epoch_time_sec,
            "train_time_hms": format_seconds(train_time_sec),
            "val_time_hms": format_seconds(val_time_sec),
            "epoch_time_hms": format_seconds(epoch_time_sec),
            "params_total": model_info["params_total"],
            "params_trainable": model_info["params_trainable"],
            "datetime": now_string(),
        })

        print(
            f"[{mode} {method} x{scale}] "
            f"Epoch {epoch:03d}/{EPOCHS} | "
            f"Train={train_loss:.6f} | "
            f"Val={val_loss:.6f} | "
            f"Time={format_seconds(epoch_time_sec)}"
        )

        improved = np.isfinite(val_loss) and (best_val - val_loss) > MIN_DELTA

        if improved:
            best_val = val_loss
            epochs_no_improve = 0

            if SAVE_CHECKPOINT:
                torch.save({
                    "mode": mode,
                    "method": method,
                    "scale": scale,
                    "channels": channels,
                    "band_index": band_index,
                    "band_name": band_name,
                    "model_state_dict": model.state_dict(),
                    "best_val": best_val,
                    "epoch": epoch,
                    "params_total": model_info["params_total"],
                    "params_trainable": model_info["params_trainable"],
                    "early_stopping": EARLY_STOPPING,
                    "use_amp": amp_enabled,
                }, ckpt_path)
        else:
            epochs_no_improve += 1

        if EARLY_STOPPING and epochs_no_improve >= PATIENCE:
            print(
                f"[INFO] Early stopping triggered: {mode} {method} x{scale} "
                f"band={band_name if band_name is not None else 'all'} | "
                f"best_val={best_val:.6f} | patience={PATIENCE}"
            )
            break

    if mode == "joint5ch":
        history_name = f"train_history_{mode}_{method}_x{scale}.csv"
    else:
        history_name = f"train_history_{mode}_{method}_{band_name}_x{scale}.csv"

    history_path = OUTPUT_DIR / "sr_logs" / history_name
    write_dict_csv(history_path, history)

    if SAVE_CHECKPOINT and ckpt_path.exists():
        checkpoint = torch.load(ckpt_path, map_location=DEVICE)
        model.load_state_dict(checkpoint["model_state_dict"])

    total_train_time_sec = time.perf_counter() - train_start_total

    print("\n====================================")
    print(f"Training time summary: {mode} {method} x{scale}")
    if mode == "bandwise":
        print(f"Band: {band_name}")
    print(f"Total training time: {format_seconds(total_train_time_sec)}")
    print("====================================")

    return model


def train_esrgan(
    scale,
    train_records,
    val_records,
    mode="joint5ch",
    channels=5,
    band_index=None,
    band_name=None,
):
    method = "esrgan"

    print("\n====================================")
    print(f"Train {mode} ESRGAN x{scale}")
    if mode == "bandwise":
        print(f"Band: {band_name}")
    print("====================================")

    if SKIP_TRAIN_IF_CHECKPOINT_EXISTS:
        existing_model = load_trained_model_from_checkpoint(
            method=method,
            scale=scale,
            mode=mode,
            channels=channels,
            band_index=band_index,
            band_name=band_name,
        )
        if existing_model is not None:
            return existing_model

    generator = ESRGANGenerator(
        scale=scale,
        channels=channels,
        nf=64,
        n_rrdb=6
    ).to(DEVICE)

    discriminator = ESRGANDiscriminator(
        channels=channels,
        base=64
    ).to(DEVICE)

    gen_info = count_model_parameters(generator)
    disc_info = count_model_parameters(discriminator)

    print(f"[INFO] Generator params total    : {gen_info['params_total']:,}")
    print(f"[INFO] Generator params trainable: {gen_info['params_trainable']:,}")
    print(f"[INFO] Discriminator params total    : {disc_info['params_total']:,}")
    print(f"[INFO] Discriminator params trainable: {disc_info['params_trainable']:,}")

    train_mode = TRAIN_MODE.get(method, "scratch")
    start_epoch = 0

    if train_mode == "finetune":
        print("[INFO] Train mode: finetune")
        generator = load_partial_pretrained(
            model=generator,
            pretrained_path=PRETRAINED_PATHS.get(method, ""),
            strict=False
        ).to(DEVICE)

    elif train_mode == "resume":
        print("[INFO] Train mode: resume")
        generator, start_epoch = load_resume_checkpoint(
            model=generator,
            checkpoint_path=RESUME_CHECKPOINTS.get(method, ""),
            method=method
        )
        generator = generator.to(DEVICE)

    else:
        print("[INFO] Train mode: scratch")

    train_dataset = SRPatchDataset(
        records=train_records,
        scale=scale,
        channels=channels,
        band_index=band_index,
        patches_per_epoch=PATCHES_PER_EPOCH,
        train=True
    )

    val_dataset = SRPatchDataset(
        records=val_records,
        scale=scale,
        channels=channels,
        band_index=band_index,
        patches_per_epoch=max(len(val_records) * VAL_PATCHES_PER_SCENE, 1),
        train=False
    )

    train_loader = make_dataloader(
        dataset=train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY
    )

    val_loader = make_dataloader(
        dataset=val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY
    )

    opt_g = torch.optim.AdamW(
        generator.parameters(),
        lr=LR,
        weight_decay=WEIGHT_DECAY
    )

    opt_d = torch.optim.AdamW(
        discriminator.parameters(),
        lr=LR,
        weight_decay=WEIGHT_DECAY
    )

    bce = nn.BCEWithLogitsLoss()

    amp_enabled = bool(USE_AMP and DEVICE == "cuda" and torch.cuda.is_available())
    scaler_g = torch.cuda.amp.GradScaler(enabled=amp_enabled)
    scaler_d = torch.cuda.amp.GradScaler(enabled=amp_enabled)

    best_val = float("inf")
    epochs_no_improve = 0

    ckpt_name = get_checkpoint_name(
        mode=mode,
        method=method,
        scale=scale,
        band_name=band_name
    )

    ckpt_path = get_checkpoint_path(mode=mode, method=method, scale=scale, band_name=band_name)

    history = []
    train_start_total = time.perf_counter()

    for epoch in range(start_epoch + 1, EPOCHS + 1):
        epoch_start_time = time.perf_counter()

        generator.train()
        discriminator.train()

        g_losses = []
        d_losses = []

        use_gan = epoch > ESRGAN_PRETRAIN_EPOCHS

        train_start_time = time.perf_counter()

        for lr_patch, hr_patch, mask_patch in train_loader:
            lr_patch = lr_patch.to(DEVICE, non_blocking=True)
            hr_patch = hr_patch.to(DEVICE, non_blocking=True)
            mask_patch = mask_patch.to(DEVICE, non_blocking=True)

            if not torch.isfinite(lr_patch).all() or not torch.isfinite(hr_patch).all():
                print("[WARN] Non-finite ESRGAN input patch detected. Skip this batch.")
                continue

            opt_g.zero_grad(set_to_none=True)

            with torch.cuda.amp.autocast(enabled=amp_enabled):
                sr_patch = generator(lr_patch)

                l1 = masked_l1_loss(sr_patch, hr_patch, mask_patch)

                if channels == 5:
                    sam = sam_loss(sr_patch, hr_patch, mask_patch)
                    vi = vi_loss(sr_patch, hr_patch, mask_patch)
                else:
                    sam = torch.tensor(0.0, device=DEVICE)
                    vi = torch.tensor(0.0, device=DEVICE)

                if use_gan:
                    pred_fake_for_g = discriminator(sr_patch)
                    valid_label = torch.ones_like(pred_fake_for_g)
                    adv = bce(pred_fake_for_g, valid_label)

                    g_loss = (
                        ESRGAN_L1_WEIGHT * l1 +
                        ESRGAN_SAM_WEIGHT * sam +
                        ESRGAN_VI_WEIGHT * vi +
                        ESRGAN_ADV_WEIGHT * adv
                    )
                else:
                    g_loss = (
                        ESRGAN_L1_WEIGHT * l1 +
                        ESRGAN_SAM_WEIGHT * sam +
                        ESRGAN_VI_WEIGHT * vi
                    )

            if not torch.isfinite(g_loss):
                print("[WARN] Non-finite generator loss detected. Skip this batch.")
                continue

            scaler_g.scale(g_loss).backward()
            scaler_g.unscale_(opt_g)
            torch.nn.utils.clip_grad_norm_(generator.parameters(), max_norm=1.0)
            scaler_g.step(opt_g)
            scaler_g.update()

            g_losses.append(float(g_loss.detach().cpu().item()))

            if use_gan:
                opt_d.zero_grad(set_to_none=True)

                with torch.no_grad():
                    fake_detached = sr_patch.detach()

                with torch.cuda.amp.autocast(enabled=amp_enabled):
                    pred_real = discriminator(hr_patch)
                    pred_fake = discriminator(fake_detached)

                    real_label = torch.ones_like(pred_real)
                    fake_label = torch.zeros_like(pred_fake)

                    d_real = bce(pred_real, real_label)
                    d_fake = bce(pred_fake, fake_label)

                    d_loss = 0.5 * (d_real + d_fake)

                if not torch.isfinite(d_loss):
                    print("[WARN] Non-finite discriminator loss detected. Skip this batch.")
                    continue

                scaler_d.scale(d_loss).backward()
                scaler_d.unscale_(opt_d)
                torch.nn.utils.clip_grad_norm_(discriminator.parameters(), max_norm=1.0)
                scaler_d.step(opt_d)
                scaler_d.update()

                d_losses.append(float(d_loss.detach().cpu().item()))

        train_time_sec = time.perf_counter() - train_start_time

        generator.eval()

        val_losses = []
        val_start_time = time.perf_counter()

        with torch.no_grad():
            for lr_patch, hr_patch, mask_patch in val_loader:
                lr_patch = lr_patch.to(DEVICE, non_blocking=True)
                hr_patch = hr_patch.to(DEVICE, non_blocking=True)
                mask_patch = mask_patch.to(DEVICE, non_blocking=True)

                if not torch.isfinite(lr_patch).all() or not torch.isfinite(hr_patch).all():
                    print("[WARN] Non-finite ESRGAN val input patch detected. Skip this batch.")
                    continue

                with torch.cuda.amp.autocast(enabled=amp_enabled):
                    sr_patch = generator(lr_patch)
                    loss = standard_sr_loss(sr_patch, hr_patch, mask_patch)

                if not torch.isfinite(loss):
                    print("[WARN] Non-finite ESRGAN val loss detected. Skip this batch.")
                    continue

                val_losses.append(loss.item())

        val_time_sec = time.perf_counter() - val_start_time
        epoch_time_sec = time.perf_counter() - epoch_start_time

        g_loss_mean = float(np.mean(g_losses)) if len(g_losses) > 0 else float("nan")
        d_loss_mean = float(np.mean(d_losses)) if len(d_losses) > 0 else 0.0
        val_loss = float(np.mean(val_losses)) if len(val_losses) > 0 else g_loss_mean

        history.append({
            "epoch": epoch,
            "mode": mode,
            "method": method,
            "scale": scale,
            "band": band_name if band_name is not None else "all",
            "g_loss": g_loss_mean,
            "d_loss": d_loss_mean,
            "val_loss": val_loss,
            "use_gan": int(use_gan),
            "train_time_sec": train_time_sec,
            "val_time_sec": val_time_sec,
            "epoch_time_sec": epoch_time_sec,
            "train_time_hms": format_seconds(train_time_sec),
            "val_time_hms": format_seconds(val_time_sec),
            "epoch_time_hms": format_seconds(epoch_time_sec),
            "generator_params_total": gen_info["params_total"],
            "generator_params_trainable": gen_info["params_trainable"],
            "discriminator_params_total": disc_info["params_total"],
            "discriminator_params_trainable": disc_info["params_trainable"],
            "datetime": now_string(),
        })

        print(
            f"[{mode} ESRGAN x{scale}] "
            f"Epoch {epoch:03d}/{EPOCHS} | "
            f"G={g_loss_mean:.6f} | "
            f"D={d_loss_mean:.6f} | "
            f"Val={val_loss:.6f} | "
            f"GAN={use_gan} | "
            f"Time={format_seconds(epoch_time_sec)}"
        )

        improved = np.isfinite(val_loss) and (best_val - val_loss) > MIN_DELTA

        if improved:
            best_val = val_loss
            epochs_no_improve = 0

            if SAVE_CHECKPOINT:
                torch.save({
                    "mode": mode,
                    "method": method,
                    "scale": scale,
                    "channels": channels,
                    "band_index": band_index,
                    "band_name": band_name,
                    "generator_state_dict": generator.state_dict(),
                    "discriminator_state_dict": discriminator.state_dict(),
                    "best_val": best_val,
                    "epoch": epoch,
                    "generator_params_total": gen_info["params_total"],
                    "generator_params_trainable": gen_info["params_trainable"],
                    "discriminator_params_total": disc_info["params_total"],
                    "discriminator_params_trainable": disc_info["params_trainable"],
                    "early_stopping": EARLY_STOPPING,
                    "use_amp": amp_enabled,
                }, ckpt_path)
        else:
            epochs_no_improve += 1

        if EARLY_STOPPING and epochs_no_improve >= PATIENCE:
            print(
                f"[INFO] Early stopping triggered: {mode} ESRGAN x{scale} "
                f"band={band_name if band_name is not None else 'all'} | "
                f"best_val={best_val:.6f} | patience={PATIENCE}"
            )
            break

    if mode == "joint5ch":
        history_name = f"train_history_{mode}_{method}_x{scale}.csv"
    else:
        history_name = f"train_history_{mode}_{method}_{band_name}_x{scale}.csv"

    history_path = OUTPUT_DIR / "sr_logs" / history_name
    write_dict_csv(history_path, history)

    if SAVE_CHECKPOINT and ckpt_path.exists():
        checkpoint = torch.load(ckpt_path, map_location=DEVICE)
        generator.load_state_dict(checkpoint["generator_state_dict"])

    total_train_time_sec = time.perf_counter() - train_start_total

    print("\n====================================")
    print(f"Training time summary: {mode} ESRGAN x{scale}")
    if mode == "bandwise":
        print(f"Band: {band_name}")
    print(f"Total training time: {format_seconds(total_train_time_sec)}")
    print("====================================")

    return generator


def train_bandwise_models(method, scale, train_records, val_records):
    models = {}

    for band_index, band_name in enumerate(BANDS):
        print("\n####################################")
        print(f"Band-wise training: {method} x{scale} | Band={band_name}")
        print("####################################")

        if method == "bicubic":
            models[band_name] = None

        elif method == "esrgan":
            model = train_esrgan(
                scale=scale,
                train_records=train_records,
                val_records=val_records,
                mode="bandwise",
                channels=1,
                band_index=band_index,
                band_name=band_name
            )

            models[band_name] = model

        else:
            model = train_standard_model(
                method=method,
                scale=scale,
                train_records=train_records,
                val_records=val_records,
                mode="bandwise",
                channels=1,
                band_index=band_index,
                band_name=band_name
            )

            models[band_name] = model

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return models


# ============================================================
# 13-1. Training loss plot only
# ============================================================

def plot_training_loss_curves_from_history():
    """
    Train-only 후처리:
    - sr_logs/train_history_*.csv 파일을 읽음
    - 모델별 train loss 그래프만 저장
    - 평가 metric 계산은 수행하지 않음

    저장 위치:
      sr_logs/train_loss_plots/individual/
      sr_logs/train_loss_plots/combined/
      sr_logs/train_loss_plots/train_loss_plot_summary.csv
    """
    log_dir = OUTPUT_DIR / "sr_logs"
    plot_root = log_dir / "train_loss_plots"
    individual_dir = plot_root / "individual"
    combined_dir = plot_root / "combined"

    individual_dir.mkdir(parents=True, exist_ok=True)
    combined_dir.mkdir(parents=True, exist_ok=True)

    history_files = sorted([
        p for p in log_dir.glob("train_history_*.csv")
        if p.name != "train_history_all.csv"
    ])

    if len(history_files) == 0:
        print("[WARN] No train_history_*.csv files found. Skip loss plotting.")
        return

    dfs = []

    for path in history_files:
        try:
            df = pd.read_csv(path)

            if df.empty:
                print(f"[WARN] Empty train history skipped: {path.name}")
                continue

            df["source_file"] = path.name

            if "train_loss" in df.columns:
                df["plot_train_loss"] = pd.to_numeric(df["train_loss"], errors="coerce")
                df["loss_source"] = "train_loss"
            elif "g_loss" in df.columns:
                # ESRGAN uses generator loss as training loss.
                df["plot_train_loss"] = pd.to_numeric(df["g_loss"], errors="coerce")
                df["loss_source"] = "g_loss"
            else:
                print(f"[WARN] No train_loss/g_loss column in {path.name}. Skip.")
                continue

            if "epoch" in df.columns:
                df["epoch"] = pd.to_numeric(df["epoch"], errors="coerce")
            else:
                print(f"[WARN] No epoch column in {path.name}. Skip.")
                continue

            if "scale" in df.columns:
                df["scale"] = pd.to_numeric(df["scale"], errors="coerce")
            else:
                df["scale"] = np.nan

            for col in ["mode", "method", "band"]:
                if col not in df.columns:
                    df[col] = "unknown"

            df["band"] = df["band"].fillna("all").astype(str)
            df["mode"] = df["mode"].fillna("unknown").astype(str)
            df["method"] = df["method"].fillna("unknown").astype(str)

            df = df.dropna(subset=["epoch", "plot_train_loss"])

            if df.empty:
                print(f"[WARN] No valid loss rows in {path.name}. Skip.")
                continue

            dfs.append(df)

        except Exception as e:
            print(f"[WARN] Failed to read train history: {path} | {e}")

    if len(dfs) == 0:
        print("[WARN] No valid train history data. Skip loss plotting.")
        return

    all_df = pd.concat(dfs, ignore_index=True, sort=False)

    merged_csv = plot_root / "train_history_for_loss_plots.csv"
    all_df.to_csv(merged_csv, index=False, encoding="utf-8-sig")

    def _safe_name(x):
        x = str(x)
        x = re.sub(r"[^\w\-.]+", "_", x)
        x = re.sub(r"_+", "_", x)
        return x.strip("_")

    def _scale_label(x):
        try:
            return f"x{int(float(x))}"
        except Exception:
            return f"x{x}"

    try:
        import matplotlib.pyplot as plt
    except Exception as e:
        print(f"[WARN] matplotlib import failed. CSV is saved but plots are skipped. Reason: {e}")
        return

    plt.rcParams["font.family"] = "Times New Roman"
    plt.rcParams["axes.unicode_minus"] = False

    summary_rows = []

    # --------------------------------------------------------
    # 1) Individual loss curve per scale/mode/method/band
    # --------------------------------------------------------
    for (scale, mode, method, band), g in all_df.groupby(["scale", "mode", "method", "band"], dropna=False):
        g = g.sort_values("epoch")

        if g.empty:
            continue

        scale_label = _scale_label(scale)
        band_label = str(band)

        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(
            g["epoch"].values,
            g["plot_train_loss"].values,
            linewidth=2.0,
            label="Training loss"
        )

        title = f"{scale_label} | {mode} | {method}"
        if band_label.lower() != "all":
            title += f" | {band_label}"

        ax.set_title(title, fontsize=14, fontweight="bold")
        ax.set_xlabel("Epoch", fontsize=12, fontweight="bold")
        ax.set_ylabel("Training loss", fontsize=12, fontweight="bold")
        ax.tick_params(axis="both", labelsize=10)
        ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.5)
        ax.legend(fontsize=9)
        fig.tight_layout()

        out_path = individual_dir / (
            f"train_loss_{scale_label}_{_safe_name(mode)}_"
            f"{_safe_name(method)}_{_safe_name(band_label)}.png"
        )

        fig.savefig(out_path, dpi=300, bbox_inches="tight")
        plt.close(fig)

        losses = g["plot_train_loss"].dropna().values

        summary_rows.append({
            "scale": scale_label,
            "mode": mode,
            "method": method,
            "band": band_label,
            "n_epochs": int(len(g)),
            "loss_start": float(losses[0]) if len(losses) else np.nan,
            "loss_end": float(losses[-1]) if len(losses) else np.nan,
            "loss_min": float(np.min(losses)) if len(losses) else np.nan,
            "loss_max": float(np.max(losses)) if len(losses) else np.nan,
            "loss_source": str(g["loss_source"].iloc[0]) if "loss_source" in g.columns else "",
            "plot_path": str(out_path),
        })

    # --------------------------------------------------------
    # 2) Combined curve by scale/mode
    #    bandwise는 method별로 밴드 평균 loss를 그립니다.
    # --------------------------------------------------------
    for (scale, mode), g_mode in all_df.groupby(["scale", "mode"], dropna=False):
        scale_label = _scale_label(scale)
        mode_label = str(mode)

        fig, ax = plt.subplots(figsize=(10, 6))

        for method, g_method in g_mode.groupby("method", dropna=False):
            gg = (
                g_method
                .groupby("epoch", as_index=False)["plot_train_loss"]
                .mean()
                .sort_values("epoch")
            )

            if gg.empty:
                continue

            ax.plot(
                gg["epoch"].values,
                gg["plot_train_loss"].values,
                linewidth=2.0,
                label=str(method)
            )

        ax.set_title(
            f"Training loss comparison | {scale_label} | {mode_label}",
            fontsize=14,
            fontweight="bold"
        )
        ax.set_xlabel("Epoch", fontsize=12, fontweight="bold")
        ax.set_ylabel("Training loss", fontsize=12, fontweight="bold")
        ax.tick_params(axis="both", labelsize=10)
        ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.5)
        ax.legend(fontsize=9, ncol=2)
        fig.tight_layout()

        out_path = combined_dir / f"train_loss_compare_{scale_label}_{_safe_name(mode_label)}.png"
        fig.savefig(out_path, dpi=300, bbox_inches="tight")
        plt.close(fig)

    summary_df = pd.DataFrame(summary_rows)
    summary_csv = plot_root / "train_loss_plot_summary.csv"
    summary_df.to_csv(summary_csv, index=False, encoding="utf-8-sig")

    print("\n====================================")
    print("Training loss plots saved")
    print("====================================")
    print(f"Merged loss CSV : {merged_csv}")
    print(f"Summary CSV     : {summary_csv}")
    print(f"Individual dir  : {individual_dir}")
    print(f"Combined dir    : {combined_dir}")
    print("====================================")

# ============================================================
# 14. Main
# ============================================================



def merge_train_history_files():
    """
    sr_logs 폴더에 분리 저장된 train_history_*.csv 파일을 하나로 통합 저장합니다.

    저장 결과:
      1) sr_logs/train_history_all.csv
      2) sr_logs/train_history_all.xlsx

    주의:
      - 이전 실행에서 생성된 train_history_all.csv는 다시 읽지 않도록 제외합니다.
      - joint5ch, bandwise, ESRGAN history처럼 컬럼 구성이 일부 달라도 concat(sort=False)로 통합합니다.
    """
    log_dir = OUTPUT_DIR / "sr_logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    history_files = sorted([
        p for p in log_dir.glob("train_history_*.csv")
        if p.name != "train_history_all.csv"
    ])

    if len(history_files) == 0:
        print("[WARN] No train_history_*.csv files found to merge.")
        return

    dfs = []

    for csv_path in history_files:
        try:
            df = pd.read_csv(csv_path)

            if df.empty:
                print(f"[WARN] Empty train history file skipped: {csv_path}")
                continue

            # 원본 파일명을 기록해두면 나중에 어떤 개별 history에서 온 행인지 추적 가능
            df["source_history_file"] = csv_path.name

            dfs.append(df)

        except Exception as e:
            print(f"[WARN] Failed to read train history file: {csv_path}")
            print(f"Reason: {e}")

    if len(dfs) == 0:
        print("[WARN] No valid train history files to merge.")
        return

    merged_df = pd.concat(dfs, ignore_index=True, sort=False)

    # 숫자 컬럼은 가능하면 숫자형으로 변환
    for col in [
        "epoch",
        "scale",
        "train_loss",
        "val_loss",
        "g_loss",
        "d_loss",
        "lr",
        "train_time_sec",
        "val_time_sec",
        "epoch_time_sec",
        "params_total",
        "params_trainable",
        "generator_params_total",
        "generator_params_trainable",
        "discriminator_params_total",
        "discriminator_params_trainable",
    ]:
        if col in merged_df.columns:
            merged_df[col] = pd.to_numeric(merged_df[col], errors="coerce")

    # 보기 좋게 정렬
    sort_cols = [
        c for c in ["scale", "mode", "method", "band", "epoch", "source_history_file"]
        if c in merged_df.columns
    ]

    if len(sort_cols) > 0:
        merged_df = merged_df.sort_values(sort_cols, kind="stable").reset_index(drop=True)

    out_csv = log_dir / "train_history_all.csv"
    out_excel = log_dir / "train_history_all.xlsx"

    merged_df.to_csv(out_csv, index=False, encoding="utf-8-sig")

    # 모델/스케일/모드/밴드별 시간 요약표 생성
    group_cols = [c for c in ["scale", "mode", "method", "band"] if c in merged_df.columns]
    time_cols = [c for c in ["train_time_sec", "val_time_sec", "epoch_time_sec"] if c in merged_df.columns]
    loss_cols = [c for c in ["train_loss", "val_loss", "g_loss", "d_loss"] if c in merged_df.columns]

    if len(group_cols) > 0 and len(time_cols) > 0:
        time_summary_df = (
            merged_df
            .groupby(group_cols, dropna=False)[time_cols]
            .agg(["mean", "std", "min", "max", "sum"])
            .reset_index()
        )

        time_summary_df.columns = [
            "_".join([str(x) for x in col if str(x) != ""])
            for col in time_summary_df.columns
        ]
    else:
        time_summary_df = pd.DataFrame()

    if len(group_cols) > 0 and len(loss_cols) > 0:
        loss_summary_df = (
            merged_df
            .groupby(group_cols, dropna=False)[loss_cols]
            .agg(["mean", "std", "min", "max"])
            .reset_index()
        )

        loss_summary_df.columns = [
            "_".join([str(x) for x in col if str(x) != ""])
            for col in loss_summary_df.columns
        ]
    else:
        loss_summary_df = pd.DataFrame()

    # 파일별 epoch 수 확인용 요약표
    file_summary_rows = []
    for file_name, g in merged_df.groupby("source_history_file", dropna=False):
        row = {
            "source_history_file": file_name,
            "rows": int(len(g)),
        }

        for col in ["scale", "mode", "method", "band"]:
            if col in g.columns:
                values = [str(v) for v in g[col].dropna().unique().tolist()]
                row[col] = ", ".join(values)

        if "epoch" in g.columns:
            row["epoch_min"] = float(g["epoch"].min()) if pd.notna(g["epoch"].min()) else ""
            row["epoch_max"] = float(g["epoch"].max()) if pd.notna(g["epoch"].max()) else ""

        if "epoch_time_sec" in g.columns:
            row["epoch_time_sec_sum"] = float(g["epoch_time_sec"].sum())

        file_summary_rows.append(row)

    file_summary_df = pd.DataFrame(file_summary_rows)

    with pd.ExcelWriter(out_excel, engine="openpyxl") as writer:
        merged_df.to_excel(writer, sheet_name="All_train_history", index=False)
        time_summary_df.to_excel(writer, sheet_name="Time_summary", index=False)
        loss_summary_df.to_excel(writer, sheet_name="Loss_summary", index=False)
        file_summary_df.to_excel(writer, sheet_name="File_summary", index=False)

    print(f"[INFO] Saved merged train history CSV  : {out_csv}")
    print(f"[INFO] Saved merged train history Excel: {out_excel}")

def main():
    global RUN_CONFIG_CACHE

    total_start_time = time.perf_counter()

    set_seed(RANDOM_SEED)
    ensure_dirs()

    run_config = {
        "DATASET_DIR": str(DATASET_DIR),
        "SCENE_INDEX_PATH": str(SCENE_INDEX_PATH),
        "OUTPUT_ROOT": str(OUTPUT_ROOT),
        "RUN_NAME": RUN_NAME,
        "OUTPUT_DIR": str(OUTPUT_DIR),
        "CHECKPOINT_RUN_NAME": CHECKPOINT_RUN_NAME,
        "CHECKPOINT_DIR": str(CHECKPOINT_DIR),
        "DEVICE": DEVICE,
        "EXPERIMENT_PRESET": EXPERIMENT_PRESET,
        "USER_SCALES": USER_SCALES,
        "SCALES": SCALES,
        "EXPERIMENT_MODES": EXPERIMENT_MODES,
        "METHODS_TO_RUN": METHODS_TO_RUN,
        "JOINT_METHODS_TO_RUN": JOINT_METHODS_TO_RUN,
        "BANDWISE_METHODS_TO_RUN": BANDWISE_METHODS_TO_RUN,
        "RUN_JOINT_5CH": RUN_JOINT_5CH,
        "RUN_BANDWISE_ABLATION": RUN_BANDWISE_ABLATION,
        "EPOCHS": EPOCHS,
        "PATCHES_PER_EPOCH": PATCHES_PER_EPOCH,
        "BATCH_SIZE": BATCH_SIZE,
        "NUM_WORKERS": NUM_WORKERS,
        "LR": LR,
        "WEIGHT_DECAY": WEIGHT_DECAY,
        "SAVE_CHECKPOINT": SAVE_CHECKPOINT,
        "INPUT_ALREADY_NORMALIZED": INPUT_ALREADY_NORMALIZED,
        "USE_RESIDUAL_SR": USE_RESIDUAL_SR,
        "MODEL_FORWARD_CLAMP": False,
        "INFERENCE_CLIP_RANGE": [0.0, 1.5],
        "ESRGAN_RESIDUAL_SR": USE_RESIDUAL_SR,
        "RESIDUAL_DETAIL_ZERO_INIT": True,
        "VAL_PATCHES_PER_SCENE": VAL_PATCHES_PER_SCENE,
        "USE_SHARPNESS_LOSS": USE_SHARPNESS_LOSS,
        "USE_AMP": USE_AMP,
        "EARLY_STOPPING": EARLY_STOPPING,
        "PATIENCE": PATIENCE,
        "MIN_DELTA": MIN_DELTA,
        "TRAIN_ONLY_NO_EVAL": True,
        "SRCNN_IMPROVED_RESIDUAL": True,
    }

    RUN_CONFIG_CACHE = run_config

    run_config_path = OUTPUT_DIR / "sr_logs" / "run_config_train_only.json"
    with open(run_config_path, "w", encoding="utf-8") as f:
        json.dump(run_config, f, ensure_ascii=False, indent=4)

    print("\n====================================")
    print("5-band Multispectral SR Training-only Pipeline")
    print("Model training + checkpoint saving + train loss plots")
    print("Evaluation is disabled in this script.")
    print("====================================")
    print(f"Dataset    : {DATASET_DIR}")
    print(f"Output     : {OUTPUT_DIR}")
    print(f"Device     : {DEVICE}")
    print(f"Scales     : {SCALES}")
    print(f"Modes      : {EXPERIMENT_MODES}")
    print(f"Methods    : {METHODS_TO_RUN}")
    print(f"Epochs     : {EPOCHS}")
    print(f"Batch size : {BATCH_SIZE}")
    print(f"Val patches: {VAL_PATCHES_PER_SCENE} per scene")
    print(f"Save ckpt  : {SAVE_CHECKPOINT}")
    print(f"Residual SR: {USE_RESIDUAL_SR}")
    print(f"Start      : {now_string()}")
    print("====================================")

    records = read_scene_index()
    update_dynamic_sizes(records)

    train_records, val_records, test_records_unused = split_records(records)

    # 평가 전용 sample selection은 수행하지 않습니다.
    # test_records_unused는 split 확인용으로만 생성됩니다.
    print("\n====================================")
    print("Training-only split summary")
    print("====================================")
    print(f"Train records: {len(train_records)}")
    print(f"Val records  : {len(val_records)}")
    print(f"Test records : {len(test_records_unused)}  [not used]")
    print("====================================")

    all_training_jobs = []

    for scale in SCALES:
        print("\n####################################")
        print(f"Training scale x{scale}")
        print("####################################")

        if RUN_JOINT_5CH:
            mode = "joint5ch"

            for method in JOINT_METHODS_TO_RUN:
                print("\n------------------------------------")
                print(f"Train-only job: mode={mode}, method={method}, scale=x{scale}")
                print("------------------------------------")

                if method == "bicubic":
                    print("[INFO] Bicubic has no trainable parameters. Skip training.")
                    continue

                if method == "esrgan":
                    model = train_esrgan(
                        scale=scale,
                        train_records=train_records,
                        val_records=val_records,
                        mode=mode,
                        channels=5
                    )
                else:
                    model = train_standard_model(
                        method=method,
                        scale=scale,
                        train_records=train_records,
                        val_records=val_records,
                        mode=mode,
                        channels=5
                    )

                all_training_jobs.append({
                    "scale": scale,
                    "mode": mode,
                    "method": method,
                    "band": "all",
                    "datetime": now_string(),
                })

                del model

                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

        if RUN_BANDWISE_ABLATION:
            mode = "bandwise"

            for method in BANDWISE_METHODS_TO_RUN:
                print("\n------------------------------------")
                print(f"Train-only job: mode={mode}, method={method}, scale=x{scale}")
                print("------------------------------------")

                if method == "bicubic":
                    print("[INFO] Bicubic has no trainable parameters. Skip training.")
                    continue

                models = train_bandwise_models(
                    method=method,
                    scale=scale,
                    train_records=train_records,
                    val_records=val_records
                )

                for band_name in BANDS:
                    all_training_jobs.append({
                        "scale": scale,
                        "mode": mode,
                        "method": method,
                        "band": band_name,
                        "datetime": now_string(),
                    })

                del models

                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

    if len(all_training_jobs) > 0:
        train_job_csv = OUTPUT_DIR / "sr_logs" / "train_only_jobs_completed.csv"
        write_dict_csv(train_job_csv, all_training_jobs)

    merge_train_history_files()
    plot_training_loss_curves_from_history()

    total_time_sec = time.perf_counter() - total_start_time

    total_summary = {
        "RUN_NAME": RUN_NAME,
        "OUTPUT_DIR": str(OUTPUT_DIR),
        "SCALES": SCALES,
        "MODES": EXPERIMENT_MODES,
        "METHODS": METHODS_TO_RUN,
        "TRAIN_ONLY_NO_EVAL": True,
        "TOTAL_TIME_SEC": total_time_sec,
        "TOTAL_TIME_HMS": format_seconds(total_time_sec),
        "END_TIME": now_string(),
    }

    total_summary_path = OUTPUT_DIR / "sr_logs" / "train_only_total_summary.json"
    with open(total_summary_path, "w", encoding="utf-8") as f:
        json.dump(total_summary, f, ensure_ascii=False, indent=4)

    print("\n====================================")
    print("Training-only pipeline finished")
    print("====================================")
    print(f"Total time : {format_seconds(total_time_sec)}")
    print(f"Output     : {OUTPUT_DIR}")
    print(f"Loss plots : {OUTPUT_DIR / 'sr_logs' / 'train_loss_plots'}")
    print("Evaluation : skipped")
    print("====================================")


if __name__ == "__main__":
    main()
