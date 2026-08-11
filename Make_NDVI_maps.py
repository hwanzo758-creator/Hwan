# Make_Figures_9methods.py
# ------------------------------------------------------------
# Manuscript Figures 3-7 for NINE SR methods (DRCT included).
#   bicubic | SRCNN | EDSR | RCAN | SwinIR | ESRGAN | HAT | DAT | DRCT
#
# Changed vs Make_NDVI_hat_dat.py:
#   - DRCT (CVPRW 2024) model definition added, copied verbatim from
#     Final_version6_drct.py so trained checkpoints load with strict=True
#   - checkpoint auto-discovery: searches every */sr_checkpoints folder under
#     the SR_RESULTS root, so DRCT weights are found whatever the run name was
#   - column-label lists fixed (they were one entry short of the method lists)
#   - every path overridable by environment variable; no need to edit the file
#
# Environment overrides (all optional):
#   SR_DATASET_DIR  SR_OUTPUT_ROOT  SR_FIG_OUTPUT_DIR  SR_HR_PATH
#   SR_CKPT_DIRS (colon-separated)  SR_FIG_METHODS  SR_SCALES
# ------------------------------------------------------------
#
# (original header follows)
# Make_NDVI_bandwise_fixed.py
# ------------------------------------------------------------
# 목적:
# - 특정 HR npy 1개(scene_012483_1배추_2전라도_2_HR.npy)에 대해
# - x2 / x3 / x4
# - joint5ch / bandwise
# - srcnn / edsr / rcan / swinir / esrgan
# 체크포인트를 사용하여 SR 수행
# - NDVI / GNDVI / NDRE 이미지 및 panel 저장
#
# 입력 HR:
#   /home/whanjo/datasets/Python_file/SR_HR_LR_TEST/light_cabbage_training_dataset/HR_npy/
#   scene_012483_1배추_2전라도_2_HR.npy
#
# 체크포인트:
#   /home/whanjo/datasets/Python_file/SR_HR_LR_TEST/light_cabbage_training_dataset_SR_RESULTS/
#   final_x234_joint5ch_bandwise_6models_v19_resume_amp_earlystop_rank/sr_checkpoints
#
# 특징:
# 1) 기존 LR_x2/x3/x4 npy가 있으면 우선 사용
# 2) 없으면 HR에서 bicubic downsample로 LR 자동 생성
# 3) x3는 800x800 -> 798x798처럼 자연 정렬 crop이 발생할 수 있음
# 4) checkpoint가 없으면 summary CSV에 checkpoint_missing으로 기록
# ------------------------------------------------------------

from pathlib import Path
import csv
import os
import warnings

import cv2
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================
# 1. 사용자 설정
# ============================================================

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

DATASET_DIR = Path(os.environ.get(
    "SR_DATASET_DIR",
    "/home/whanjo/datasets/Python_file/SR_HR_LR_TEST/light_cabbage_training_dataset",
))

SR_RESULTS_ROOT = Path(os.environ.get(
    "SR_OUTPUT_ROOT",
    "/home/whanjo/datasets/Python_file/SR_HR_LR_TEST/"
    "light_cabbage_training_dataset_SR_RESULTS",
))

HR_PATH = Path(os.environ["SR_HR_PATH"]) if os.environ.get("SR_HR_PATH") else (
    DATASET_DIR / "HR_npy" / "scene_012483_1배추_2전라도_2_HR.npy"
)

CHECKPOINT_DIR = Path(
    "/home/whanjo/datasets/Python_file/SR_HR_LR_TEST/"
    "light_cabbage_training_dataset_SR_RESULTS/"
    "final_x234_joint5ch_bandwise_6models_v19_resume_amp_earlystop_rank/"
    "sr_checkpoints"
)

# HAT/DAT checkpoints live in CHECKPOINT_DIR (MAIN run); the base six models
# (srcnn/edsr/rcan/swinir/esrgan) were reused by the MAIN run from this fallback
# folder, so we look there too. This matches exactly the checkpoints the MAIN
# results tables were computed from.
CHECKPOINT_DIR_FALLBACK = Path(
    "/home/whanjo/datasets/Python_file/SR_HR_LR_TEST/"
    "light_cabbage_training_dataset_SR_RESULTS/"
    "final_train_only_all_models_retrain_ignore_existing_ckpt/"
    "sr_checkpoints"
)

# REVISION (B6): DRCT was trained in a separate run whose folder name this
# script does not know, so extra sr_checkpoints directories may be listed via
# SR_CKPT_DIRS and, failing that, the whole SR_RESULTS tree is indexed once at
# start-up. The script then works regardless of the SR_RUN_NAME used for DRCT.
EXTRA_CHECKPOINT_DIRS = [
    Path(p) for p in os.environ.get("SR_CKPT_DIRS", "").split(":") if p.strip()
]
AUTO_DISCOVER_CHECKPOINTS = os.environ.get("SR_CKPT_AUTODISCOVER", "1") != "0"

OUTPUT_DIR = Path(os.environ.get(
    "SR_FIG_OUTPUT_DIR",
    "/home/whanjo/datasets/Python_file/SR_HR_LR_TEST/"
    "single_scene_vi_outputs_scene_012483_ndvi_gndvi_ndre",
))

SCALES = [int(v) for v in os.environ.get("SR_SCALES", "2,3,4").split(",") if v.strip()]

# REVISION (B6): DRCT (CVPRW 2024) added as the ninth method.
ALL_METHODS = ["srcnn", "edsr", "rcan", "swinir", "esrgan", "hat", "dat", "drct"]
METHODS = ([m.strip() for m in os.environ["SR_FIG_METHODS"].split(",") if m.strip()]
           if os.environ.get("SR_FIG_METHODS") else list(ALL_METHODS))

# Display labels used by every figure grid. Keeping one dictionary here removes
# the column-label / method-list mismatch that existed in the previous version.
METHOD_LABEL = {
    "hr": "HR", "bicubic": "Bicubic", "srcnn": "SRCNN", "edsr": "EDSR",
    "rcan": "RCAN", "swinir": "SwinIR", "esrgan": "ESRGAN",
    "hat": "HAT", "dat": "DAT", "drct": "DRCT",
}

# Method order used in the overview grids (bicubic first, learned models after).
GRID_METHODS = ["bicubic"] + list(METHODS)
MODES = ["joint5ch", "bandwise"]
INCLUDE_BICUBIC_BASELINE = True

BAND_NAMES = ["Blue", "Green", "Red", "RedEdge", "NIR"]

# bandwise checkpoint 파일명은 B/G/R/E/N 코드로 저장되어 있음
# 예:
#   bandwise_edsr_B_x2_best.pth
#   bandwise_edsr_G_x2_best.pth
#   bandwise_edsr_R_x2_best.pth
#   bandwise_edsr_E_x2_best.pth
#   bandwise_edsr_N_x2_best.pth
BAND_CODES = ["B", "G", "R", "E", "N"]

CHANNELS = 5
NODATA_THRESHOLD = -9999.0

USE_EXISTING_LR_IF_AVAILABLE = True
SAVE_SR_NPY = True

# REVISION (B6): reuse SR cubes that a previous run already saved, so adding one
# method costs one method's inference instead of re-running all nine. The panels
# and summary rows are still rebuilt for every method, which is what the Fig 6/7
# overview grids need - only the GPU work is skipped.
#   SR_REUSE_SAVED=0        recompute everything
#   SR_FORCE_METHODS=drct   recompute just these, reuse the rest
REUSE_SAVED_SR = os.environ.get("SR_REUSE_SAVED", "1") != "0"
FORCE_INFER_METHODS = {
    m.strip() for m in os.environ.get("SR_FORCE_METHODS", "").split(",") if m.strip()
}
_REUSE_STATS = {"reused": 0, "inferred": 0}
SAVE_LR_NPY_IF_GENERATED = True

# Inference tile
DEFAULT_TILE_LR_SIZE = 256
DEFAULT_TILE_OVERLAP = 16
SWINIR_TILE_LR_SIZE = 64
SWINIR_TILE_OVERLAP = 8
HAT_TILE_LR_SIZE = 64
HAT_TILE_OVERLAP = 8

# NDVI / GNDVI / NDRE visualization
NDVI_VIS_MIN = -1.0
NDVI_VIS_MAX = 1.0
GNDVI_VIS_MIN = -1.0
GNDVI_VIS_MAX = 1.0
NDRE_VIS_MIN = -1.0
NDRE_VIS_MAX = 1.0
ERROR_MAP_PERCENTILE = 99.0
# Fixed shared color-scale max for the error-map overview grids, matching the
# original manuscript (Figures 6-7 used 0 to 0.114). Set to None to fall back to
# the dynamic per-grid 99th-percentile vmax. A fixed value keeps the joint and
# band-wise grids on the same, informative scale and prevents one outlier method
# (e.g. ESRGAN x2) from washing out all other cells.
ERROR_MAP_FIXED_VMAX = 0.114

# 기존 학습 코드의 residual SR 설정과 일치
USE_RESIDUAL_SR = True


# ============================================================
# 2. 기본 유틸
# ============================================================

def ensure_dir(path):
    Path(path).mkdir(parents=True, exist_ok=True)


def make_valid_mask_np(cube):
    return np.isfinite(cube) & (cube > NODATA_THRESHOLD)


def clean_nodata_to_zero(cube, valid_mask=None):
    cube = np.asarray(cube, dtype=np.float32).copy()
    if valid_mask is None:
        valid_mask = make_valid_mask_np(cube)
    cube[~valid_mask] = 0.0
    cube = np.nan_to_num(cube, nan=0.0, posinf=1.5, neginf=0.0)
    return cube.astype(np.float32)


def normalize_cube(cube, norm_factor=1.0, valid_mask=None):
    # 기존 코드와 동일하게 입력 npy가 이미 normalized reflectance-scale이라고 가정
    return clean_nodata_to_zero(cube, valid_mask)


def denormalize_cube(cube, norm_factor=1.0):
    return np.nan_to_num(cube.astype(np.float32), nan=0.0, posinf=1.5, neginf=0.0)


def np_to_tensor(cube):
    return torch.from_numpy(cube.transpose(2, 0, 1)).float()


def tensor_to_np(tensor):
    arr = tensor.detach().cpu().float().numpy()
    return arr.transpose(1, 2, 0)


def crop_common_np(a, b):
    h = min(a.shape[0], b.shape[0])
    w = min(a.shape[1], b.shape[1])
    c = min(a.shape[2], b.shape[2])
    return a[:h, :w, :c], b[:h, :w, :c]


def center_crop_to_shape(cube, target_h, target_w):
    cube = np.asarray(cube)
    h, w = cube.shape[:2]
    target_h = int(target_h)
    target_w = int(target_w)

    if target_h <= 0 or target_w <= 0:
        raise ValueError(f"Invalid target crop size: ({target_h}, {target_w})")

    if target_h > h or target_w > w:
        raise ValueError(
            f"Target crop larger than input: input={cube.shape}, target=({target_h}, {target_w})"
        )

    y1 = (h - target_h) // 2
    x1 = (w - target_w) // 2
    y2 = y1 + target_h
    x2 = x1 + target_w
    return cube[y1:y2, x1:x2, :]


def align_hr_to_lr_scale(hr, lr, scale):
    target_h = int(lr.shape[0] * scale)
    target_w = int(lr.shape[1] * scale)

    if hr.shape[0] == target_h and hr.shape[1] == target_w:
        return hr

    if hr.shape[0] >= target_h and hr.shape[1] >= target_w:
        return center_crop_to_shape(hr, target_h, target_w)

    common_h = min(hr.shape[0], target_h)
    common_w = min(hr.shape[1], target_w)
    return center_crop_to_shape(hr, common_h, common_w)


def resize_cube(cube, out_h, out_w, interpolation=cv2.INTER_CUBIC):
    out = cv2.resize(cube.astype(np.float32), (int(out_w), int(out_h)), interpolation=interpolation)
    if out.ndim == 2:
        out = out[..., None]
    return out.astype(np.float32)


def get_ref_vmin_vmax(ref_img, p_low=2, p_high=98):
    ref_img = ref_img.astype(np.float32)
    valid = np.isfinite(ref_img) & (ref_img > NODATA_THRESHOLD)

    if valid.sum() == 0:
        return 0.0, 1.0

    vmin = float(np.percentile(ref_img[valid], p_low))
    vmax = float(np.percentile(ref_img[valid], p_high))

    if not np.isfinite(vmin) or not np.isfinite(vmax) or vmax <= vmin:
        vmin = float(np.nanmin(ref_img[valid]))
        vmax = float(np.nanmax(ref_img[valid]))

    if not np.isfinite(vmin) or not np.isfinite(vmax) or vmax <= vmin:
        return 0.0, 1.0

    return vmin, vmax


def normalize_to_uint8(img, vmin=None, vmax=None):
    img = img.astype(np.float32)

    if vmin is None or vmax is None:
        vmin, vmax = get_ref_vmin_vmax(img)

    if not np.isfinite(vmin) or not np.isfinite(vmax) or vmax <= vmin:
        return np.zeros(img.shape, dtype=np.uint8)

    out = (img - vmin) / (vmax - vmin)
    out = np.clip(out, 0, 1)
    return (out * 255).astype(np.uint8)


def add_label_rgb(img, text):
    out = img.copy()
    cv2.putText(
        out,
        text,
        (10, 24),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.75,
        (255, 255, 0),
        2,
        cv2.LINE_AA,
    )
    return out


def diff_to_heatmap(diff, vmin=0.0, vmax=None):
    valid = np.isfinite(diff)

    if vmax is None:
        vmax = float(np.percentile(diff[valid], ERROR_MAP_PERCENTILE)) if np.any(valid) else 1.0

    if (not np.isfinite(vmax)) or vmax <= vmin:
        vmax = vmin + 1e-6

    diff_u8 = normalize_to_uint8(np.nan_to_num(diff, nan=vmin), vmin=vmin, vmax=vmax)
    heat = cv2.applyColorMap(diff_u8, cv2.COLORMAP_JET)
    return cv2.cvtColor(heat, cv2.COLOR_BGR2RGB), vmax


def _format_colorbar_value(value):
    if not np.isfinite(value):
        return "nan"
    value = float(value)
    if abs(value) >= 1.0:
        return f"{value:.2f}"
    if abs(value) >= 0.1:
        return f"{value:.3f}"
    return f"{value:.4f}"


def append_vertical_colorbar_rgb(
    img_rgb,
    vmin,
    vmax,
    colormap=cv2.COLORMAP_JET,
    colorbar_width=78,
    bar_width=18,
):
    """Append a vertical color bar to the right side of an RGB heatmap."""
    img_rgb = np.asarray(img_rgb, dtype=np.uint8)
    h = int(img_rgb.shape[0])

    if (not np.isfinite(vmin)) or (not np.isfinite(vmax)) or vmax <= vmin:
        vmax = vmin + 1e-6

    panel = np.full((h, colorbar_width, 3), 255, dtype=np.uint8)
    gradient = np.linspace(vmax, vmin, h, dtype=np.float32)[:, None]
    gradient_u8 = normalize_to_uint8(gradient, vmin=vmin, vmax=vmax)
    bar_bgr = cv2.applyColorMap(gradient_u8, colormap)
    bar_bgr = cv2.resize(bar_bgr, (bar_width, h), interpolation=cv2.INTER_NEAREST)
    bar_rgb = cv2.cvtColor(bar_bgr, cv2.COLOR_BGR2RGB)

    x0 = 8
    panel[:, x0:x0 + bar_width, :] = bar_rgb
    cv2.rectangle(panel, (x0, 0), (x0 + bar_width - 1, h - 1), (0, 0, 0), 1)

    tick_values = [float(vmax), float((vmin + vmax) / 2.0), float(vmin)]
    tick_ys = [0, h // 2, h - 1]
    text_x = x0 + bar_width + 5

    for value, y in zip(tick_values, tick_ys):
        y = int(y)
        cv2.line(panel, (x0 + bar_width, y), (x0 + bar_width + 4, y), (0, 0, 0), 1)
        text_y = min(max(y + 5, 12), h - 5)
        cv2.putText(
            panel,
            _format_colorbar_value(value),
            (text_x, text_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.34,
            (0, 0, 0),
            1,
            cv2.LINE_AA,
        )

    return np.concatenate([img_rgb, panel], axis=1)


def compute_index_map(cube, index_name, eps=1e-6):
    """
    Compute vegetation index map from 5-band cube.

    Band order:
        0 = Blue
        1 = Green
        2 = Red
        3 = RedEdge
        4 = NIR

    Supported indices:
        NDVI  = (NIR - Red) / (NIR + Red)
        GNDVI = (NIR - Green) / (NIR + Green)
        NDRE  = (NIR - RedEdge) / (NIR + RedEdge)
    """
    cube = np.asarray(cube, dtype=np.float32)

    if cube.ndim != 3 or cube.shape[2] < 5:
        return None

    green = cube[..., 1]
    red = cube[..., 2]
    rededge = cube[..., 3]
    nir = cube[..., 4]

    index_name = str(index_name).lower()

    if index_name == "ndvi":
        denom = np.maximum(nir + red, eps)
        return (nir - red) / denom

    if index_name == "gndvi":
        denom = np.maximum(nir + green, eps)
        return (nir - green) / denom

    if index_name == "ndre":
        denom = np.maximum(nir + rededge, eps)
        return (nir - rededge) / denom

    return None


def index_to_colormap(index_map, vmin=-1.0, vmax=1.0, colormap=cv2.COLORMAP_TURBO):
    index_map = np.asarray(index_map, dtype=np.float32)
    valid = np.isfinite(index_map)

    if not np.any(valid):
        index_u8 = np.zeros(index_map.shape, dtype=np.uint8)
    else:
        clipped = np.clip(index_map, vmin, vmax)
        clipped = np.nan_to_num(clipped, nan=vmin, posinf=vmax, neginf=vmin)
        index_u8 = normalize_to_uint8(clipped, vmin=vmin, vmax=vmax)

    color_bgr = cv2.applyColorMap(index_u8, colormap)
    color_rgb = cv2.cvtColor(color_bgr, cv2.COLOR_BGR2RGB)
    color_rgb[~valid] = 0

    return color_rgb


def mask_nodata_for_metrics(hr, sr):
    hr, sr = crop_common_np(hr, sr)
    valid = make_valid_mask_np(hr) & np.isfinite(sr)

    hr_m = hr.astype(np.float32).copy()
    sr_m = sr.astype(np.float32).copy()

    hr_m[~valid] = np.nan
    sr_m[~valid] = np.nan

    valid_ratio = float(valid.sum() / max(valid.size, 1))
    return hr_m, sr_m, valid, valid_ratio


def calc_ndvi_mae(hr, sr):
    hr_m, sr_m, _, _ = mask_nodata_for_metrics(hr, sr)
    hr_ndvi = compute_index_map(hr_m, "ndvi")
    sr_ndvi = compute_index_map(sr_m, "ndvi")

    if hr_ndvi is None or sr_ndvi is None:
        return np.nan

    return float(np.nanmean(np.abs(hr_ndvi - sr_ndvi)))


def save_rgb_png(path, img_rgb):
    cv2.imwrite(str(path), cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR))


def band_to_gray_rgb(band, vmin=None, vmax=None):
    band = np.asarray(band, dtype=np.float32)
    valid = np.isfinite(band) & (band > NODATA_THRESHOLD)
    band_clean = np.nan_to_num(band, nan=0.0, posinf=1.5, neginf=0.0)

    gray = normalize_to_uint8(band_clean, vmin=vmin, vmax=vmax)
    rgb = np.repeat(gray[..., None], 3, axis=2)
    rgb[~valid] = 0
    return rgb


def save_band_error_outputs(base_out_dir, scene_id, scale, mode, method, hr, sr):
    """
    Save per-band HR/SR/error maps and three-panel comparison images.

    Outputs for each band:
      - HR_<band>.png
      - SR_<band>.png
      - <band>_error.png
      - <band>_panel.png
      - <band>_MAE in the summary CSV
    """
    ensure_dir(base_out_dir)

    hr, sr = crop_common_np(hr, sr)
    result = {}

    for bi, band_name in enumerate(BAND_NAMES):
        key = band_name.replace(" ", "")
        result.update({
            f"HR_{key}_path": "",
            f"SR_{key}_path": "",
            f"{key}_error_path": "",
            f"{key}_error_npy_path": "",
            f"{key}_panel_path": "",
            f"{key}_error_vmax": np.nan,
            f"{key}_MAE": np.nan,
        })

        if bi >= hr.shape[2] or bi >= sr.shape[2]:
            continue

        hr_band = hr[..., bi].astype(np.float32)
        sr_band = sr[..., bi].astype(np.float32)
        valid = np.isfinite(hr_band) & np.isfinite(sr_band) & (hr_band > NODATA_THRESHOLD)

        if not np.any(valid):
            continue

        vmin, vmax = get_ref_vmin_vmax(hr_band)
        err = np.full_like(hr_band, np.nan, dtype=np.float32)
        err[valid] = np.abs(hr_band[valid] - sr_band[valid])

        hr_img = band_to_gray_rgb(hr_band, vmin=vmin, vmax=vmax)
        sr_img = band_to_gray_rgb(sr_band, vmin=vmin, vmax=vmax)
        err_img, err_vmax = diff_to_heatmap(err, vmin=0.0, vmax=None)
        err_img[~valid] = 0
        err_img = append_vertical_colorbar_rgb(err_img, 0.0, err_vmax)

        hr_img = add_label_rgb(hr_img, f"HR {band_name}")
        sr_img = add_label_rgb(sr_img, f"SR {band_name}")
        err_img = add_label_rgb(err_img, f"{band_name} error")

        hr_path = base_out_dir / f"{scene_id}_x{scale}_{mode}_{method}_HR_{key}.png"
        sr_path = base_out_dir / f"{scene_id}_x{scale}_{mode}_{method}_SR_{key}.png"
        err_path = base_out_dir / f"{scene_id}_x{scale}_{mode}_{method}_{key}_error.png"
        err_npy_path = base_out_dir / f"{scene_id}_x{scale}_{mode}_{method}_{key}_error.npy"
        panel_path = base_out_dir / f"{scene_id}_x{scale}_{mode}_{method}_{key}_panel.png"

        save_rgb_png(hr_path, hr_img)
        save_rgb_png(sr_path, sr_img)
        save_rgb_png(err_path, err_img)
        np.save(err_npy_path, err.astype(np.float32))

        panel = np.concatenate([hr_img, sr_img, err_img], axis=1)
        save_rgb_png(panel_path, panel)

        result.update({
            f"HR_{key}_path": str(hr_path),
            f"SR_{key}_path": str(sr_path),
            f"{key}_error_path": str(err_path),
            f"{key}_error_npy_path": str(err_npy_path),
            f"{key}_panel_path": str(panel_path),
            f"{key}_error_vmax": float(err_vmax),
            f"{key}_MAE": float(np.nanmean(err)),
        })

    return result


# ============================================================
# 3. LR 탐색 / 생성
# ============================================================

def try_find_existing_lr_path(hr_path, scale):
    hr_path = Path(hr_path)
    candidates = []

    if "HR_npy" in hr_path.parts:
        parts = list(hr_path.parts)
        idx = parts.index("HR_npy")
        parts[idx] = f"LR_x{scale}_npy"

        # case 1: scene_xxx_HR.npy -> scene_xxx_LR_x2.npy
        stem1 = hr_path.stem.replace("_HR", f"_LR_x{scale}")
        candidates.append(Path(*parts).with_name(stem1 + ".npy"))

        # case 2: same stem in LR folder
        candidates.append(Path(*parts).with_name(hr_path.name))

    # fallback
    candidates.append(
        hr_path.parent.parent / f"LR_x{scale}_npy" / hr_path.name.replace("_HR.npy", f"_LR_x{scale}.npy")
    )
    candidates.append(
        hr_path.parent.parent / f"LR_x{scale}_npy" / hr_path.name
    )

    for p in candidates:
        if p.exists():
            return p

    return None


def load_or_make_lr(hr_raw, hr_path, scale, out_dir):
    """
    1) 기존 LR 파일이 있으면 사용
    2) 없으면 HR에서 생성
    """
    if USE_EXISTING_LR_IF_AVAILABLE:
        lr_path = try_find_existing_lr_path(hr_path, scale)
        if lr_path is not None and lr_path.exists():
            lr = np.load(lr_path).astype(np.float32)
            hr_aligned = align_hr_to_lr_scale(hr_raw, lr, scale)
            return lr, hr_aligned, str(lr_path), False

    # LR가 없으면 HR에서 downsample 생성
    h, w = hr_raw.shape[:2]
    target_h = (h // scale) * scale
    target_w = (w // scale) * scale
    hr_aligned = center_crop_to_shape(hr_raw, target_h, target_w)

    lr_h = target_h // scale
    lr_w = target_w // scale
    lr = resize_cube(hr_aligned, lr_h, lr_w, interpolation=cv2.INTER_CUBIC)

    generated_lr_path = out_dir / f"{Path(hr_path).stem}_generated_LR_x{scale}.npy"
    if SAVE_LR_NPY_IF_GENERATED:
        np.save(generated_lr_path, lr.astype(np.float32))

    return lr, hr_aligned, str(generated_lr_path), True


# ============================================================
# 4. 모델 정의
# ============================================================

def make_activation(name):
    if name == "relu":
        return nn.ReLU(inplace=True)
    if name == "gelu":
        return nn.GELU()
    if name == "lrelu":
        return nn.LeakyReLU(0.2, inplace=True)
    raise ValueError(f"Unsupported activation: {name}")


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
        self.net = nn.Sequential(
            nn.Conv2d(channels, 64, kernel_size=9, padding=4),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 32, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, channels, kernel_size=5, padding=2),
        )

    def forward(self, x):
        base = F.interpolate(x, scale_factor=self.scale, mode="bicubic", align_corners=False)
        detail = self.net(base)
        out = base + detail if USE_RESIDUAL_SR else detail
        return torch.clamp(out, 0.0, 1.5)


class ResidualBlock(nn.Module):
    def __init__(self, nf=64):
        super().__init__()
        self.conv1 = nn.Conv2d(nf, nf, 3, padding=1)
        self.conv2 = nn.Conv2d(nf, nf, 3, padding=1)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        res = self.relu(self.conv1(x))
        res = self.conv2(res)
        return x + res * 0.1


class EDSR(nn.Module):
    def __init__(self, scale, channels=5, nf=64, n_blocks=8):
        super().__init__()
        self.scale = scale
        self.head = nn.Conv2d(channels, nf, 3, padding=1)
        self.body = nn.Sequential(*[ResidualBlock(nf) for _ in range(n_blocks)])
        self.body_conv = nn.Conv2d(nf, nf, 3, padding=1)
        self.up = make_upsampler(scale, nf, activation="relu")
        self.tail = nn.Conv2d(nf, channels, 3, padding=1)

    def forward(self, x):
        base = F.interpolate(x, scale_factor=self.scale, mode="bicubic", align_corners=False)
        feat = self.head(x)
        res = self.body(feat)
        res = self.body_conv(res)
        feat = feat + res
        feat = self.up(feat)
        detail = self.tail(feat)
        out = base + detail if USE_RESIDUAL_SR else detail
        return torch.clamp(out, 0.0, 1.5)


class ChannelAttention(nn.Module):
    def __init__(self, nf=64, reduction=16):
        super().__init__()
        hidden = max(nf // reduction, 4)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.ca = nn.Sequential(
            nn.Conv2d(nf, hidden, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, nf, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        weight = self.ca(self.pool(x))
        return x * weight


class RCAB(nn.Module):
    def __init__(self, nf=64):
        super().__init__()
        self.conv1 = nn.Conv2d(nf, nf, 3, padding=1)
        self.conv2 = nn.Conv2d(nf, nf, 3, padding=1)
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
        self.blocks = nn.Sequential(*[RCAB(nf) for _ in range(n_rcab)])
        self.conv = nn.Conv2d(nf, nf, 3, padding=1)

    def forward(self, x):
        res = self.blocks(x)
        res = self.conv(res)
        return x + res


class RCAN(nn.Module):
    def __init__(self, scale, channels=5, nf=64, n_groups=4, n_rcab=4):
        super().__init__()
        self.scale = scale
        self.head = nn.Conv2d(channels, nf, 3, padding=1)
        self.body = nn.Sequential(*[ResidualGroup(nf, n_rcab) for _ in range(n_groups)])
        self.body_conv = nn.Conv2d(nf, nf, 3, padding=1)
        self.up = make_upsampler(scale, nf, activation="relu")
        self.tail = nn.Conv2d(nf, channels, 3, padding=1)

    def forward(self, x):
        base = F.interpolate(x, scale_factor=self.scale, mode="bicubic", align_corners=False)
        feat = self.head(x)
        res = self.body(feat)
        res = self.body_conv(res)
        feat = feat + res
        feat = self.up(feat)
        detail = self.tail(feat)
        out = base + detail if USE_RESIDUAL_SR else detail
        return torch.clamp(out, 0.0, 1.5)


class SwinLikeBlock(nn.Module):
    def __init__(self, dim=64, heads=4, mlp_ratio=2.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(embed_dim=dim, num_heads=heads, batch_first=True)
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
        self.head = nn.Conv2d(channels, dim, 3, padding=1)
        self.blocks = nn.Sequential(*[SwinLikeBlock(dim=dim, heads=heads) for _ in range(depth)])
        self.body_conv = nn.Conv2d(dim, dim, 3, padding=1)
        self.up = make_upsampler(scale, dim, activation="gelu")
        self.tail = nn.Conv2d(dim, channels, 3, padding=1)

    def forward(self, x):
        base = F.interpolate(x, scale_factor=self.scale, mode="bicubic", align_corners=False)
        feat = self.head(x)
        res = self.blocks(feat)
        res = self.body_conv(res)
        feat = feat + res
        feat = self.up(feat)
        detail = self.tail(feat)
        out = base + detail if USE_RESIDUAL_SR else detail
        return torch.clamp(out, 0.0, 1.5)


class DenseResidualBlock(nn.Module):
    def __init__(self, nf=64, gc=32):
        super().__init__()
        self.conv1 = nn.Conv2d(nf, gc, 3, padding=1)
        self.conv2 = nn.Conv2d(nf + gc, gc, 3, padding=1)
        self.conv3 = nn.Conv2d(nf + gc * 2, gc, 3, padding=1)
        self.conv4 = nn.Conv2d(nf + gc * 3, gc, 3, padding=1)
        self.conv5 = nn.Conv2d(nf + gc * 4, nf, 3, padding=1)
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
        self.conv_first = nn.Conv2d(channels, nf, 3, padding=1)
        self.trunk = nn.Sequential(*[RRDB(nf) for _ in range(n_rrdb)])
        self.trunk_conv = nn.Conv2d(nf, nf, 3, padding=1)
        self.up = make_upsampler(scale, nf, activation="lrelu")
        self.conv_hr = nn.Conv2d(nf, nf, 3, padding=1)
        self.conv_last = nn.Conv2d(nf, channels, 3, padding=1)
        self.lrelu = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x):
        fea = self.conv_first(x)
        trunk = self.trunk(fea)
        trunk = self.trunk_conv(trunk)
        fea = fea + trunk
        out = self.up(fea)
        out = self.lrelu(self.conv_hr(out))
        out = self.conv_last(out)
        out = torch.clamp(out, 0.0, 1.5)
        return out


# ============================================================
# HAT / DAT (recent SOTA baselines) - identical architectures to the training
# script so checkpoints load with strict=True.
# ============================================================
class HABlock(nn.Module):
    def __init__(self, dim=96, heads=4, mlp_ratio=2.0, ca_weight=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(embed_dim=dim, num_heads=heads, batch_first=True)
        self.cab = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(dim, dim, kernel_size=3, padding=1),
            ChannelAttention(dim),
        )
        self.ca_weight = ca_weight
        self.norm2 = nn.LayerNorm(dim)
        hidden = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(nn.Linear(dim, hidden), nn.GELU(), nn.Linear(hidden, dim))

    def forward(self, x):
        b, c, h, w = x.shape
        tokens = x.flatten(2).transpose(1, 2)
        y = self.norm1(tokens)
        attn_out, _ = self.attn(y, y, y, need_weights=False)
        cab_out = self.cab(x).flatten(2).transpose(1, 2)
        tokens = tokens + attn_out + self.ca_weight * cab_out
        y = self.norm2(tokens)
        tokens = tokens + self.mlp(y)
        return tokens.transpose(1, 2).reshape(b, c, h, w)


class HybridAttentionGroup(nn.Module):
    def __init__(self, dim=96, heads=4, depth=4):
        super().__init__()
        self.blocks = nn.Sequential(*[HABlock(dim=dim, heads=heads) for _ in range(depth)])
        self.conv = nn.Conv2d(dim, dim, kernel_size=3, padding=1)

    def forward(self, x):
        res = self.blocks(x)
        res = self.conv(res)
        return x + res


class HAT(nn.Module):
    def __init__(self, scale, channels=5, dim=96, n_groups=4, depth=4, heads=4):
        super().__init__()
        self.scale = scale
        self.head = nn.Conv2d(channels, dim, kernel_size=3, padding=1)
        self.body = nn.Sequential(*[HybridAttentionGroup(dim=dim, heads=heads, depth=depth) for _ in range(n_groups)])
        self.body_conv = nn.Conv2d(dim, dim, kernel_size=3, padding=1)
        self.up = make_upsampler(scale, dim, activation="gelu")
        self.tail = nn.Conv2d(dim, channels, kernel_size=3, padding=1)

    def forward(self, x):
        base = F.interpolate(x, scale_factor=self.scale, mode="bicubic", align_corners=False)
        feat = self.head(x)
        res = self.body(feat)
        res = self.body_conv(res)
        feat = feat + res
        feat = self.up(feat)
        detail = self.tail(feat)
        out = base + detail if USE_RESIDUAL_SR else detail
        return torch.clamp(out, 0.0, 1.5)


class ChannelSelfAttention(nn.Module):
    def __init__(self, dim=96, heads=4):
        super().__init__()
        self.heads = heads
        self.temperature = nn.Parameter(torch.ones(heads, 1, 1))
        self.qkv = nn.Conv2d(dim, dim * 3, kernel_size=1)
        self.qkv_dw = nn.Conv2d(dim * 3, dim * 3, kernel_size=3, padding=1, groups=dim * 3)
        self.proj = nn.Conv2d(dim, dim, kernel_size=1)

    def forward(self, x):
        b, c, h, w = x.shape
        qkv = self.qkv_dw(self.qkv(x))
        q, k, v = qkv.chunk(3, dim=1)
        hd = c // self.heads
        q = q.reshape(b, self.heads, hd, h * w)
        k = k.reshape(b, self.heads, hd, h * w)
        v = v.reshape(b, self.heads, hd, h * w)
        q = F.normalize(q, dim=-1)
        k = F.normalize(k, dim=-1)
        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)
        out = attn @ v
        out = out.reshape(b, c, h, w)
        return self.proj(out)


class DATBlock(nn.Module):
    def __init__(self, dim=96, heads=4, mlp_ratio=2.0):
        super().__init__()
        self.norm_s = nn.LayerNorm(dim)
        self.spatial_attn = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.channel_attn = ChannelSelfAttention(dim, heads)
        self.norm_m = nn.LayerNorm(dim)
        hidden = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(nn.Linear(dim, hidden), nn.GELU(), nn.Linear(hidden, dim))

    def forward(self, x):
        b, c, h, w = x.shape
        tokens = x.flatten(2).transpose(1, 2)
        y = self.norm_s(tokens)
        attn_out, _ = self.spatial_attn(y, y, y, need_weights=False)
        tokens = tokens + attn_out
        x2 = tokens.transpose(1, 2).reshape(b, c, h, w)
        x2 = x2 + self.channel_attn(x2)
        tokens = x2.flatten(2).transpose(1, 2)
        tokens = tokens + self.mlp(self.norm_m(tokens))
        return tokens.transpose(1, 2).reshape(b, c, h, w)


class DualAggregationGroup(nn.Module):
    def __init__(self, dim=96, heads=4, depth=4):
        super().__init__()
        self.blocks = nn.Sequential(*[DATBlock(dim=dim, heads=heads) for _ in range(depth)])
        self.conv = nn.Conv2d(dim, dim, kernel_size=3, padding=1)

    def forward(self, x):
        res = self.blocks(x)
        res = self.conv(res)
        return x + res


class DAT(nn.Module):
    def __init__(self, scale, channels=5, dim=96, n_groups=4, depth=4, heads=4):
        super().__init__()
        self.scale = scale
        self.head = nn.Conv2d(channels, dim, kernel_size=3, padding=1)
        self.body = nn.Sequential(*[DualAggregationGroup(dim=dim, heads=heads, depth=depth) for _ in range(n_groups)])
        self.body_conv = nn.Conv2d(dim, dim, kernel_size=3, padding=1)
        self.up = make_upsampler(scale, dim, activation="gelu")
        self.tail = nn.Conv2d(dim, channels, kernel_size=3, padding=1)

    def forward(self, x):
        base = F.interpolate(x, scale_factor=self.scale, mode="bicubic", align_corners=False)
        feat = self.head(x)
        res = self.body(feat)
        res = self.body_conv(res)
        feat = feat + res
        feat = self.up(feat)
        detail = self.tail(feat)
        out = base + detail if USE_RESIDUAL_SR else detail
        return torch.clamp(out, 0.0, 1.5)


# ============================================================
# REVISION (B6): DRCT - Dense-Residual-Connected Transformer
# (Hsu et al., "DRCT: Saving Image Super-Resolution away from Information
#  Bottleneck", CVPRW 2024, NTIRE workshop)
#
# Compact reimplementation matched to the existing training budget and written
# in the same residual-SR template as the other models in this script.
#
# DRCT starts from the SwinIR/HAT design and addresses a specific failure mode:
# in a plain sequential stack of transformer layers the intensity range of the
# feature maps widens with depth and is then squeezed at the output, so spatial
# information is lost through an information bottleneck. DRCT removes the
# bottleneck by connecting the layers densely inside each group - every layer
# receives the concatenation of the group input and all preceding layer outputs,
# projected back to a narrow working width by a 1x1 convolution - so early
# spatial detail stays reachable at the end of the group. The construction is
# the residual-dense idea of RRDB applied to attention layers rather than to
# convolutions.
#
# Deliberate simplifications, so the model stays inside the same compact and
# equalized budget as the other baselines:
#   * the shifted-window attention of the original is replaced by the same
#     global multi-head attention used by the SwinIR-based and HAT-based
#     baselines in this script, so the three share one attention implementation;
#   * the group count, layer count and growth width are reduced.
# It is therefore referred to as DRCT-based SR, exactly as the SwinIR-based,
# HAT-based and DAT-based baselines are.
# ============================================================

DRCT_DIM = int(os.environ.get("SR_DRCT_DIM", "96"))
DRCT_GROWTH = int(os.environ.get("SR_DRCT_GROWTH", "64"))   # width of each dense layer
DRCT_DEPTH = int(os.environ.get("SR_DRCT_DEPTH", "5"))      # dense layers per group
DRCT_GROUPS = int(os.environ.get("SR_DRCT_GROUPS", "6"))
DRCT_HEADS = int(os.environ.get("SR_DRCT_HEADS", "4"))
DRCT_RES_SCALE = 0.2


class DRCTGroup(nn.Module):
    """Residual dense group: attention layers joined by dense connections.

    Layer i sees cat(group input, outputs of layers 0..i-1), reduced to
    DRCT_GROWTH channels by a 1x1 convolution. The group output fuses the same
    concatenation back to the model width and is added residually, so the group
    learns a correction rather than a replacement.
    """

    def __init__(self, dim=DRCT_DIM, growth=DRCT_GROWTH, depth=DRCT_DEPTH,
                 heads=DRCT_HEADS):
        super().__init__()

        self.depth = depth
        self.reduce = nn.ModuleList()
        self.layers = nn.ModuleList()

        for i in range(depth):
            self.reduce.append(nn.Conv2d(dim + i * growth, growth, kernel_size=1))
            self.layers.append(SwinLikeBlock(dim=growth, heads=heads))

        self.fuse = nn.Conv2d(dim + depth * growth, dim, kernel_size=1)
        self.conv = nn.Conv2d(dim, dim, kernel_size=3, padding=1)

    def forward(self, x):
        feats = [x]

        for i in range(self.depth):
            y = self.reduce[i](torch.cat(feats, dim=1))
            y = self.layers[i](y)
            feats.append(y)

        out = self.fuse(torch.cat(feats, dim=1))
        out = self.conv(out)

        return x + out * DRCT_RES_SCALE


class DRCT(nn.Module):
    def __init__(self, scale, channels=5, dim=DRCT_DIM, growth=DRCT_GROWTH,
                 depth=DRCT_DEPTH, n_groups=DRCT_GROUPS, heads=DRCT_HEADS):
        super().__init__()

        self.scale = scale
        self.head = nn.Conv2d(channels, dim, kernel_size=3, padding=1)

        self.body = nn.Sequential(
            *[DRCTGroup(dim=dim, growth=growth, depth=depth, heads=heads)
              for _ in range(n_groups)]
        )

        self.body_conv = nn.Conv2d(dim, dim, kernel_size=3, padding=1)
        self.up = make_upsampler(scale, dim, activation="gelu")
        self.tail = nn.Conv2d(dim, channels, kernel_size=3, padding=1)

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

        out = torch.clamp(out, 0.0, 1.5)

        return out


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
    if method == "hat":
        return HAT(scale=scale, channels=channels, dim=96, n_groups=4, depth=4, heads=4)
    if method == "dat":
        return DAT(scale=scale, channels=channels, dim=96, n_groups=4, depth=4, heads=4)
    # REVISION (B6): DRCT recent SOTA baseline (compact, matched budget).
    if method == "drct":
        return DRCT(scale=scale, channels=channels)
    raise ValueError(f"Unsupported method: {method}")


# ============================================================
# 5. 체크포인트 로드
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
        for p in ["module.", "model.", "net_g.", "generator."]:
            if nk.startswith(p):
                nk = nk[len(p):]
        new_state[nk] = v
    return new_state


def get_checkpoint_name(mode, method, scale, band_name=None):
    if mode == "joint5ch":
        return f"{mode}_{method}_x{scale}_best.pth"

    if mode == "bandwise":
        return f"{mode}_{method}_{band_name}_x{scale}_best.pth"

    raise ValueError(f"Unsupported mode: {mode}")


_DISCOVERED_CKPTS = None


def _discover_checkpoints():
    """Index every '<name>.pth' under any sr_checkpoints folder in SR_RESULTS_ROOT.

    DRCT was trained in its own run folder, so hard-coding a single directory is
    fragile. Scanning once at start-up costs a few seconds and makes the script
    independent of the SR_RUN_NAME that was used. When the same checkpoint name
    exists in several runs the most recently modified file wins.
    """
    global _DISCOVERED_CKPTS
    if _DISCOVERED_CKPTS is not None:
        return _DISCOVERED_CKPTS

    found = {}
    if AUTO_DISCOVER_CHECKPOINTS and SR_RESULTS_ROOT.exists():
        for p in SR_RESULTS_ROOT.glob("*/sr_checkpoints/*.pth"):
            prev = found.get(p.name)
            if prev is None or p.stat().st_mtime > prev.stat().st_mtime:
                found[p.name] = p
    _DISCOVERED_CKPTS = found
    if found:
        print(f"[INFO] checkpoint index: {len(found)} files under {SR_RESULTS_ROOT}")
    return found


def get_checkpoint_path(mode, method, scale, band_name=None):
    name = get_checkpoint_name(mode, method, scale, band_name)

    # 1) explicit directories, in priority order
    for d in [CHECKPOINT_DIR, CHECKPOINT_DIR_FALLBACK] + EXTRA_CHECKPOINT_DIRS:
        cand = d / name
        if cand.exists():
            return cand

    # 2) whole-tree index (finds DRCT whatever its run folder is called)
    hit = _discover_checkpoints().get(name)
    if hit is not None:
        return hit

    return CHECKPOINT_DIR / name  # not found; triggers the missing-file handling


def report_checkpoint_availability():
    """Print a mode x method x scale availability matrix before doing any work."""
    print("\n[CHECK] checkpoint availability")
    missing = 0
    for mode in MODES:
        for method in METHODS:
            marks = []
            for scale in SCALES:
                if mode == "joint5ch":
                    ok = get_checkpoint_path(mode, method, scale).exists()
                else:
                    ok = all(get_checkpoint_path(mode, method, scale, b).exists()
                             for b in BAND_CODES)
                marks.append("O" if ok else "X")
                if not ok:
                    missing += 1
            print(f"   {mode:9} {method:8} " +
                  "  ".join(f"x{sc}:{mk}" for sc, mk in zip(SCALES, marks)))
    if missing:
        print(f"   [WARN] {missing} scale-mode-method combinations have no checkpoint.")
        if REUSE_SAVED_SR:
            print("          Cells whose SR cube was already saved by an earlier run "
                  "are still rendered from disk; only genuinely new ones stay blank.")
        else:
            print("          Those cells will be blank in the figures.")
    else:
        print("   all required checkpoints found")
    return missing


def load_model_from_checkpoint(method, scale, mode="joint5ch", channels=5, band_name=None):
    ckpt_path = get_checkpoint_path(mode=mode, method=method, scale=scale, band_name=band_name)

    if not ckpt_path.exists():
        return None, ckpt_path

    if method == "esrgan":
        model = ESRGANGenerator(scale=scale, channels=channels, nf=64, n_rrdb=6).to(DEVICE)
    else:
        model = build_model(method, scale, channels=channels).to(DEVICE)

    checkpoint = torch.load(ckpt_path, map_location=DEVICE)

    if method == "esrgan":
        if isinstance(checkpoint, dict) and "generator_state_dict" in checkpoint:
            state = checkpoint["generator_state_dict"]
        else:
            state = extract_state_dict(checkpoint)
    else:
        if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
            state = checkpoint["model_state_dict"]
        else:
            state = extract_state_dict(checkpoint)

    state = clean_state_dict_keys(state)

    # checkpoint와 모델 구조가 동일해야 strict=True가 정상.
    # 혹시 일부 prefix/구조 차이가 있으면 경고 후 strict=False 재시도.
    try:
        model.load_state_dict(state, strict=True)
    except Exception as e:
        warnings.warn(f"Strict checkpoint loading failed for {ckpt_path}: {e}\nRetry strict=False.")
        model.load_state_dict(state, strict=False)

    model.eval()
    return model, ckpt_path


# ============================================================
# 6. Inference
# ============================================================

def get_infer_tile_settings(method):
    if method == "swinir":
        return SWINIR_TILE_LR_SIZE, SWINIR_TILE_OVERLAP
    if method in ("hat", "dat", "drct"):
        return HAT_TILE_LR_SIZE, HAT_TILE_OVERLAP
    return DEFAULT_TILE_LR_SIZE, DEFAULT_TILE_OVERLAP


@torch.no_grad()
def infer_full_image(model, lr_cube, scale, channels=5, band_index=None, tile_lr_size=None, tile_overlap=None):
    model.eval()

    lr_norm_full = normalize_cube(lr_cube, 1.0)

    if channels == 5:
        lr_norm = lr_norm_full
    elif channels == 1:
        lr_norm = lr_norm_full[..., band_index:band_index + 1]
    else:
        raise ValueError(f"Unsupported channels: {channels}")

    h_lr, w_lr = lr_norm.shape[:2]
    h_sr = h_lr * scale
    w_sr = w_lr * scale

    output = np.zeros((h_sr, w_sr, channels), dtype=np.float32)
    weight = np.zeros((h_sr, w_sr, channels), dtype=np.float32)

    if tile_lr_size is None:
        tile_lr_size = DEFAULT_TILE_LR_SIZE
    if tile_overlap is None:
        tile_overlap = DEFAULT_TILE_OVERLAP

    tile_lr_size = int(max(8, tile_lr_size))
    tile_overlap = int(max(0, min(tile_overlap, tile_lr_size - 1)))

    step = tile_lr_size - tile_overlap
    if step <= 0:
        step = tile_lr_size

    y_list = list(range(0, h_lr, step))
    x_list = list(range(0, w_lr, step))

    for y in y_list:
        for x in x_list:
            y2 = min(y + tile_lr_size, h_lr)
            x2 = min(x + tile_lr_size, w_lr)

            y1 = max(0, y2 - tile_lr_size)
            x1 = max(0, x2 - tile_lr_size)

            tile = lr_norm[y1:y2, x1:x2, :]
            inp = np_to_tensor(tile).unsqueeze(0).to(DEVICE)

            pred = model(inp)[0]
            pred_np = tensor_to_np(pred)
            pred_np = np.nan_to_num(pred_np, nan=0.0, posinf=1.5, neginf=0.0)

            sy1 = y1 * scale
            sx1 = x1 * scale
            sy2 = sy1 + pred_np.shape[0]
            sx2 = sx1 + pred_np.shape[1]

            output[sy1:sy2, sx1:sx2, :] += pred_np
            weight[sy1:sy2, sx1:sx2, :] += 1.0

    output = output / np.maximum(weight, 1e-6)
    output = np.nan_to_num(output, nan=0.0, posinf=1.5, neginf=0.0)
    output = denormalize_cube(output, 1.0)
    return output.astype(np.float32)


def infer_joint5ch(method, scale, lr_cube):
    model, ckpt_path = load_model_from_checkpoint(
        method=method,
        scale=scale,
        mode="joint5ch",
        channels=5,
    )

    if model is None:
        return None, ckpt_path

    tile_size, tile_overlap = get_infer_tile_settings(method)

    sr = infer_full_image(
        model=model,
        lr_cube=lr_cube,
        scale=scale,
        channels=5,
        tile_lr_size=tile_size,
        tile_overlap=tile_overlap,
    )

    return sr, ckpt_path


def infer_bandwise(method, scale, lr_cube):
    """
    bandwise 모델 inference.

    중요:
    - 내부 밴드 순서: Blue, Green, Red, RedEdge, NIR
    - checkpoint 파일명: B, G, R, E, N
    따라서 bi는 BAND_NAMES 기준으로 유지하고,
    checkpoint를 찾을 때만 BAND_CODES를 사용합니다.
    """
    h_lr, w_lr, _ = lr_cube.shape
    h_sr = h_lr * scale
    w_sr = w_lr * scale

    sr_cube = np.zeros((h_sr, w_sr, 5), dtype=np.float32)
    loaded_paths = []

    for bi, (band_name, band_code) in enumerate(zip(BAND_NAMES, BAND_CODES)):
        model, ckpt_path = load_model_from_checkpoint(
            method=method,
            scale=scale,
            mode="bandwise",
            channels=1,
            band_name=band_code,  # checkpoint 파일명은 B/G/R/E/N 사용
        )

        if model is None:
            print(
                f"[WARN] Missing bandwise checkpoint | "
                f"method={method}, scale=x{scale}, band={band_name}, code={band_code}, path={ckpt_path}"
            )
            return None, ckpt_path

        tile_size, tile_overlap = get_infer_tile_settings(method)

        sr_band = infer_full_image(
            model=model,
            lr_cube=lr_cube,
            scale=scale,
            channels=1,
            band_index=bi,  # 실제 cube에서 추출할 밴드 index는 Blue/Green/Red/RedEdge/NIR 순서
            tile_lr_size=tile_size,
            tile_overlap=tile_overlap,
        )

        sr_cube[..., bi:bi + 1] = sr_band
        loaded_paths.append(str(ckpt_path))

    return sr_cube.astype(np.float32), loaded_paths


# ============================================================
# 7. NDVI 저장
# ============================================================

def save_index_outputs(base_out_dir, scene_id, scale, mode, method, hr, sr):
    """
    저장:
      - HR NDVI png
      - SR NDVI png
      - NDVI error png
      - NDVI panel png
      - HR GNDVI png
      - SR GNDVI png
      - GNDVI error png
      - GNDVI panel png
      - HR NDRE png
      - SR NDRE png
      - NDRE error png
      - NDRE panel png
    """
    ensure_dir(base_out_dir)

    hr, sr = crop_common_np(hr, sr)
    hr_m, sr_m, _, _ = mask_nodata_for_metrics(hr, sr)

    result = save_band_error_outputs(
        base_out_dir=base_out_dir,
        scene_id=scene_id,
        scale=scale,
        mode=mode,
        method=method,
        hr=hr_m,
        sr=sr_m,
    )
    result.update({
        "HR_NDVI_path": "",
        "SR_NDVI_path": "",
        "NDVI_error_path": "",
        "NDVI_error_npy_path": "",
        "NDVI_panel_path": "",
        "NDVI_error_vmax": np.nan,
        "NDVI_MAE": np.nan,
        "HR_GNDVI_path": "",
        "SR_GNDVI_path": "",
        "GNDVI_error_path": "",
        "GNDVI_error_npy_path": "",
        "GNDVI_panel_path": "",
        "GNDVI_error_vmax": np.nan,
        "GNDVI_MAE": np.nan,
        "HR_NDRE_path": "",
        "SR_NDRE_path": "",
        "NDRE_error_path": "",
        "NDRE_error_npy_path": "",
        "NDRE_panel_path": "",
        "NDRE_error_vmax": np.nan,
        "NDRE_MAE": np.nan,
    })

    # --------------------------------------------------------
    # NDVI
    # --------------------------------------------------------
    hr_ndvi = compute_index_map(hr_m, "ndvi")
    sr_ndvi = compute_index_map(sr_m, "ndvi")

    if hr_ndvi is not None and sr_ndvi is not None:
        ndvi_err = np.abs(hr_ndvi - sr_ndvi)

        hr_ndvi_img = index_to_colormap(hr_ndvi, vmin=NDVI_VIS_MIN, vmax=NDVI_VIS_MAX)
        sr_ndvi_img = index_to_colormap(sr_ndvi, vmin=NDVI_VIS_MIN, vmax=NDVI_VIS_MAX)
        ndvi_err_img, ndvi_err_vmax = diff_to_heatmap(ndvi_err, vmin=0.0, vmax=None)
        ndvi_err_img = append_vertical_colorbar_rgb(ndvi_err_img, 0.0, ndvi_err_vmax)

        hr_ndvi_img = add_label_rgb(hr_ndvi_img, "HR NDVI")
        sr_ndvi_img = add_label_rgb(sr_ndvi_img, "SR NDVI")
        ndvi_err_img = add_label_rgb(ndvi_err_img, "NDVI error")

        hr_ndvi_path = base_out_dir / f"{scene_id}_x{scale}_{mode}_{method}_HR_NDVI.png"
        sr_ndvi_path = base_out_dir / f"{scene_id}_x{scale}_{mode}_{method}_SR_NDVI.png"
        ndvi_err_path = base_out_dir / f"{scene_id}_x{scale}_{mode}_{method}_NDVI_error.png"
        ndvi_err_npy_path = base_out_dir / f"{scene_id}_x{scale}_{mode}_{method}_NDVI_error.npy"
        ndvi_panel_path = base_out_dir / f"{scene_id}_x{scale}_{mode}_{method}_NDVI_panel.png"

        save_rgb_png(hr_ndvi_path, hr_ndvi_img)
        save_rgb_png(sr_ndvi_path, sr_ndvi_img)
        save_rgb_png(ndvi_err_path, ndvi_err_img)
        np.save(ndvi_err_npy_path, ndvi_err.astype(np.float32))

        ndvi_panel = np.concatenate([hr_ndvi_img, sr_ndvi_img, ndvi_err_img], axis=1)
        save_rgb_png(ndvi_panel_path, ndvi_panel)

        result.update({
            "HR_NDVI_path": str(hr_ndvi_path),
            "SR_NDVI_path": str(sr_ndvi_path),
            "NDVI_error_path": str(ndvi_err_path),
            "NDVI_error_npy_path": str(ndvi_err_npy_path),
            "NDVI_panel_path": str(ndvi_panel_path),
            "NDVI_error_vmax": float(ndvi_err_vmax),
            "NDVI_MAE": float(np.nanmean(ndvi_err)),
        })

    # --------------------------------------------------------
    # GNDVI
    # --------------------------------------------------------
    hr_gndvi = compute_index_map(hr_m, "gndvi")
    sr_gndvi = compute_index_map(sr_m, "gndvi")

    if hr_gndvi is not None and sr_gndvi is not None:
        gndvi_err = np.abs(hr_gndvi - sr_gndvi)

        hr_gndvi_img = index_to_colormap(hr_gndvi, vmin=GNDVI_VIS_MIN, vmax=GNDVI_VIS_MAX)
        sr_gndvi_img = index_to_colormap(sr_gndvi, vmin=GNDVI_VIS_MIN, vmax=GNDVI_VIS_MAX)
        gndvi_err_img, gndvi_err_vmax = diff_to_heatmap(gndvi_err, vmin=0.0, vmax=None)
        gndvi_err_img = append_vertical_colorbar_rgb(gndvi_err_img, 0.0, gndvi_err_vmax)

        hr_gndvi_img = add_label_rgb(hr_gndvi_img, "HR GNDVI")
        sr_gndvi_img = add_label_rgb(sr_gndvi_img, "SR GNDVI")
        gndvi_err_img = add_label_rgb(gndvi_err_img, "GNDVI error")

        hr_gndvi_path = base_out_dir / f"{scene_id}_x{scale}_{mode}_{method}_HR_GNDVI.png"
        sr_gndvi_path = base_out_dir / f"{scene_id}_x{scale}_{mode}_{method}_SR_GNDVI.png"
        gndvi_err_path = base_out_dir / f"{scene_id}_x{scale}_{mode}_{method}_GNDVI_error.png"
        gndvi_err_npy_path = base_out_dir / f"{scene_id}_x{scale}_{mode}_{method}_GNDVI_error.npy"
        gndvi_panel_path = base_out_dir / f"{scene_id}_x{scale}_{mode}_{method}_GNDVI_panel.png"

        save_rgb_png(hr_gndvi_path, hr_gndvi_img)
        save_rgb_png(sr_gndvi_path, sr_gndvi_img)
        save_rgb_png(gndvi_err_path, gndvi_err_img)
        np.save(gndvi_err_npy_path, gndvi_err.astype(np.float32))

        gndvi_panel = np.concatenate([hr_gndvi_img, sr_gndvi_img, gndvi_err_img], axis=1)
        save_rgb_png(gndvi_panel_path, gndvi_panel)

        result.update({
            "HR_GNDVI_path": str(hr_gndvi_path),
            "SR_GNDVI_path": str(sr_gndvi_path),
            "GNDVI_error_path": str(gndvi_err_path),
            "GNDVI_error_npy_path": str(gndvi_err_npy_path),
            "GNDVI_panel_path": str(gndvi_panel_path),
            "GNDVI_error_vmax": float(gndvi_err_vmax),
            "GNDVI_MAE": float(np.nanmean(gndvi_err)),
        })

    # --------------------------------------------------------
    # NDRE
    # --------------------------------------------------------
    hr_ndre = compute_index_map(hr_m, "ndre")
    sr_ndre = compute_index_map(sr_m, "ndre")

    if hr_ndre is not None and sr_ndre is not None:
        ndre_err = np.abs(hr_ndre - sr_ndre)

        hr_ndre_img = index_to_colormap(hr_ndre, vmin=NDRE_VIS_MIN, vmax=NDRE_VIS_MAX)
        sr_ndre_img = index_to_colormap(sr_ndre, vmin=NDRE_VIS_MIN, vmax=NDRE_VIS_MAX)
        ndre_err_img, ndre_err_vmax = diff_to_heatmap(ndre_err, vmin=0.0, vmax=None)
        ndre_err_img = append_vertical_colorbar_rgb(ndre_err_img, 0.0, ndre_err_vmax)

        hr_ndre_img = add_label_rgb(hr_ndre_img, "HR NDRE")
        sr_ndre_img = add_label_rgb(sr_ndre_img, "SR NDRE")
        ndre_err_img = add_label_rgb(ndre_err_img, "NDRE error")

        hr_ndre_path = base_out_dir / f"{scene_id}_x{scale}_{mode}_{method}_HR_NDRE.png"
        sr_ndre_path = base_out_dir / f"{scene_id}_x{scale}_{mode}_{method}_SR_NDRE.png"
        ndre_err_path = base_out_dir / f"{scene_id}_x{scale}_{mode}_{method}_NDRE_error.png"
        ndre_err_npy_path = base_out_dir / f"{scene_id}_x{scale}_{mode}_{method}_NDRE_error.npy"
        ndre_panel_path = base_out_dir / f"{scene_id}_x{scale}_{mode}_{method}_NDRE_panel.png"

        save_rgb_png(hr_ndre_path, hr_ndre_img)
        save_rgb_png(sr_ndre_path, sr_ndre_img)
        save_rgb_png(ndre_err_path, ndre_err_img)
        np.save(ndre_err_npy_path, ndre_err.astype(np.float32))

        ndre_panel = np.concatenate([hr_ndre_img, sr_ndre_img, ndre_err_img], axis=1)
        save_rgb_png(ndre_panel_path, ndre_panel)

        result.update({
            "HR_NDRE_path": str(hr_ndre_path),
            "SR_NDRE_path": str(sr_ndre_path),
            "NDRE_error_path": str(ndre_err_path),
            "NDRE_error_npy_path": str(ndre_err_npy_path),
            "NDRE_panel_path": str(ndre_panel_path),
            "NDRE_error_vmax": float(ndre_err_vmax),
            "NDRE_MAE": float(np.nanmean(ndre_err)),
        })

    return result


# ============================================================
# 8. Overview grid panels
# ============================================================

def read_rgb_png(path):
    path = Path(path)
    if (not str(path)) or (not path.exists()):
        return None

    img_bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img_bgr is None:
        return None

    return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)


def resize_rgb_for_grid(img_rgb, cell_size):
    if img_rgb is None:
        return None
    cell_w, cell_h = cell_size
    return cv2.resize(img_rgb, (int(cell_w), int(cell_h)), interpolation=cv2.INTER_AREA)


def draw_centered_text(img, text, center_x, baseline_y, font_scale, color, thickness=1):
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
    x = int(center_x - tw / 2)
    y = int(baseline_y)
    cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, thickness, cv2.LINE_AA)


def draw_left_text(img, text, x, center_y, font_scale, color, thickness=1):
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
    y = int(center_y + th / 2)
    cv2.putText(img, text, (int(x), y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, thickness, cv2.LINE_AA)


def find_summary_row(summary_rows, scale, mode, method):
    for row in summary_rows:
        if row.get("status") != "done":
            continue
        if int(row.get("scale", -1)) != int(scale):
            continue
        if row.get("mode") == mode and row.get("method") == method:
            return row
    return None


def make_vi_overview_grid(summary_rows, out_path, mode, cell_size=(120, 120)):
    """Create an overview panel like: rows = VI x scale, columns = HR + methods."""
    methods = ["hr"] + GRID_METHODS
    columns = [METHOD_LABEL[m] for m in methods]
    indices = ["NDVI", "GNDVI", "NDRE"]

    cell_w, cell_h = cell_size
    left_margin = 110
    top_margin = 46
    right_margin = 18
    bottom_margin = 18
    gap_x = 10
    gap_y = 10

    n_rows = len(indices) * len(SCALES)
    n_cols = len(columns)
    width = left_margin + n_cols * cell_w + (n_cols - 1) * gap_x + right_margin
    height = top_margin + n_rows * cell_h + (n_rows - 1) * gap_y + bottom_margin

    canvas = np.full((height, width, 3), 255, dtype=np.uint8)
    label_color = (28, 57, 114)

    for ci, label in enumerate(columns):
        x = left_margin + ci * (cell_w + gap_x)
        draw_centered_text(canvas, label, x + cell_w / 2, 24, 0.56, label_color, 1)

    for ri, index_name in enumerate(indices):
        for si, scale in enumerate(SCALES):
            row_i = ri * len(SCALES) + si
            y = top_margin + row_i * (cell_h + gap_y)
            row_label = f"{index_name} x{scale}"
            draw_left_text(canvas, row_label, 8, y + cell_h / 2, 0.56, label_color, 1)

            for ci, method in enumerate(methods):
                x = left_margin + ci * (cell_w + gap_x)

                if method == "hr":
                    src_row = find_summary_row(summary_rows, scale, "joint5ch", "bicubic")
                    path_key = f"HR_{index_name}_path"
                elif method == "bicubic":
                    src_row = find_summary_row(summary_rows, scale, "joint5ch", "bicubic")
                    path_key = f"SR_{index_name}_path"
                else:
                    src_row = find_summary_row(summary_rows, scale, mode, method)
                    path_key = f"SR_{index_name}_path"

                img = None
                if src_row is not None:
                    img = resize_rgb_for_grid(read_rgb_png(src_row.get(path_key, "")), cell_size)

                if img is None:
                    img = np.full((cell_h, cell_w, 3), 245, dtype=np.uint8)
                    draw_centered_text(img, "missing", cell_w / 2, cell_h / 2, 0.5, (80, 80, 80), 1)

                canvas[y:y + cell_h, x:x + cell_w, :] = img

    ensure_dir(Path(out_path).parent)
    save_rgb_png(out_path, canvas)
    return out_path


def load_error_npy(path):
    path = Path(path)
    if (not str(path)) or (not path.exists()):
        return None
    try:
        return np.load(path).astype(np.float32)
    except Exception:
        return None


def compute_shared_error_vmax(summary_rows, mode, row_specs):
    """Use one vmax for the whole grid: max of per-map 99th percentile values."""
    # Fixed scale (paper convention) unless disabled.
    if ERROR_MAP_FIXED_VMAX is not None:
        return float(ERROR_MAP_FIXED_VMAX)

    vmax_values = []
    methods = list(GRID_METHODS)

    for _, _, npy_key in row_specs:
        for scale in SCALES:
            for method in methods:
                if method == "bicubic":
                    src_row = find_summary_row(summary_rows, scale, "joint5ch", "bicubic")
                else:
                    src_row = find_summary_row(summary_rows, scale, mode, method)

                if src_row is None:
                    continue

                err = load_error_npy(src_row.get(npy_key, ""))
                if err is None:
                    continue

                valid = np.isfinite(err)
                if np.any(valid):
                    vmax_values.append(float(np.percentile(err[valid], ERROR_MAP_PERCENTILE)))

    if len(vmax_values) == 0:
        return 1.0

    vmax = float(np.nanmax(vmax_values))
    if (not np.isfinite(vmax)) or vmax <= 0.0:
        return 1.0
    return vmax


def error_array_to_grid_rgb(err, shared_vmax, cell_size):
    if err is None:
        return None

    valid = np.isfinite(err)
    if not np.any(valid):
        return None

    err_img, _ = diff_to_heatmap(err, vmin=0.0, vmax=shared_vmax)
    err_img[~valid] = 0
    return resize_rgb_for_grid(err_img, cell_size)


def make_error_overview_grid(summary_rows, out_path, mode, row_specs, cell_size=(120, 120)):
    """Create an error-map overview panel with one shared color bar for the full grid."""
    methods = list(GRID_METHODS)
    columns = [METHOD_LABEL[m] for m in methods]

    cell_w, cell_h = cell_size
    left_margin = 122
    top_margin = 46
    right_margin = 96
    bottom_margin = 18
    gap_x = 10
    gap_y = 10

    n_rows = len(row_specs) * len(SCALES)
    n_cols = len(columns)
    width = left_margin + n_cols * cell_w + (n_cols - 1) * gap_x + right_margin
    height = top_margin + n_rows * cell_h + (n_rows - 1) * gap_y + bottom_margin

    canvas = np.full((height, width, 3), 255, dtype=np.uint8)
    label_color = (28, 57, 114)
    shared_vmax = compute_shared_error_vmax(summary_rows, mode, row_specs)

    for ci, label in enumerate(columns):
        x = left_margin + ci * (cell_w + gap_x)
        draw_centered_text(canvas, label, x + cell_w / 2, 24, 0.56, label_color, 1)

    for ri, (row_label_base, png_key, npy_key) in enumerate(row_specs):
        for si, scale in enumerate(SCALES):
            row_i = ri * len(SCALES) + si
            y = top_margin + row_i * (cell_h + gap_y)
            row_label = f"{row_label_base} x{scale}"
            draw_left_text(canvas, row_label, 8, y + cell_h / 2, 0.50, label_color, 1)

            for ci, method in enumerate(methods):
                x = left_margin + ci * (cell_w + gap_x)

                if method == "bicubic":
                    src_row = find_summary_row(summary_rows, scale, "joint5ch", "bicubic")
                else:
                    src_row = find_summary_row(summary_rows, scale, mode, method)

                img = None
                if src_row is not None:
                    img = error_array_to_grid_rgb(load_error_npy(src_row.get(npy_key, "")), shared_vmax, cell_size)

                    # Backward-compatible fallback for runs made before raw error NPYs were added.
                    if img is None:
                        fallback = read_rgb_png(src_row.get(png_key, ""))
                        if fallback is not None:
                            fallback = fallback[:, :min(fallback.shape[1], fallback.shape[0]), :]
                            img = resize_rgb_for_grid(fallback, cell_size)

                if img is None:
                    img = np.full((cell_h, cell_w, 3), 245, dtype=np.uint8)
                    draw_centered_text(img, "missing", cell_w / 2, cell_h / 2, 0.5, (80, 80, 80), 1)

                canvas[y:y + cell_h, x:x + cell_w, :] = img

    # One shared colorbar for the full grid.
    bar_x = left_margin + n_cols * cell_w + (n_cols - 1) * gap_x + 18
    bar_y0 = top_margin
    bar_h = n_rows * cell_h + (n_rows - 1) * gap_y
    bar_canvas = append_vertical_colorbar_rgb(
        np.full((bar_h, 1, 3), 255, dtype=np.uint8),
        0.0,
        shared_vmax,
        colorbar_width=74,
        bar_width=18,
    )[:, 1:, :]
    canvas[bar_y0:bar_y0 + bar_h, bar_x:bar_x + bar_canvas.shape[1], :] = bar_canvas

    ensure_dir(Path(out_path).parent)
    save_rgb_png(out_path, canvas)
    return out_path


def save_error_overview_grids(summary_rows, output_dir):
    output_dir = Path(output_dir)
    out_paths = []

    vi_error_specs = [
        ("NDVI error", "NDVI_error_path", "NDVI_error_npy_path"),
        ("GNDVI error", "GNDVI_error_path", "GNDVI_error_npy_path"),
        ("NDRE error", "NDRE_error_path", "NDRE_error_npy_path"),
    ]
    band_error_specs = [
        (
            f"{band_name} error",
            f"{band_name.replace(' ', '')}_error_path",
            f"{band_name.replace(' ', '')}_error_npy_path",
        )
        for band_name in BAND_NAMES
    ]

    for mode in MODES:
        vi_out = output_dir / f"overview_vi_error_{mode}_grid.png"
        make_error_overview_grid(summary_rows, vi_out, mode=mode, row_specs=vi_error_specs)
        out_paths.append(str(vi_out))

        band_out = output_dir / f"overview_band_error_{mode}_grid.png"
        make_error_overview_grid(summary_rows, band_out, mode=mode, row_specs=band_error_specs)
        out_paths.append(str(band_out))

    return out_paths


def save_vi_overview_grids(summary_rows, output_dir):
    output_dir = Path(output_dir)
    out_paths = []
    for mode in MODES:
        out_path = output_dir / f"overview_vi_{mode}_grid.png"
        make_vi_overview_grid(summary_rows, out_path, mode=mode)
        out_paths.append(str(out_path))
    return out_paths


# ============================================================
# 8. Main
# ============================================================

def main():
    ensure_dir(OUTPUT_DIR)

    if not HR_PATH.exists():
        raise FileNotFoundError(f"HR file not found: {HR_PATH}")

    if not CHECKPOINT_DIR.exists() and not AUTO_DISCOVER_CHECKPOINTS:
        raise FileNotFoundError(f"CHECKPOINT_DIR not found: {CHECKPOINT_DIR}")

    hr_raw = np.load(HR_PATH).astype(np.float32)

    if hr_raw.ndim != 3 or hr_raw.shape[2] != CHANNELS:
        raise ValueError(f"Invalid HR shape: {hr_raw.shape}. Expected H x W x 5.")

    scene_id = HR_PATH.stem.replace("_HR", "")
    summary_rows = []

    print("====================================================")
    print("Single-scene vegetation index generation: NDVI / GNDVI / NDRE")
    print("====================================================")
    print(f"HR_PATH        : {HR_PATH}")
    print(f"CHECKPOINT_DIR : {CHECKPOINT_DIR}")
    print(f"OUTPUT_DIR     : {OUTPUT_DIR}")
    print(f"DEVICE         : {DEVICE}")
    print(f"HR shape       : {hr_raw.shape}")
    print(f"METHODS        : {', '.join(METHODS)}")
    print(f"REUSE saved SR : {REUSE_SAVED_SR}"
          + (f"  (force re-infer: {', '.join(sorted(FORCE_INFER_METHODS))})"
             if FORCE_INFER_METHODS else ""))
    print(f"SCALES         : {SCALES}")
    print("====================================================")

    report_checkpoint_availability()

    for scale in SCALES:
        scale_out_dir = OUTPUT_DIR / f"x{scale}"
        ensure_dir(scale_out_dir)

        print(f"\n[INFO] Processing scale x{scale}")

        lr_cube, hr_aligned, lr_info_path, lr_generated = load_or_make_lr(
            hr_raw=hr_raw,
            hr_path=HR_PATH,
            scale=scale,
            out_dir=scale_out_dir,
        )

        print(f"[INFO] LR shape       : {lr_cube.shape}")
        print(f"[INFO] HR aligned     : {hr_aligned.shape}")
        print(f"[INFO] LR path        : {lr_info_path}")
        print(f"[INFO] LR generated   : {lr_generated}")

        # ----------------------------------------------------
        # Bicubic baseline
        # ----------------------------------------------------
        if INCLUDE_BICUBIC_BASELINE:
            bicubic_sr = resize_cube(
                lr_cube,
                out_h=hr_aligned.shape[0],
                out_w=hr_aligned.shape[1],
                interpolation=cv2.INTER_CUBIC,
            )

            bicubic_dir = scale_out_dir / "bicubic"
            ensure_dir(bicubic_dir)

            index_info = save_index_outputs(
                base_out_dir=bicubic_dir,
                scene_id=scene_id,
                scale=scale,
                mode="joint5ch",
                method="bicubic",
                hr=hr_aligned,
                sr=bicubic_sr,
            )

            sr_npy_path = bicubic_dir / f"{scene_id}_x{scale}_joint5ch_bicubic_SR.npy"
            if SAVE_SR_NPY:
                np.save(sr_npy_path, bicubic_sr.astype(np.float32))

            row = {
                "scene_id": scene_id,
                "scale": scale,
                "mode": "joint5ch",
                "method": "bicubic",
                "status": "done",
                "lr_path": lr_info_path,
                "lr_generated": lr_generated,
                "checkpoint": "",
                "sr_npy_path": str(sr_npy_path) if SAVE_SR_NPY else "",
            }
            row.update(index_info)
            summary_rows.append(row)

        # ----------------------------------------------------
        # Checkpoint-based SR models
        # ----------------------------------------------------
        for mode in MODES:
            for method in METHODS:
                print(f"[INFO] {mode} | {method} | x{scale}")

                method_out_dir = scale_out_dir / mode / method
                ensure_dir(method_out_dir)

                cached_sr_path = (method_out_dir /
                                  f"{scene_id}_x{scale}_{mode}_{method}_SR.npy")
                reuse = (REUSE_SAVED_SR
                         and method not in FORCE_INFER_METHODS
                         and cached_sr_path.exists())

                try:
                    sr_cube = None
                    checkpoint_str = ""

                    if reuse:
                        cached = np.load(cached_sr_path).astype(np.float32)
                        if cached.shape[:2] == hr_aligned.shape[:2]:
                            sr_cube = cached
                            checkpoint_str = f"reused saved cube: {cached_sr_path}"
                            _REUSE_STATS["reused"] += 1
                            print("       reusing saved SR cube (no inference)")
                        else:
                            print(f"       saved cube shape {cached.shape[:2]} != "
                                  f"HR {hr_aligned.shape[:2]}; re-running inference")
                            reuse = False

                    if sr_cube is None:
                        if mode == "joint5ch":
                            sr_cube, ckpt_info = infer_joint5ch(method, scale, lr_cube)
                            checkpoint_str = str(ckpt_info)

                        elif mode == "bandwise":
                            sr_cube, ckpt_info = infer_bandwise(method, scale, lr_cube)
                            checkpoint_str = " | ".join(ckpt_info) if isinstance(ckpt_info, list) else str(ckpt_info)

                        else:
                            raise ValueError(f"Unsupported mode: {mode}")

                        if sr_cube is not None:
                            _REUSE_STATS["inferred"] += 1

                    if sr_cube is None:
                        print(f"[WARN] Missing checkpoint -> skip: {checkpoint_str}")

                        row = {
                            "scene_id": scene_id,
                            "scale": scale,
                            "mode": mode,
                            "method": method,
                            "status": "checkpoint_missing",
                            "lr_path": lr_info_path,
                            "lr_generated": lr_generated,
                            "checkpoint": checkpoint_str,
                            "sr_npy_path": "",
                            "HR_NDVI_path": "",
                            "SR_NDVI_path": "",
                            "NDVI_error_path": "",
                            "NDVI_panel_path": "",
                            "NDVI_MAE": np.nan,
                            "HR_GNDVI_path": "",
                            "SR_GNDVI_path": "",
                            "GNDVI_error_path": "",
                            "GNDVI_panel_path": "",
                            "GNDVI_MAE": np.nan,
                            "HR_NDRE_path": "",
                            "SR_NDRE_path": "",
                            "NDRE_error_path": "",
                            "NDRE_panel_path": "",
                            "NDRE_MAE": np.nan,
                        }
                        summary_rows.append(row)
                        continue

                    sr_cube = np.nan_to_num(
                        sr_cube.astype(np.float32),
                        nan=0.0,
                        posinf=1.5,
                        neginf=0.0,
                    )

                    sr_npy_path = method_out_dir / f"{scene_id}_x{scale}_{mode}_{method}_SR.npy"
                    if SAVE_SR_NPY:
                        np.save(sr_npy_path, sr_cube)

                    index_info = save_index_outputs(
                        base_out_dir=method_out_dir,
                        scene_id=scene_id,
                        scale=scale,
                        mode=mode,
                        method=method,
                        hr=hr_aligned,
                        sr=sr_cube,
                    )

                    row = {
                        "scene_id": scene_id,
                        "scale": scale,
                        "mode": mode,
                        "method": method,
                        "status": "done",
                        "lr_path": lr_info_path,
                        "lr_generated": lr_generated,
                        "checkpoint": checkpoint_str,
                        "sr_npy_path": str(sr_npy_path) if SAVE_SR_NPY else "",
                    }
                    row.update(index_info)
                    summary_rows.append(row)

                except Exception as e:
                    print(f"[ERROR] {mode} | {method} | x{scale} -> {e}")

                    row = {
                        "scene_id": scene_id,
                        "scale": scale,
                        "mode": mode,
                        "method": method,
                        "status": f"error: {e}",
                        "lr_path": lr_info_path,
                        "lr_generated": lr_generated,
                        "checkpoint": "",
                        "sr_npy_path": "",
                        "HR_NDVI_path": "",
                        "SR_NDVI_path": "",
                        "NDVI_error_path": "",
                        "NDVI_panel_path": "",
                        "NDVI_MAE": np.nan,
                        "HR_GNDVI_path": "",
                        "SR_GNDVI_path": "",
                        "GNDVI_error_path": "",
                        "GNDVI_panel_path": "",
                        "GNDVI_MAE": np.nan,
                        "HR_NDRE_path": "",
                        "SR_NDRE_path": "",
                        "NDRE_error_path": "",
                        "NDRE_panel_path": "",
                        "NDRE_MAE": np.nan,
                    }
                    summary_rows.append(row)

    summary_csv = OUTPUT_DIR / "summary_single_scene_vi_ndvi_gndvi_ndre.csv"

    if len(summary_rows) > 0:
        fieldnames = []
        for row in summary_rows:
            for key in row.keys():
                if key not in fieldnames:
                    fieldnames.append(key)
        with open(summary_csv, "w", encoding="utf-8-sig", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for row in summary_rows:
                writer.writerow(row)

    overview_grid_paths = []
    if len(summary_rows) > 0:
        overview_grid_paths = save_vi_overview_grids(summary_rows, OUTPUT_DIR)
        overview_grid_paths.extend(save_error_overview_grids(summary_rows, OUTPUT_DIR))


    ok = sum(1 for r in summary_rows if r.get("status") == "done")
    skipped = sum(1 for r in summary_rows if r.get("status") == "checkpoint_missing")

    print("\n====================================================")
    print("[DONE]")
    print(f"Cells      : {ok} rendered, {skipped} skipped (no checkpoint)")
    print(f"GPU work   : {_REUSE_STATS['inferred']} inferred, "
          f"{_REUSE_STATS['reused']} reused from saved cubes")
    if skipped:
        miss = sorted({(r["mode"], r["method"]) for r in summary_rows
                       if r.get("status") == "checkpoint_missing"})
        print("  missing    : " + ", ".join(f"{m}/{me}" for m, me in miss))
    print(f"Summary CSV: {summary_csv}")
    for p in overview_grid_paths:
        print(f"Overview   : {p}")
    print(f"Outputs    : {OUTPUT_DIR}")
    print("====================================================")


if __name__ == "__main__":
    main()
