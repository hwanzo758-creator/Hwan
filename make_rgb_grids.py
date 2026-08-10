"""
Build RGB / false-color / zoom comparison grids (manuscript Figures 3, 4, 5)
WITH HAT and DAT columns, from the SR .npy cubes already saved by
Make_NDVI_hat_dat.py (SAVE_SR_NPY=True).

No GPU / no re-inference needed — this only reads the saved cubes and renders.
Run AFTER Make_NDVI_hat_dat.py has produced the single-scene outputs.

Columns:  HR | LR | Bicubic | SRCNN | EDSR | RCAN | SwinIR | ESRGAN | HAT | DAT
Rows:     x2 | x3 | x4
Outputs (in OUTPUT_DIR):
  overview_rgb_joint5ch_grid.png / overview_rgb_bandwise_grid.png     (Fig 3)
  overview_falsecolor_joint5ch_grid.png / ..._bandwise_grid.png       (Fig 4)
  overview_zoom_rgb_joint5ch_grid.png / ..._bandwise_grid.png         (Fig 5)
"""
from pathlib import Path
import numpy as np
import cv2
from PIL import Image, ImageDraw, ImageFont

# ============================================================
# Config  (must match Make_NDVI_hat_dat.py)
# ============================================================
DATASET_DIR = Path(
    "/home/whanjo/datasets/Python_file/SR_HR_LR_TEST/light_cabbage_training_dataset"
)
HR_PATH = DATASET_DIR / "HR_npy" / "scene_012483_1배추_2전라도_2_HR.npy"
OUTPUT_DIR = Path(
    "/home/whanjo/datasets/Python_file/SR_HR_LR_TEST/"
    "single_scene_vi_outputs_scene_012483_ndvi_gndvi_ndre"
)

SCALES = [2, 3, 4]
MODES = ["joint5ch", "bandwise"]
METHODS = ["bicubic", "srcnn", "edsr", "rcan", "swinir", "esrgan", "hat", "dat", "mambair"]
METHOD_LABEL = {
    "bicubic": "Bicubic", "srcnn": "SRCNN", "edsr": "EDSR", "rcan": "RCAN",
    "swinir": "SwinIR", "esrgan": "ESRGAN", "hat": "HAT", "dat": "DAT", "mambair": "MambaIR",
}
COLUMNS = ["HR", "LR"] + [METHOD_LABEL[m] for m in METHODS]

# Band order: Blue, Green, Red, RedEdge, NIR  (indices 0..4)
RGB_BANDS = (2, 1, 0)          # Red, Green, Blue
FALSECOLOR_BANDS = (4, 2, 1)   # NIR, Red, Green
NODATA_THRESHOLD = -9999.0
STRETCH_LOW, STRETCH_HIGH = 2, 98   # percentile stretch

CELL = 150            # grid cell size (px)
GAP = 8
LEFT_MARGIN = 70      # for row labels
TOP_MARGIN = 34       # for column labels

# Zoom (Fig 5): crop box on the HR grid (row0, col0, size). Adjust as needed.
ZOOM_Y, ZOOM_X, ZOOM_SIZE = 250, 250, 200


# ============================================================
# Helpers
# ============================================================
def scene_id_from_hr(p):
    return p.stem.replace("_HR", "")


SCENE_ID = scene_id_from_hr(HR_PATH)


def load_cube(path):
    return np.load(path).astype(np.float32)


def align_hr_x3(hr, scale):
    """For x3, center-crop HR to the natural LR*scale grid (800 -> 798)."""
    h, w = hr.shape[:2]
    lr_h, lr_w = h // scale, w // scale
    sh, sw = lr_h * scale, lr_w * scale
    if (sh, sw) == (h, w):
        return hr
    y0 = (h - sh) // 2
    x0 = (w - sw) // 2
    return hr[y0:y0 + sh, x0:x0 + sw]


def make_lr(hr, scale):
    h, w = hr.shape[:2]
    lr_h, lr_w = h // scale, w // scale
    bands = [cv2.resize(hr[..., c], (lr_w, lr_h), interpolation=cv2.INTER_AREA)
             for c in range(hr.shape[-1])]
    return np.stack(bands, axis=-1)


def band_ranges(ref_cube):
    """Per-band vmin/vmax from the HR reference (percentile stretch)."""
    vmins, vmaxs = [], []
    for c in range(ref_cube.shape[-1]):
        b = ref_cube[..., c]
        valid = np.isfinite(b) & (b > NODATA_THRESHOLD)
        if valid.sum() == 0:
            vmins.append(0.0); vmaxs.append(1.0); continue
        lo = float(np.percentile(b[valid], STRETCH_LOW))
        hi = float(np.percentile(b[valid], STRETCH_HIGH))
        if not np.isfinite(hi) or hi <= lo:
            lo, hi = float(np.nanmin(b[valid])), float(np.nanmax(b[valid]))
        if hi <= lo:
            hi = lo + 1e-6
        vmins.append(lo); vmaxs.append(hi)
    return vmins, vmaxs


def cube_to_display(cube, bands, vmins, vmaxs):
    chans = []
    for bi in bands:
        b = cube[..., bi].astype(np.float32)
        out = (b - vmins[bi]) / (vmaxs[bi] - vmins[bi])
        out = np.clip(out, 0, 1)
        chans.append((out * 255).astype(np.uint8))
    return np.stack(chans, axis=-1)  # H,W,3 (RGB order)


def resize_cell(rgb, size=CELL):
    return cv2.resize(rgb, (size, size), interpolation=cv2.INTER_AREA)


def sr_npy_path(scale, mode, method):
    if method == "bicubic":
        return OUTPUT_DIR / f"x{scale}" / "bicubic" / f"{SCENE_ID}_x{scale}_joint5ch_bicubic_SR.npy"
    return OUTPUT_DIR / f"x{scale}" / mode / method / f"{SCENE_ID}_x{scale}_{mode}_{method}_SR.npy"


def load_font(size=15):
    for f in ["/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
              "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"]:
        if Path(f).exists():
            return ImageFont.truetype(f, size)
    return ImageFont.load_default()


# ============================================================
# Grid builder
# ============================================================
def build_grid(kind, mode, out_path, zoom=False):
    """kind: 'rgb' or 'fc'. rows=SCALES, cols=COLUMNS."""
    bands = RGB_BANDS if kind == "rgb" else FALSECOLOR_BANDS
    n_rows, n_cols = len(SCALES), len(COLUMNS)
    W = LEFT_MARGIN + n_cols * CELL + (n_cols - 1) * GAP + 8
    H = TOP_MARGIN + n_rows * CELL + (n_rows - 1) * GAP + 8
    canvas = Image.new("RGB", (W, H), "white")
    draw = ImageDraw.Draw(canvas)
    font = load_font(15)

    # column labels
    for j, col in enumerate(COLUMNS):
        x = LEFT_MARGIN + j * (CELL + GAP) + CELL // 2
        draw.text((x, 8), col, fill="black", font=font, anchor="ma")

    for i, scale in enumerate(SCALES):
        y = TOP_MARGIN + i * (CELL + GAP)
        draw.text((6, y + CELL // 2), f"x{scale}", fill="black", font=font, anchor="lm")

        hr = align_hr_x3(load_cube(HR_PATH), scale)
        vmins, vmaxs = band_ranges(hr)
        lr = make_lr(hr, scale)
        lr_up = np.stack([cv2.resize(lr[..., c], (hr.shape[1], hr.shape[0]),
                                     interpolation=cv2.INTER_NEAREST)
                          for c in range(lr.shape[-1])], axis=-1)

        def crop(cube):
            if not zoom:
                return cube
            y0 = min(ZOOM_Y, max(0, cube.shape[0] - ZOOM_SIZE))
            x0 = min(ZOOM_X, max(0, cube.shape[1] - ZOOM_SIZE))
            return cube[y0:y0 + ZOOM_SIZE, x0:x0 + ZOOM_SIZE]

        cells = {}
        cells["HR"] = cube_to_display(crop(hr), bands, vmins, vmaxs)
        cells["LR"] = cube_to_display(crop(lr_up), bands, vmins, vmaxs)
        for m in METHODS:
            p = sr_npy_path(scale, mode, m)
            if p.exists():
                sr = load_cube(p)
                # align sr to hr size if needed
                if sr.shape[:2] != hr.shape[:2]:
                    sr = np.stack([cv2.resize(sr[..., c], (hr.shape[1], hr.shape[0]),
                                              interpolation=cv2.INTER_CUBIC)
                                   for c in range(sr.shape[-1])], axis=-1)
                cells[METHOD_LABEL[m]] = cube_to_display(crop(sr), bands, vmins, vmaxs)
            else:
                cells[METHOD_LABEL[m]] = np.full((CELL, CELL, 3), 235, np.uint8)
                print(f"[WARN] missing SR npy: {p}")

        for j, col in enumerate(COLUMNS):
            x = LEFT_MARGIN + j * (CELL + GAP)
            img = resize_cell(cells[col])
            canvas.paste(Image.fromarray(img), (x, y))

    canvas.save(out_path)
    print(f"[SAVED] {out_path}")


def main():
    for mode in MODES:
        build_grid("rgb", mode, OUTPUT_DIR / f"overview_rgb_{mode}_grid.png")
        build_grid("fc", mode, OUTPUT_DIR / f"overview_falsecolor_{mode}_grid.png")
        build_grid("rgb", mode, OUTPUT_DIR / f"overview_zoom_rgb_{mode}_grid.png", zoom=True)
    print("[DONE] RGB / false-color / zoom grids with HAT+DAT")


if __name__ == "__main__":
    main()
