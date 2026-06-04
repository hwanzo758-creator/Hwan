# build_light_cabbage_gangwon_dataset.py
# ============================================================
# AIHub 배추 강원도 다분광 UAV 데이터셋 구축 코드
# ============================================================
#
# 목적:
# - 원본 B/G/R/E/N TIFF를 stack하여 5-band HR cube 생성
# - HR cube를 .npy로 저장하거나, 용량 절약을 위해 저장하지 않을 수 있음
# - LR_x2, LR_x3, LR_x4 cube를 .npy로 저장
# - JSON polygon annotation을 mask로 변환
# - RGB preview / false-color preview / overlay image 저장
# - scene_index.csv에 원본 TIFF, HR/LR/mask/preview 경로 저장
#
# 밴드 순서:
#   cube[:, :, 0] = Blue
#   cube[:, :, 1] = Green
#   cube[:, :, 2] = Red
#   cube[:, :, 3] = RedEdge
#   cube[:, :, 4] = NIR
#
# 입력 예시:
#   /home/whanjo/Downloads/Opensource_data/107.노지작물(배추 등) 작황 데이터/
#     01-1.정식개방데이터/Training/01.원천데이터/TS_1.배추_1.강원도/
#     01-1.정식개방데이터/Training/02.라벨링데이터/TL_1.배추_1.강원도/
#
# 출력 예시:
#   light_cabbage_gangwon_dataset/
#   ├── index/scene_index.csv
#   ├── HR_npy/
#   ├── LR_x2_npy/
#   ├── LR_x3_npy/
#   ├── LR_x4_npy/
#   ├── masks/
#   ├── rgb_preview/
#   ├── false_color/
#   ├── overlay/
#   ├── metadata/
#   └── logs/
#
# 실행:
#   python build_light_cabbage_gangwon_dataset.py
#
# 필요 패키지:
#   pip install numpy tifffile opencv-python pillow tqdm
# ============================================================


# ============================================================
# 1. 사용자 설정
# ============================================================

from pathlib import Path
from collections import defaultdict, Counter
import csv
import json
import re
import traceback
from datetime import datetime

import numpy as np
import tifffile as tiff
import cv2
from PIL import Image, ImageDraw
from tqdm import tqdm


# ------------------------------------------------------------
# 원본 데이터 경로
# ------------------------------------------------------------
RAW_ROOT = Path(
    "/home/whanjo/Downloads/Opensource_data/"
    "107.노지작물(배추 등) 작황 데이터/"
    "01-1.정식개방데이터/Training/01.원천데이터/TS_1.배추_1.강원도"
)

LABEL_ROOT = Path(
    "/home/whanjo/Downloads/Opensource_data/"
    "107.노지작물(배추 등) 작황 데이터/"
    "01-1.정식개방데이터/Training/02.라벨링데이터/TL_1.배추_1.강원도"
)

# ------------------------------------------------------------
# 결과 저장 경로
# ------------------------------------------------------------
OUT_ROOT = Path("./light_cabbage_gangwon_dataset")

# ------------------------------------------------------------
# 저장 옵션
# ------------------------------------------------------------
SAVE_HR_NPY = True          # 용량이 부족하면 False로 변경
SAVE_LR_NPY = True          # LR 데이터 저장
SAVE_MASK_NPY = True        # mask npy 저장
SAVE_PREVIEW = True         # RGB / false color 저장
SAVE_OVERLAY = True         # 라벨 overlay 저장
SAVE_METADATA = True        # scene별 metadata json 저장

# ------------------------------------------------------------
# LR scale
# ------------------------------------------------------------
SCALES = [2, 3, 4]

# ------------------------------------------------------------
# 파일 확장자
# ------------------------------------------------------------
TIFF_EXTS = [".tif", ".tiff", ".TIF", ".TIFF"]
JSON_EXTS = [".json", ".JSON"]

# ------------------------------------------------------------
# mask 설정
# ------------------------------------------------------------
# AIHub JSON 내부 구조가 데이터마다 조금씩 다를 수 있어 가능한 여러 key를 탐색함
# 배추 polygon만 1로 저장하고 싶으면 CABBAGE_ONLY=True
CABBAGE_ONLY = True

# label name 후보
CABBAGE_KEYWORDS = [
    "배추", "cabbage", "Cabbage", "CABBAGE",
    "CTVT_CROPS_SPCHCKN", "crops"
]

# ------------------------------------------------------------
# preview stretch percentile
# ------------------------------------------------------------
PREVIEW_LOW_P = 2
PREVIEW_HIGH_P = 98

# ------------------------------------------------------------
# dtype
# ------------------------------------------------------------
SAVE_DTYPE = np.float32

# ------------------------------------------------------------
# resize interpolation
# ------------------------------------------------------------
LR_INTERPOLATION = cv2.INTER_AREA

# ------------------------------------------------------------
# 파일명에서 scene id를 추출하기 위한 옵션
# 예: 220719KW023B0093.tif, 220719KW023G0093.tif ...
# band 문자 B/G/R/E/N만 제거하고 나머지를 scene key로 사용
# ------------------------------------------------------------
BAND_LETTERS = ["B", "G", "R", "E", "N"]
BAND_NAMES = ["Blue", "Green", "Red", "RedEdge", "NIR"]


# ============================================================
# 2. 유틸 함수
# ============================================================

def ensure_dirs(out_root: Path):
    """결과 저장 폴더 생성."""
    dirs = {
        "index": out_root / "index",
        "hr": out_root / "HR_npy",
        "mask": out_root / "masks",
        "rgb": out_root / "rgb_preview",
        "false": out_root / "false_color",
        "overlay": out_root / "overlay",
        "metadata": out_root / "metadata",
        "logs": out_root / "logs",
    }

    for scale in SCALES:
        dirs[f"lr_x{scale}"] = out_root / f"LR_x{scale}_npy"

    for d in dirs.values():
        d.mkdir(parents=True, exist_ok=True)

    return dirs


def read_tiff_as_array(path: Path) -> np.ndarray:
    """TIFF 파일을 numpy array로 읽음."""
    arr = tiff.imread(str(path))

    # 일부 TIFF가 (1,H,W) 또는 (H,W,1) 형태일 수 있음
    arr = np.asarray(arr)

    if arr.ndim == 3:
        if arr.shape[0] == 1:
            arr = arr[0]
        elif arr.shape[-1] == 1:
            arr = arr[..., 0]
        else:
            # 다중채널 TIFF일 경우 첫 채널만 사용
            arr = arr[..., 0]

    if arr.ndim != 2:
        raise ValueError(f"Unsupported TIFF shape: {path}, shape={arr.shape}")

    return arr.astype(np.float32)


def crop_to_common_size(arrays):
    """여러 밴드 이미지 크기가 다를 경우 최소 공통 크기로 crop."""
    heights = [a.shape[0] for a in arrays]
    widths = [a.shape[1] for a in arrays]

    h_min = min(heights)
    w_min = min(widths)

    cropped = [a[:h_min, :w_min] for a in arrays]
    return cropped


def stack_5band_cube(paths_by_band: dict) -> np.ndarray:
    """B/G/R/E/N 5개 TIFF를 (H,W,5) cube로 stack."""
    arrays = []
    for b in BAND_LETTERS:
        arr = read_tiff_as_array(paths_by_band[b])
        arrays.append(arr)

    arrays = crop_to_common_size(arrays)
    cube = np.stack(arrays, axis=-1).astype(SAVE_DTYPE)
    return cube


def resize_cube_lr(cube: np.ndarray, scale: int) -> np.ndarray:
    """HR cube를 scale만큼 낮은 해상도의 LR cube로 변환."""
    h, w, c = cube.shape

    new_h = max(1, h // scale)
    new_w = max(1, w // scale)

    # scale로 정확히 나누어지지 않는 경우도 안전하게 처리
    lr_bands = []
    for i in range(c):
        band = cube[:, :, i]
        lr = cv2.resize(band, (new_w, new_h), interpolation=LR_INTERPOLATION)
        lr_bands.append(lr)

    lr_cube = np.stack(lr_bands, axis=-1).astype(SAVE_DTYPE)
    return lr_cube


def robust_percentile_stretch(img, low_p=2, high_p=98):
    """preview용 percentile stretch. 학습 데이터에는 적용하지 않음."""
    img = img.astype(np.float32)

    finite = np.isfinite(img)
    if not np.any(finite):
        return np.zeros_like(img, dtype=np.uint8)

    vals = img[finite]
    lo = np.percentile(vals, low_p)
    hi = np.percentile(vals, high_p)

    if hi <= lo:
        hi = lo + 1e-6

    out = (img - lo) / (hi - lo)
    out = np.clip(out, 0, 1)
    return (out * 255).astype(np.uint8)


def make_rgb_preview(cube: np.ndarray) -> np.ndarray:
    """RGB preview 생성. cube order: B,G,R,E,N."""
    r = robust_percentile_stretch(cube[:, :, 2], PREVIEW_LOW_P, PREVIEW_HIGH_P)
    g = robust_percentile_stretch(cube[:, :, 1], PREVIEW_LOW_P, PREVIEW_HIGH_P)
    b = robust_percentile_stretch(cube[:, :, 0], PREVIEW_LOW_P, PREVIEW_HIGH_P)
    rgb = np.stack([r, g, b], axis=-1)
    return rgb


def make_false_color_preview(cube: np.ndarray) -> np.ndarray:
    """False color preview 생성. NIR, Red, Green."""
    nir = robust_percentile_stretch(cube[:, :, 4], PREVIEW_LOW_P, PREVIEW_HIGH_P)
    red = robust_percentile_stretch(cube[:, :, 2], PREVIEW_LOW_P, PREVIEW_HIGH_P)
    green = robust_percentile_stretch(cube[:, :, 1], PREVIEW_LOW_P, PREVIEW_HIGH_P)
    false = np.stack([nir, red, green], axis=-1)
    return false


def save_png(path: Path, img: np.ndarray):
    """PNG 저장."""
    Image.fromarray(img).save(str(path))


def is_valid_cube(cube: np.ndarray) -> bool:
    """NaN/Inf 포함 여부 확인."""
    return np.all(np.isfinite(cube))


# ============================================================
# 3. 파일 grouping 함수
# ============================================================

def find_files(root: Path, exts):
    """root 하위 모든 파일 탐색."""
    files = []
    for ext in exts:
        files.extend(root.rglob(f"*{ext}"))
    return sorted(files)


def parse_band_and_scene_key(path: Path):
    """
    TIFF 파일명에서 band letter와 scene key 추출.

    기대 예시:
      220719KW023B0093.tif
      220719KW023G0093.tif
      220719KW023R0093.tif
      220719KW023E0093.tif
      220719KW023N0093.tif

    처리 방식:
      - 확장자 제거
      - 마지막 숫자 블록 앞쪽의 B/G/R/E/N을 band로 판단
      - 해당 band 문자만 제거한 문자열을 scene key로 사용

    반환:
      band, scene_key
    """
    stem = path.stem

    # 가장 일반적인 패턴: 중간에 B/G/R/E/N이 있고 그 뒤에 숫자가 오는 경우
    # 예: 220719KW023B0093
    m = re.match(r"^(.*?)([BGREN])(\d+)$", stem)
    if m:
        prefix, band, suffix = m.groups()
        scene_key = f"{prefix}{suffix}"
        return band, scene_key

    # fallback: stem 안에서 band 문자 후보를 찾아 하나씩 제거해보기
    for band in BAND_LETTERS:
        # band 문자가 하나만 있고 주변에 숫자/영문자가 있는 경우
        if band in stem:
            scene_key = stem.replace(band, "", 1)
            return band, scene_key

    return None, stem


def group_tiff_scenes(raw_root: Path):
    """
    TIFF 파일을 scene 단위로 grouping.
    반환:
      scenes: dict
        scene_key -> {"B": path, "G": path, "R": path, "E": path, "N": path}
    """
    tiff_files = find_files(raw_root, TIFF_EXTS)

    scenes = defaultdict(dict)

    for p in tiff_files:
        band, scene_key = parse_band_and_scene_key(p)

        if band in BAND_LETTERS:
            # 같은 scene/band가 여러 개 있을 경우 첫 번째 유지
            if band not in scenes[scene_key]:
                scenes[scene_key][band] = p

    complete_scenes = {}
    incomplete_scenes = {}

    for scene_key, d in scenes.items():
        missing = [b for b in BAND_LETTERS if b not in d]
        if len(missing) == 0:
            complete_scenes[scene_key] = d
        else:
            incomplete_scenes[scene_key] = {
                "available": sorted(d.keys()),
                "missing": missing
            }

    return complete_scenes, incomplete_scenes


def parse_scene_info(scene_key: str):
    """
    scene_key에서 날짜/지역/필드/번호를 가능한 만큼 추출.
    실패해도 scene_key만 유지.
    """
    info = {
        "scene_key": scene_key,
        "date": "",
        "region_code": "",
        "field_code": "",
        "image_index": "",
    }

    # 예: 220719KW0230093
    m = re.match(r"^(\d{6})([A-Za-z]+)(\d{3})(\d+)$", scene_key)
    if m:
        date, region, field, idx = m.groups()
        info["date"] = date
        info["region_code"] = region
        info["field_code"] = field
        info["image_index"] = idx
        return info

    # 날짜만이라도 추출
    m2 = re.search(r"(\d{6})", scene_key)
    if m2:
        info["date"] = m2.group(1)

    return info


# ============================================================
# 4. JSON annotation / mask 함수
# ============================================================

def find_json_for_scene(scene_key: str, label_root: Path):
    """
    scene_key와 가장 유사한 JSON 파일 탐색.
    정확히 일치하지 않는 경우가 있어 숫자 기반 후보도 탐색.
    """
    json_files = find_files(label_root, JSON_EXTS)

    if len(json_files) == 0:
        return None

    # 1) scene_key가 파일명에 포함되는 경우
    for jp in json_files:
        if scene_key in jp.stem:
            return jp

    # 2) scene_key의 숫자 부분과 JSON stem 숫자 부분 비교
    scene_digits = "".join(re.findall(r"\d+", scene_key))
    if scene_digits:
        for jp in json_files:
            jd = "".join(re.findall(r"\d+", jp.stem))
            if jd == scene_digits:
                return jp

    # 3) 이미지 index가 같은 경우
    info = parse_scene_info(scene_key)
    idx = info.get("image_index", "")
    if idx:
        for jp in json_files:
            if idx in jp.stem:
                return jp

    return None


def load_json_safely(path: Path):
    """JSON 안전하게 로드."""
    if path is None or not path.exists():
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except UnicodeDecodeError:
        with open(path, "r", encoding="cp949") as f:
            return json.load(f)


def is_cabbage_annotation(obj: dict):
    """
    annotation object가 배추인지 판단.
    JSON 구조가 다양할 수 있어 모든 문자열 값을 검색.
    """
    if not CABBAGE_ONLY:
        return True

    text_values = []

    def collect_strings(x):
        if isinstance(x, dict):
            for v in x.values():
                collect_strings(v)
        elif isinstance(x, list):
            for v in x:
                collect_strings(v)
        elif isinstance(x, str):
            text_values.append(x)

    collect_strings(obj)
    joined = " ".join(text_values)

    for kw in CABBAGE_KEYWORDS:
        if kw in joined:
            return True

    # 라벨명이 안 들어있는 데이터의 경우, polygon이 있으면 기본적으로 사용하도록 완화
    # 완전 엄격하게 하려면 아래 return을 False로 바꾸면 됨.
    return True


def extract_polygons_from_json(data):
    """
    JSON에서 polygon 좌표를 최대한 robust하게 추출.

    지원 가능한 형태 예:
    - {"ANNOTATIONS": [{"POINTS": [{"X":..., "Y":...}, ...]}]}
    - {"annotations": [{"polygon": [[x,y], ...]}]}
    - {"shapes": [{"points": [[x,y], ...]}]}
    - nested dict/list 내부의 points 후보 자동 탐색

    반환:
      polygons: list of list[(x,y)]
    """
    polygons = []

    def normalize_points(points):
        norm = []

        if not isinstance(points, list):
            return None

        for pt in points:
            if isinstance(pt, dict):
                # 가능한 key 후보
                x = pt.get("x", pt.get("X", pt.get("lon", pt.get("LON", None))))
                y = pt.get("y", pt.get("Y", pt.get("lat", pt.get("LAT", None))))
                if x is not None and y is not None:
                    norm.append((float(x), float(y)))

            elif isinstance(pt, (list, tuple)) and len(pt) >= 2:
                norm.append((float(pt[0]), float(pt[1])))

        if len(norm) >= 3:
            return norm

        return None

    def search(obj, parent_obj=None):
        if isinstance(obj, dict):
            # annotation object 단위로 배추 여부 판단
            current_is_cabbage = is_cabbage_annotation(obj)

            # points key 후보
            for key in [
                "points", "Points", "POINTS",
                "polygon", "Polygon", "POLYGON",
                "segmentation", "Segmentation", "SEGMENTATION",
                "coordinates", "Coordinates", "COORDINATES"
            ]:
                if key in obj:
                    pts = obj[key]

                    # COCO segmentation처럼 [[x1,y1,x2,y2,...]] 형태일 수 있음
                    if key.lower() == "segmentation" and isinstance(pts, list):
                        for seg in pts:
                            if isinstance(seg, list) and len(seg) >= 6 and all(isinstance(v, (int, float)) for v in seg):
                                poly = [(float(seg[i]), float(seg[i+1])) for i in range(0, len(seg)-1, 2)]
                                if len(poly) >= 3 and current_is_cabbage:
                                    polygons.append(poly)

                    # 일반 points
                    poly = normalize_points(pts)
                    if poly is not None and current_is_cabbage:
                        polygons.append(poly)

            # nested search
            for v in obj.values():
                search(v, obj)

        elif isinstance(obj, list):
            for item in obj:
                search(item, parent_obj)

    search(data)

    # 중복 polygon 제거
    unique = []
    seen = set()
    for poly in polygons:
        key = tuple((round(x, 2), round(y, 2)) for x, y in poly)
        if key not in seen:
            unique.append(poly)
            seen.add(key)

    return unique


def polygon_to_mask(json_path: Path, shape_hw):
    """
    JSON polygon을 binary mask로 변환.
    shape_hw: (H,W)
    """
    h, w = shape_hw
    mask = Image.new("L", (w, h), 0)
    draw = ImageDraw.Draw(mask)

    if json_path is None:
        return np.zeros((h, w), dtype=np.uint8), 0

    data = load_json_safely(json_path)
    if data is None:
        return np.zeros((h, w), dtype=np.uint8), 0

    polygons = extract_polygons_from_json(data)

    for poly in polygons:
        # 좌표 범위 보정
        pts = []
        for x, y in poly:
            x = max(0, min(w - 1, float(x)))
            y = max(0, min(h - 1, float(y)))
            pts.append((x, y))

        if len(pts) >= 3:
            draw.polygon(pts, outline=1, fill=1)

    mask_np = np.array(mask, dtype=np.uint8)
    return mask_np, len(polygons)


def make_overlay(rgb: np.ndarray, mask: np.ndarray):
    """
    RGB preview 위에 mask contour overlay 생성.
    """
    overlay = rgb.copy()

    if mask is None or mask.sum() == 0:
        return overlay

    mask_u8 = (mask > 0).astype(np.uint8) * 255
    contours, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # 반투명 green overlay
    color_layer = overlay.copy()
    color_layer[mask > 0] = [0, 255, 0]
    overlay = cv2.addWeighted(overlay, 0.75, color_layer, 0.25, 0)

    # contour
    cv2.drawContours(overlay, contours, -1, (255, 0, 0), 2)

    return overlay


# ============================================================
# 5. 메인 빌드 함수
# ============================================================

def build_dataset():
    dirs = ensure_dirs(OUT_ROOT)

    print("=" * 80)
    print("[INFO] AIHub cabbage multispectral dataset builder")
    print("=" * 80)
    print(f"[INFO] RAW_ROOT   : {RAW_ROOT}")
    print(f"[INFO] LABEL_ROOT : {LABEL_ROOT}")
    print(f"[INFO] OUT_ROOT   : {OUT_ROOT}")
    print(f"[INFO] SAVE_HR_NPY: {SAVE_HR_NPY}")
    print(f"[INFO] SCALES     : {SCALES}")
    print("=" * 80)

    if not RAW_ROOT.exists():
        raise FileNotFoundError(f"RAW_ROOT does not exist: {RAW_ROOT}")

    if not LABEL_ROOT.exists():
        print(f"[WARN] LABEL_ROOT does not exist: {LABEL_ROOT}")
        print("[WARN] mask will be empty.")

    complete_scenes, incomplete_scenes = group_tiff_scenes(RAW_ROOT)

    print(f"[INFO] complete scenes  : {len(complete_scenes)}")
    print(f"[INFO] incomplete scenes: {len(incomplete_scenes)}")

    # logs
    log_path = dirs["logs"] / "scene_grouping_log.json"
    with open(log_path, "w", encoding="utf-8") as f:
        json.dump({
            "created_at": datetime.now().isoformat(),
            "raw_root": str(RAW_ROOT),
            "label_root": str(LABEL_ROOT),
            "complete_scene_count": len(complete_scenes),
            "incomplete_scene_count": len(incomplete_scenes),
            "incomplete_scenes": incomplete_scenes,
        }, f, ensure_ascii=False, indent=2)

    csv_path = dirs["index"] / "scene_index.csv"

    fieldnames = [
        "scene_id",
        "scene_key",
        "date",
        "region_code",
        "field_code",
        "image_index",
        "height",
        "width",
        "bands",
        "band_order",
        "blue_tif",
        "green_tif",
        "red_tif",
        "rededge_tif",
        "nir_tif",
        "json_path",
        "polygon_count",
        "mask_pixel_count",
        "hr_npy",
    ]

    for scale in SCALES:
        fieldnames.append(f"lr_x{scale}_npy")

    fieldnames.extend([
        "mask_npy",
        "rgb_preview",
        "false_color",
        "overlay",
        "metadata_json",
        "status",
        "error_message",
    ])

    rows = []
    success_count = 0
    fail_count = 0
    skipped_nan_count = 0

    scene_items = sorted(complete_scenes.items(), key=lambda x: x[0])

    for idx, (scene_key, paths_by_band) in enumerate(tqdm(scene_items, desc="Building dataset")):
        scene_id = f"scene_{idx:05d}"
        info = parse_scene_info(scene_key)

        row = {
            "scene_id": scene_id,
            "scene_key": scene_key,
            "date": info.get("date", ""),
            "region_code": info.get("region_code", ""),
            "field_code": info.get("field_code", ""),
            "image_index": info.get("image_index", ""),
            "height": "",
            "width": "",
            "bands": 5,
            "band_order": "Blue,Green,Red,RedEdge,NIR",
            "blue_tif": str(paths_by_band["B"]),
            "green_tif": str(paths_by_band["G"]),
            "red_tif": str(paths_by_band["R"]),
            "rededge_tif": str(paths_by_band["E"]),
            "nir_tif": str(paths_by_band["N"]),
            "json_path": "",
            "polygon_count": 0,
            "mask_pixel_count": 0,
            "hr_npy": "",
            "mask_npy": "",
            "rgb_preview": "",
            "false_color": "",
            "overlay": "",
            "metadata_json": "",
            "status": "pending",
            "error_message": "",
        }

        for scale in SCALES:
            row[f"lr_x{scale}_npy"] = ""

        try:
            cube = stack_5band_cube(paths_by_band)
            h, w, c = cube.shape

            row["height"] = h
            row["width"] = w
            row["bands"] = c

            if not is_valid_cube(cube):
                row["status"] = "skipped_nan_or_inf"
                row["error_message"] = "Cube contains NaN or Inf"
                skipped_nan_count += 1
                rows.append(row)
                continue

            # JSON 찾기
            json_path = find_json_for_scene(scene_key, LABEL_ROOT) if LABEL_ROOT.exists() else None
            row["json_path"] = str(json_path) if json_path is not None else ""

            # mask 생성
            mask = None
            polygon_count = 0
            if SAVE_MASK_NPY or SAVE_OVERLAY:
                mask, polygon_count = polygon_to_mask(json_path, (h, w))
                row["polygon_count"] = polygon_count
                row["mask_pixel_count"] = int(mask.sum())

            # HR 저장
            hr_path = dirs["hr"] / f"{scene_id}_HR.npy"
            if SAVE_HR_NPY:
                np.save(hr_path, cube.astype(SAVE_DTYPE))
                row["hr_npy"] = str(hr_path)

            # LR 저장
            if SAVE_LR_NPY:
                for scale in SCALES:
                    lr_cube = resize_cube_lr(cube, scale)
                    lr_path = dirs[f"lr_x{scale}"] / f"{scene_id}_LR_x{scale}.npy"
                    np.save(lr_path, lr_cube.astype(SAVE_DTYPE))
                    row[f"lr_x{scale}_npy"] = str(lr_path)

            # mask 저장
            if SAVE_MASK_NPY and mask is not None:
                mask_path = dirs["mask"] / f"{scene_id}_mask.npy"
                np.save(mask_path, mask.astype(np.uint8))
                row["mask_npy"] = str(mask_path)

            # preview 저장
            rgb_path = dirs["rgb"] / f"{scene_id}_rgb.png"
            false_path = dirs["false"] / f"{scene_id}_false_color.png"
            overlay_path = dirs["overlay"] / f"{scene_id}_overlay.png"

            rgb = None
            if SAVE_PREVIEW or SAVE_OVERLAY:
                rgb = make_rgb_preview(cube)

            if SAVE_PREVIEW:
                false = make_false_color_preview(cube)
                save_png(rgb_path, rgb)
                save_png(false_path, false)
                row["rgb_preview"] = str(rgb_path)
                row["false_color"] = str(false_path)

            if SAVE_OVERLAY and rgb is not None and mask is not None:
                overlay = make_overlay(rgb, mask)
                save_png(overlay_path, overlay)
                row["overlay"] = str(overlay_path)

            # metadata 저장
            if SAVE_METADATA:
                meta_path = dirs["metadata"] / f"{scene_id}_metadata.json"
                metadata = {
                    "scene_id": scene_id,
                    "scene_key": scene_key,
                    "created_at": datetime.now().isoformat(),
                    "shape": {
                        "height": h,
                        "width": w,
                        "bands": c
                    },
                    "band_order": {
                        "0": "Blue",
                        "1": "Green",
                        "2": "Red",
                        "3": "RedEdge",
                        "4": "NIR"
                    },
                    "source_tiff": {
                        "Blue": str(paths_by_band["B"]),
                        "Green": str(paths_by_band["G"]),
                        "Red": str(paths_by_band["R"]),
                        "RedEdge": str(paths_by_band["E"]),
                        "NIR": str(paths_by_band["N"]),
                    },
                    "json_path": str(json_path) if json_path is not None else None,
                    "polygon_count": int(polygon_count),
                    "mask_pixel_count": int(row["mask_pixel_count"]),
                    "save_options": {
                        "SAVE_HR_NPY": SAVE_HR_NPY,
                        "SAVE_LR_NPY": SAVE_LR_NPY,
                        "SCALES": SCALES,
                        "SAVE_MASK_NPY": SAVE_MASK_NPY,
                        "SAVE_PREVIEW": SAVE_PREVIEW,
                        "SAVE_OVERLAY": SAVE_OVERLAY
                    },
                    "note": "Preview images are percentile-stretched only for visualization. NPY values are not normalized."
                }

                with open(meta_path, "w", encoding="utf-8") as f:
                    json.dump(metadata, f, ensure_ascii=False, indent=2)
                row["metadata_json"] = str(meta_path)

            row["status"] = "success"
            success_count += 1

        except Exception as e:
            fail_count += 1
            row["status"] = "failed"
            row["error_message"] = f"{type(e).__name__}: {str(e)}"

            error_log_path = dirs["logs"] / f"{scene_id}_error.txt"
            with open(error_log_path, "w", encoding="utf-8") as f:
                f.write(traceback.format_exc())

        rows.append(row)

    # CSV 저장
    with open(csv_path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    # Summary 저장
    status_counter = Counter([r["status"] for r in rows])
    summary = {
        "created_at": datetime.now().isoformat(),
        "raw_root": str(RAW_ROOT),
        "label_root": str(LABEL_ROOT),
        "out_root": str(OUT_ROOT),
        "complete_scene_count": len(complete_scenes),
        "incomplete_scene_count": len(incomplete_scenes),
        "total_processed_rows": len(rows),
        "success_count": success_count,
        "fail_count": fail_count,
        "skipped_nan_or_inf_count": skipped_nan_count,
        "status_counter": dict(status_counter),
        "scene_index_csv": str(csv_path),
        "save_options": {
            "SAVE_HR_NPY": SAVE_HR_NPY,
            "SAVE_LR_NPY": SAVE_LR_NPY,
            "SCALES": SCALES,
            "SAVE_MASK_NPY": SAVE_MASK_NPY,
            "SAVE_PREVIEW": SAVE_PREVIEW,
            "SAVE_OVERLAY": SAVE_OVERLAY,
            "SAVE_METADATA": SAVE_METADATA,
        }
    }

    summary_path = dirs["logs"] / "build_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print("=" * 80)
    print("[DONE] Dataset build completed")
    print("=" * 80)
    print(f"[RESULT] success              : {success_count}")
    print(f"[RESULT] failed               : {fail_count}")
    print(f"[RESULT] skipped NaN/Inf      : {skipped_nan_count}")
    print(f"[RESULT] scene_index.csv      : {csv_path}")
    print(f"[RESULT] summary              : {summary_path}")
    print("=" * 80)

    return summary


# ============================================================
# 6. 실행부
# ============================================================

if __name__ == "__main__":
    build_dataset()
