"""Produce the public scene index from the internal one.

The internal index stores absolute paths into a private home directory and
repeats the full AI Hub source path for each of the five bands, which makes the
file large and unsuitable for release. This script keeps the information that is
actually needed to reproduce the reported split - the scene identity, the AI Hub
source file names, the cube paths relative to the dataset root, and the original
row order - and drops everything else.

    python make_scene_index_release.py \
        --src  /path/to/light_cabbage_training_dataset/index/scene_index.csv \
        --dst  scene_index.csv

Row order is preserved: the pipeline reshuffles with a fixed seed (RANDOM_SEED
= 42) at run time, so the 70/15/15 split is only reproducible if the rows are
read in the same order as in the manuscript run.
"""

import argparse
import csv
import os
import sys

# Columns kept in the released file, in this order.
KEEP = [
    "global_index", "scene_id", "scene_key",
    "crop_group", "crop_name", "region_group", "region_name",
    "date", "region_code", "field_code", "image_index",
    "height", "width", "bands", "band_order",
    "blue_tif", "green_tif", "red_tif", "rededge_tif", "nir_tif", "json_path",
    "hr_npy", "lr_x2_npy", "lr_x3_npy", "lr_x4_npy",
    "status",
]
# Columns reduced to a bare file name (they hold absolute private paths).
BASENAME_ONLY = {"blue_tif", "green_tif", "red_tif", "rededge_tif", "nir_tif", "json_path"}
# Dataset-root-relative paths: strip a leading dataset folder if present.
RELATIVE = {"hr_npy", "lr_x2_npy", "lr_x3_npy", "lr_x4_npy"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--dst", default="scene_index.csv")
    ap.add_argument("--dataset-root-name", default="light_cabbage_training_dataset",
                    help="leading folder stripped from the cube paths")
    a = ap.parse_args()

    if not os.path.exists(a.src):
        sys.exit(f"source not found: {a.src}")

    n_in = n_out = 0
    dropped = None
    with open(a.src, "r", encoding="utf-8-sig", newline="") as fin, \
         open(a.dst, "w", encoding="utf-8", newline="") as fout:
        rd = csv.DictReader(fin)
        missing = [c for c in KEEP if c not in rd.fieldnames]
        if missing:
            sys.exit(f"columns missing from source: {missing}")
        dropped = [c for c in rd.fieldnames if c not in KEEP]
        wr = csv.DictWriter(fout, fieldnames=KEEP)
        wr.writeheader()
        for row in rd:
            n_in += 1
            out = {}
            for c in KEEP:
                v = (row.get(c) or "").strip()
                if c in BASENAME_ONLY and v:
                    v = os.path.basename(v)
                elif c in RELATIVE and v:
                    v = v.replace("\\", "/")
                    pre = a.dataset_root_name.rstrip("/") + "/"
                    if v.startswith(pre):
                        v = v[len(pre):]
                out[c] = v
            wr.writerow(out)
            n_out += 1

    size = os.path.getsize(a.dst) / 2 ** 20
    print(f"rows      : {n_in} -> {n_out}")
    print(f"columns   : kept {len(KEEP)}, dropped {len(dropped)} -> {dropped}")
    print(f"written   : {a.dst}  ({size:.1f} MiB)")
    print("\nCheck before committing: no absolute path should remain.")
    print(f"  grep -c '/home/' {a.dst}      # expect 0")


if __name__ == "__main__":
    main()
