"""
Figure 2 - per-scene RMSE boxplot (x4, band-wise) including HAT, DAT and DRCT.

Reads the per-scene metrics CSVs produced by the SR pipeline and draws the
boxplot. No GPU required.

Changed vs make_fig2_boxplot.py:
  - DRCT column added
  - DRCT was trained in its own run folder, so metrics CSVs are looked up in a
    primary directory first and only then in fallback / auto-discovered ones.
    Existing methods therefore keep coming from exactly the same files that
    produced the published figure; only the new column is sourced elsewhere.
  - per-method n, median and IQR printed, so the figure can be sanity-checked
    against Table 3 before it goes into the manuscript
  - paths and method list overridable by environment variable

Usage:
    python make_fig2_boxplot_9methods.py

    SR_METRICS_DIR=/path/to/<main_run>/sr_logs \
    SR_OUTPUT_ROOT=/path/to/..._SR_RESULTS \
    python make_fig2_boxplot_9methods.py

Environment overrides (all optional):
    SR_METRICS_DIR    primary metrics folder (the run the tables were built from)
    SR_METRICS_DIRS   extra folders, colon-separated, searched next
    SR_OUTPUT_ROOT    root scanned for */sr_logs/ as a last resort
    SR_FIG2_OUT       output PNG path
    SR_FIG2_SCALE     default 4
    SR_FIG2_MODE      default bandwise
    SR_FIG2_METHODS   comma-separated method keys, in plotting order
"""
import csv
import os
import sys
from pathlib import Path
from statistics import median

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ============================================================
# Config
# ============================================================
METRICS_DIR = Path(os.environ.get(
    "SR_METRICS_DIR",
    "/home/whanjo/datasets/Python_file/SR_HR_LR_TEST/"
    "light_cabbage_training_dataset_SR_RESULTS/revision_full_x2x4/sr_logs",
))

EXTRA_METRICS_DIRS = [
    Path(p) for p in os.environ.get("SR_METRICS_DIRS", "").split(":") if p.strip()
]

SR_RESULTS_ROOT = Path(os.environ.get(
    "SR_OUTPUT_ROOT",
    "/home/whanjo/datasets/Python_file/SR_HR_LR_TEST/"
    "light_cabbage_training_dataset_SR_RESULTS",
))

SCALE = int(os.environ.get("SR_FIG2_SCALE", "4"))
MODE = os.environ.get("SR_FIG2_MODE", "bandwise")
OUT = Path(os.environ.get(
    "SR_FIG2_OUT", f"Fig2_boxplot_rmse_x{SCALE}_{MODE}_9methods.png"))

# SRCNN is included here, but the published Figure 2 had no SRCNN box because
# metrics_bandwise_srcnn_x4.csv does not exist in the results archive - the
# archive holds SRCNN for joint5ch only, at every scale. Run
# run_srcnn_bandwise.sh to produce it. Until then this script prints SRCNN as
# MISSING, skips it, and still draws the remaining boxes.
DEFAULT_METHODS = ["bicubic", "srcnn", "edsr", "rcan", "swinir", "esrgan",
                   "hat", "dat", "drct"]
METHODS = ([m.strip() for m in os.environ["SR_FIG2_METHODS"].split(",") if m.strip()]
           if os.environ.get("SR_FIG2_METHODS") else list(DEFAULT_METHODS))

LABEL = {
    "bicubic": "Bicubic", "srcnn": "SRCNN", "edsr": "EDSR", "rcan": "RCAN",
    "swinir": "SwinIR", "esrgan": "ESRGAN", "hat": "HAT", "dat": "DAT",
    "drct": "DRCT",
}

RMSE_EXCLUDE_THRESHOLD = 1.0   # same rule as the manuscript tables
YLIM_TOP = os.environ.get("SR_FIG2_YLIM")   # None -> keep the published 0.058


# ============================================================
# Metrics lookup
# ============================================================
_INDEX = None


def _auto_index():
    """Index metrics CSVs under every */sr_logs folder in SR_RESULTS_ROOT.

    Used only for methods that are not in the primary directory, so the boxes
    that were already published are never silently re-sourced.
    """
    global _INDEX
    if _INDEX is not None:
        return _INDEX
    found = {}
    if SR_RESULTS_ROOT.exists():
        for p in SR_RESULTS_ROOT.glob("*/sr_logs/metrics_*.csv"):
            prev = found.get(p.name)
            if prev is None or p.stat().st_mtime > prev.stat().st_mtime:
                found[p.name] = p
    _INDEX = found
    return found


def find_metrics(method):
    name = f"metrics_{MODE}_{method}_x{SCALE}.csv"
    for d in [METRICS_DIR] + EXTRA_METRICS_DIRS:
        cand = d / name
        if cand.exists():
            return cand, "primary" if d == METRICS_DIR else "extra"
    hit = _auto_index().get(name)
    if hit is not None:
        return hit, "auto"
    return None, None


def read_rmse(path):
    with open(path, encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    vals = []
    for r in rows:
        v = r.get("RMSE")
        if v in (None, "", "nan", "NaN"):
            continue
        try:
            x = float(v)
        except ValueError:
            continue
        if x == x and x < RMSE_EXCLUDE_THRESHOLD:   # x == x filters NaN
            vals.append(x)
    return vals


# ============================================================
# Main
# ============================================================
def main():
    data, labels, missing = [], [], []
    print(f"Figure 2: per-scene RMSE, x{SCALE}, {MODE}")
    print(f"  primary metrics dir : {METRICS_DIR}")
    print()
    print(f"  {'method':10}{'n':>7}{'median':>10}{'Q1':>10}{'Q3':>10}  source")

    for m in METHODS:
        path, origin = find_metrics(m)
        if path is None:
            missing.append(m)
            print(f"  {LABEL.get(m, m):10}{'-':>7}{'-':>10}{'-':>10}{'-':>10}  MISSING")
            continue
        vals = read_rmse(path)
        if not vals:
            missing.append(m)
            print(f"  {LABEL.get(m, m):10}{0:>7}{'-':>10}{'-':>10}{'-':>10}  EMPTY")
            continue
        s = sorted(vals)
        q1 = s[len(s) // 4]
        q3 = s[(3 * len(s)) // 4]
        print(f"  {LABEL.get(m, m):10}{len(vals):>7}{median(vals):>10.4f}"
              f"{q1:>10.4f}{q3:>10.4f}  {origin}")
        if origin != "primary":
            print(f"      -> {path}")
        data.append(vals)
        labels.append(LABEL.get(m, m))

    if not data:
        sys.exit("[ERROR] no metrics CSV found. Set SR_METRICS_DIR.")

    ns = {len(d) for d in data}
    if len(ns) > 1:
        print(f"\n  [WARN] scene counts differ across methods: {sorted(ns)}")
        print("         the boxes are then not a paired comparison; check that "
              "every method was evaluated on the same test set.")

    fig, ax = plt.subplots(figsize=(9 + 0.6 * max(0, len(labels) - 7), 4.2))
    bp = ax.boxplot(data, patch_artist=True, showfliers=False, widths=0.6,
                    medianprops=dict(color="#ff7f0e", linewidth=1.6),
                    whiskerprops=dict(color="black"), capprops=dict(color="black"),
                    boxprops=dict(edgecolor="black"))
    for box in bp["boxes"]:
        box.set(facecolor="#dbe9f5")
    ax.set_xticks(range(1, len(labels) + 1))
    ax.set_xticklabels(labels)
    ax.set_ylabel(f"Per-scene RMSE (x{SCALE}, {MODE.replace('bandwise', 'band-wise')})")
    ax.set_ylim(0, float(YLIM_TOP) if YLIM_TOP else 0.058)
    ax.yaxis.grid(True, color="0.9")
    ax.set_axisbelow(True)
    for s in ["top", "right"]:
        ax.spines[s].set_visible(False)
    plt.tight_layout()
    plt.savefig(OUT, dpi=200, bbox_inches="tight")

    print(f"\n[SAVED] {OUT}  ({len(labels)} boxes)")
    if missing:
        print(f"[WARN] not plotted: {', '.join(missing)}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
