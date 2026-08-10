"""
Figure 2 — per-scene RMSE boxplot (×4, band-wise) including HAT and DAT.
Reads the per-scene metrics CSVs produced by Final_version5_revision.py and
draws the boxplot. No GPU required.

Usage: edit METRICS_DIR, then `python make_fig2_boxplot.py`.
"""
import csv, os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

METRICS_DIR = "/path/to/results/<run_name>/sr_logs"  # folder with metrics_bandwise_<method>_x4.csv
OUT = "Fig2_boxplot_rmse_x4_bandwise.png"
METHODS = [("bicubic", "Bicubic"), ("edsr", "EDSR"), ("rcan", "RCAN"),
           ("swinir", "SwinIR"), ("esrgan", "ESRGAN"), ("hat", "HAT"), ("dat", "DAT")]

data, labels = [], []
for m, lab in METHODS:
    fn = os.path.join(METRICS_DIR, f"metrics_bandwise_{m}_x4.csv")
    if not os.path.exists(fn):
        print("skip (missing):", lab); continue
    rows = list(csv.DictReader(open(fn, encoding="utf-8-sig")))
    vals = [float(r["RMSE"]) for r in rows
            if r.get("RMSE") not in (None, "", "nan") and float(r["RMSE"]) < 1.0]
    data.append(vals); labels.append(lab)

fig, ax = plt.subplots(figsize=(9, 4.2))
bp = ax.boxplot(data, patch_artist=True, showfliers=False, widths=0.6,
                medianprops=dict(color="#ff7f0e", linewidth=1.6),
                whiskerprops=dict(color="black"), capprops=dict(color="black"),
                boxprops=dict(edgecolor="black"))
for box in bp["boxes"]:
    box.set(facecolor="#dbe9f5")
ax.set_xticklabels(labels)
ax.set_ylabel("Per-scene RMSE (x4, band-wise)")
ax.set_ylim(0, 0.058)
ax.yaxis.grid(True, color="0.9"); ax.set_axisbelow(True)
for s in ["top", "right"]:
    ax.spines[s].set_visible(False)
plt.tight_layout()
plt.savefig(OUT, dpi=200, bbox_inches="tight")
print("saved", OUT)
