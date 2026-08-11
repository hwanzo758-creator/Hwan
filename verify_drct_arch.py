"""
Check that the DRCT definition in Make_Figures_9methods.py is the same network
that produced the trained checkpoints, WITHOUT needing PyTorch or a GPU.

It counts the parameters of the DRCT architecture analytically and compares the
result with params_trainable recorded in the training-history CSVs. If the two
agree for every scale and mode, the checkpoints will load with strict=True and
the figures will be generated from exactly the models used for Tables 3-7.

Usage:
    python verify_drct_arch.py [train_history_x2.csv train_history_x34.csv ...]

With no arguments it searches, in order:
    1. the directory holding this script
    2. ../05_drct_results/            (local Windows layout)
    3. $SR_OUTPUT_ROOT/*/sr_logs/     (server layout, any run name)
"""
from pathlib import Path
import csv
import os
import sys

# ---- DRCT hyperparameters (must match Make_Figures_9methods.py) ----
DIM, GROWTH, DEPTH, GROUPS, HEADS, MLP_RATIO = 96, 64, 5, 6, 4, 2.0
CHANNELS = 5


def conv(ci, co, k):
    return ci * co * k * k + co


def linear(i, o):
    return i * o + o


def layernorm(d):
    return 2 * d


def multihead_attention(d):
    # in_proj_weight (3d, d) + in_proj_bias (3d) + out_proj weight/bias
    return 3 * d * d + 3 * d + d * d + d


def swin_like_block(d, mlp_ratio=MLP_RATIO):
    hidden = int(d * mlp_ratio)
    return (layernorm(d) + multihead_attention(d) + layernorm(d)
            + linear(d, hidden) + linear(hidden, d))


def drct_group(dim=DIM, growth=GROWTH, depth=DEPTH):
    total = sum(conv(dim + i * growth, growth, 1) for i in range(depth))
    total += depth * swin_like_block(growth)
    total += conv(dim + depth * growth, dim, 1)   # fuse
    total += conv(dim, dim, 3)                    # conv
    return total


def upsampler(scale, nf=DIM):
    if scale == 2:
        return conv(nf, nf * 4, 3)
    if scale == 3:
        return conv(nf, nf * 9, 3)
    if scale == 4:
        return 2 * conv(nf, nf * 4, 3)
    raise ValueError(f"Unsupported scale: {scale}")


def drct_params(scale, channels=CHANNELS):
    return (conv(channels, DIM, 3)
            + GROUPS * drct_group()
            + conv(DIM, DIM, 3)
            + upsampler(scale)
            + conv(DIM, channels, 3))


def load_history(paths):
    rows = []
    for p in paths:
        with open(p, newline="", encoding="utf-8") as f:
            rows.extend(list(csv.DictReader(f)))
    return [r for r in rows if r.get("method") == "drct"]


def find_history_files():
    here = Path(__file__).resolve().parent
    results_root = Path(os.environ.get(
        "SR_OUTPUT_ROOT",
        "/home/whanjo/datasets/Python_file/SR_HR_LR_TEST/"
        "light_cabbage_training_dataset_SR_RESULTS",
    ))
    for candidates in (
        sorted(here.glob("train_history*.csv")),
        sorted((here.parent / "05_drct_results").glob("train_history*.csv")),
        sorted(results_root.glob("*/sr_logs/train_history*.csv")),
    ):
        hits = [p for p in candidates if p.exists()]
        if hits:
            return hits
    return []


def main():
    args = [Path(a) for a in sys.argv[1:]]
    if not args:
        args = find_history_files()
    args = [p for p in args if p.exists()]
    if not args:
        sys.exit(
            "[ERROR] no train_history CSV found.\n"
            "        Pass paths explicitly, e.g.\n"
            "          python verify_drct_arch.py "
            "$SR_OUTPUT_ROOT/<run>/sr_logs/train_history_all.csv"
        )

    rows = load_history(args)
    if not rows:
        sys.exit("[ERROR] no rows with method=drct in the given CSVs.")

    measured = {}
    for r in rows:
        key = (r["mode"], int(float(r["scale"])))
        v = int(float(r["params_trainable"]))
        measured[key] = max(measured.get(key, 0), v)

    print(f"DRCT: dim={DIM}, growth={GROWTH}, depth={DEPTH}, "
          f"groups={GROUPS}, heads={HEADS}, mlp_ratio={MLP_RATIO}")
    print(f"history files: {', '.join(p.name for p in args)}\n")
    print(f"  {'scale':6}{'mode':10}{'analytic':>13}{'checkpoint':>14}   verdict")

    ok = True
    for (mode, scale) in sorted(measured, key=lambda k: (k[1], k[0])):
        want = drct_params(scale, channels=5 if mode == "joint5ch" else 1)
        got = measured[(mode, scale)]
        match = want == got
        ok &= match
        extra = f"   (x5 = {want * 5:,})" if mode == "bandwise" else ""
        print(f"  x{scale:<5}{mode:10}{want:>13,}{got:>14,}   "
              f"{'MATCH' if match else 'MISMATCH'}{extra}")

    print()
    if ok:
        print("[OK] architecture identical to the trained models; "
              "checkpoints will load with strict=True.")
        return 0
    print("[FAIL] parameter counts differ. Do NOT trust figures generated from "
          "this definition - the checkpoint would only load with strict=False, "
          "leaving layers randomly initialised.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
