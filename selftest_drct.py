"""Pre-flight check for the DRCT-based baseline.

Usage:
    python selftest_drct.py

The check imports the release pipeline, compares parameter counts for the
recent baselines, and runs a small forward/backward pass for DRCT.
"""

import os
import time

import torch

import Final_version6_drct as M


def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def check_parameter_counts():
    print("[1] Parameter counts")
    rows = {}
    for name in ("drct", "hat", "dat", "edsr", "rcan"):
        model = M.build_model(name, scale=4, channels=5)
        rows[name] = count_params(model)
        print(f"    {name:>5s}: {rows[name] / 1e6:.3f} M")
    if not (1e6 < rows["drct"] < 10e6):
        raise RuntimeError(f"Unexpected DRCT parameter count: {rows['drct']}")


def check_forward_backward(device, method="drct"):
    print(f"[2] Forward/backward check: {method} on {device}")
    scale = int(os.environ.get("SR_SELFTEST_SCALE", "4"))
    lr_patch = int(os.environ.get("SR_SELFTEST_LR_PATCH", "24"))
    model = M.build_model(method, scale=scale, channels=5).to(device)
    model.train()
    x = torch.rand(1, 5, lr_patch, lr_patch, device=device)
    y = torch.rand(1, 5, lr_patch * scale, lr_patch * scale, device=device)
    start = time.time()
    out = model(x)
    loss = torch.nn.functional.l1_loss(out, y)
    loss.backward()
    elapsed = time.time() - start
    if not torch.isfinite(out).all():
        raise RuntimeError("DRCT output contains NaN or Inf")
    if not torch.isfinite(loss):
        raise RuntimeError("DRCT loss is NaN or Inf")
    msg = f"    output={tuple(out.shape)}, loss={loss.item():.6f}, time={elapsed:.2f}s"
    if device == "cuda":
        msg += f", peak={torch.cuda.max_memory_allocated() / (1024 ** 3):.2f} GiB"
    print(msg)


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"DRCT shape: dim={M.DRCT_DIM} groups={M.DRCT_GROUPS} depth={M.DRCT_DEPTH} growth={M.DRCT_GROWTH} heads={M.DRCT_HEADS}")
    check_parameter_counts()
    check_forward_backward(device)
    print("\nDRCT self-test completed.")


if __name__ == "__main__":
    main()
