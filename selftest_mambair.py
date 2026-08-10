"""Pre-flight check for the MambaIR-based baseline (REVISION B5).

Run this BEFORE launching the full training, on the same machine/GPU:

    python selftest_mambair.py

It verifies, in order:
  1. the chunked selective scan matches a naive sequential recurrence;
  2. the model builds at x2 / x3 / x4 and its parameter count is in the same
     range as the other compact baselines;
  3. a forward+backward pass on a real training-size batch is finite;
  4. peak GPU memory and per-step time, so the full run can be costed.

Nothing is written to disk and no checkpoint is touched.
"""

import os
import time
import torch
import torch.nn.functional as F

import Final_version6_mambair as M


def check_scan():
    """Exactness of the chunked scan, at the decay magnitudes the model actually uses.

    The model floors log_a at MAMBA_LOG_A_FLOOR, so the hardest case the scan
    ever sees is a whole chunk sitting exactly on that floor. Both a realistic
    random case and that worst case are checked.
    """
    torch.manual_seed(0)
    b, d, l, n = 2, 6, 137, 4          # non-multiple of the chunk size on purpose
    ok = True

    def ref_scan(bu, log_a):
        h = torch.zeros(bu.shape[0], bu.shape[1], bu.shape[3], dtype=bu.dtype)
        out = []
        for t in range(bu.shape[2]):
            h = torch.exp(log_a[:, :, t]) * h + bu[:, :, t]
            out.append(h)
        return torch.stack(out, dim=2)

    # (a) realistic magnitudes: A = -(1..N) as initialised, dt ~ softplus(.) ~ 0.7
    bu = torch.randn(b, d, l, n, dtype=torch.float64)
    a_neg = -torch.arange(1, n + 1, dtype=torch.float64).view(1, 1, 1, n)
    dt = F.softplus(torch.randn(b, d, l, 1, dtype=torch.float64))
    log_a = (dt * a_neg).clamp(min=M.MAMBA_LOG_A_FLOOR)
    err_a = (M._selective_scan_chunked(bu, log_a, chunk=M.MAMBA_SCAN_CHUNK)
             - ref_scan(bu, log_a)).abs().max().item()

    # (b) worst case: every step pinned to the floor for a whole chunk
    log_a_w = torch.full((b, d, l, n), float(M.MAMBA_LOG_A_FLOOR), dtype=torch.float64)
    err_b = (M._selective_scan_chunked(bu, log_a_w, chunk=M.MAMBA_SCAN_CHUNK)
             - ref_scan(bu, log_a_w)).abs().max().item()

    print(f"[1a] chunked scan, realistic decay  : max abs err = {err_a:.3e}",
          "OK" if err_a < 1e-8 else "FAIL")
    print(f"[1b] chunked scan, worst-case decay : max abs err = {err_b:.3e}",
          "OK" if err_b < 1e-8 else "FAIL")
    print(f"     chunk={M.MAMBA_SCAN_CHUNK}, log_a floor={M.MAMBA_LOG_A_FLOOR}, "
          f"product={M.MAMBA_SCAN_CHUNK * abs(M.MAMBA_LOG_A_FLOOR):.1f} "
          f"< cumulative clamp {abs(M.MAMBA_LOGDECAY_FLOOR):.1f}")
    ok = err_a < 1e-8 and err_b < 1e-8

    # (c) the clamp must genuinely never bind inside a chunk
    s_max = M.MAMBA_SCAN_CHUNK * abs(M.MAMBA_LOG_A_FLOOR)
    binds = s_max >= abs(M.MAMBA_LOGDECAY_FLOOR)
    print(f"[1c] within-chunk clamp never binds :",
          "FAIL (clamp would bind)" if binds else "OK")
    return ok and not binds


def check_direction_merge():
    """The two scan directions are solved in one batched call for speed.

    This verifies that the merged path is numerically identical to solving each
    direction separately, which is what the previous (slower) implementation did.
    """
    torch.manual_seed(0)
    # float32 on purpose: SS2D.forward casts its internal state to float32, so a
    # float64 module would mismatch its own weights. Both paths run the identical
    # sequence of ops, so the difference should be at the level of float32 noise.
    mod = M.SS2D(dim=32, d_state=4, expand=1).eval()
    x = torch.randn(3, 32, 7, 5)

    with torch.no_grad():
        merged = mod(x)

        # reference: run each direction on its own, exactly as before the merge
        b, c, h, w = x.shape
        tokens = x.flatten(2).transpose(1, 2)
        xz = mod.in_proj(tokens)
        xs, z = xz.chunk(2, dim=-1)
        xs = xs.transpose(1, 2).reshape(b, mod.d_inner, h, w)
        xs = mod.act(mod.conv2d(xs))
        seq = xs.flatten(2)

        ref = None
        for k in range(mod.n_dir):
            src = seq if k == 0 else seq.flip(-1)
            bu, log_a, cmat = mod._prepare(src, k)
            hk = M._selective_scan_chunked(bu, log_a)
            yk = (hk * cmat.unsqueeze(1)).sum(dim=-1)
            yk = yk + mod.D[k].view(1, -1, 1) * src
            if k > 0:
                yk = yk.flip(-1)
            ref = yk if ref is None else ref + yk
        ref = ref / float(mod.n_dir)
        ref = mod.out_norm(ref.transpose(1, 2))
        ref = ref * F.silu(z)
        ref = mod.out_proj(ref).transpose(1, 2).reshape(b, c, h, w)

    err = (merged - ref).abs().max().item()
    print(f"[1d] batched-direction merge vs per-direction : max abs err = {err:.3e}",
          "OK" if err < 1e-6 else "FAIL")
    return err < 1e-6


def check_params():
    print("[2] parameter counts (joint, channels=5)")
    ok = True
    for scale in (2, 3, 4):
        row = {}
        for name in ("mambair", "hat", "dat", "edsr", "rcan"):
            try:
                row[name] = sum(p.numel() for p in
                                M.build_model(name, scale, channels=5).parameters()
                                if p.requires_grad)
            except Exception as e:
                row[name] = f"ERR {e}"
        print("    x%d  " % scale + "  ".join(f"{k}={v:,}" if isinstance(v, int) else f"{k}={v}"
                                              for k, v in row.items()))
        if isinstance(row["mambair"], int) and not (1e6 < row["mambair"] < 8e6):
            ok = False
    print("       -> MambaIR should sit in the same range as HAT/DAT.",
          "OK" if ok else "CHECK")
    return ok


def check_fwd_bwd(device):
    print("[3]/[4] forward+backward on a real training-size batch")
    ok = True
    for scale in (2, 3, 4):
        # The pipeline picks HR_PATCH_SIZE dynamically at run time (192 for this
        # dataset, as recorded in run_time_summary.csv of the previous runs), not
        # the 96 that sits in the module header. Measure at the real size or the
        # timing is optimistic by 4x in sequence length.
        hr_patch = int(os.environ.get("SR_SELFTEST_HR_PATCH", "192"))
        lr = hr_patch // scale
        model = M.build_model("mambair", scale, channels=5).to(device)
        x = torch.rand(M.BATCH_SIZE if hasattr(M, "BATCH_SIZE") else 16,
                       5, lr, lr, device=device)
        if device == "cuda":
            torch.cuda.reset_peak_memory_stats()
        t0 = time.time()
        with torch.autocast(device_type=device, dtype=torch.float16,
                            enabled=(device == "cuda")):
            y = model(x)
            loss = y.mean()
        loss.backward()
        if device == "cuda":
            torch.cuda.synchronize()
        dt = time.time() - t0
        gnorm = sum((p.grad.float() ** 2).sum().item()
                    for p in model.parameters() if p.grad is not None) ** 0.5
        peak = torch.cuda.max_memory_allocated() / 2 ** 30 if device == "cuda" else float("nan")
        good = torch.isfinite(y).all().item() and (gnorm == gnorm) and gnorm < 1e12
        ok &= good
        # 63 optimiser steps per epoch (PATCHES_PER_EPOCH 1000 / BATCH_SIZE 16),
        # 6 model instances per scale (1 joint + 5 band-wise), 50 epochs.
        hours = dt * 63 * 6 * 50 / 3600
        print(f"    x{scale}: HR {hr_patch} / LR {lr}x{lr} -> out {tuple(y.shape)} | "
              f"finite={torch.isfinite(y).all().item()} | grad_norm={gnorm:.3e} | "
              f"peak={peak:.2f} GiB | {dt*1000:.0f} ms/step | "
              f"~{hours:.0f} h for this scale "
              + ("OK" if good else "FAIL"))
        if device == "cuda" and peak > 0.75 * torch.cuda.get_device_properties(0).total_memory / 2**30:
            print("       WARNING: >75% of VRAM before the composite loss is added — "
                  "lower SR_MAMBA_D_STATE")
        del model, x, y, loss
        if device == "cuda":
            torch.cuda.empty_cache()
    return ok


if __name__ == "__main__":
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device = {dev}")
    print(f"MambaIR shape: dim={M.MAMBA_DIM} groups={M.MAMBA_GROUPS} depth={M.MAMBA_DEPTH} "
          f"d_state={M.MAMBA_D_STATE} expand={M.MAMBA_EXPAND}\n")
    r = [check_scan(), check_direction_merge(), check_params(), check_fwd_bwd(dev)]
    print("\nRESULT:", "ALL CHECKS PASSED" if all(r) else "SOME CHECKS FAILED")
    if all(r):
        print("\nNext:  set EXPERIMENT_PRESET = \"mambair_only\" and USER_SCALES = [2, 3, 4]")
