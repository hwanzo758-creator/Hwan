"""
Environment check for Final_version5_revision.py
Run on your server:   python check_env.py

It verifies every third-party library the SR pipeline imports, prints versions,
and reports GPU / CUDA availability. Exit code is non-zero if anything is missing.
"""
import importlib
import sys

# (import name, pip package name) pairs used by Final_version5_revision.py
REQUIRED = [
    ("torch", "torch"),
    ("cv2", "opencv-python"),
    ("numpy", "numpy"),
    ("pandas", "pandas"),
    ("matplotlib", "matplotlib"),
    ("skimage", "scikit-image"),
    ("openpyxl", "openpyxl"),
]

print("Python:", sys.version.split()[0])
print("-" * 60)

missing = []
for mod_name, pip_name in REQUIRED:
    try:
        m = importlib.import_module(mod_name)
        ver = getattr(m, "__version__", "unknown")
        print(f"[OK]      {mod_name:12s} {ver:15s} (pip install {pip_name})")
    except Exception as e:
        missing.append(pip_name)
        print(f"[MISSING] {mod_name:12s} -> {e}  (pip install {pip_name})")

print("-" * 60)

# GPU / CUDA check (the training config assumes an NVIDIA GPU + AMP)
try:
    import torch
    print("torch CUDA available :", torch.cuda.is_available())
    print("torch CUDA version   :", torch.version.cuda)
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            name = torch.cuda.get_device_name(i)
            total = torch.cuda.get_device_properties(i).total_memory / (1024**3)
            print(f"  GPU {i}: {name}  ({total:.1f} GB)")
    else:
        print("  WARNING: no CUDA GPU detected — training will run on CPU (very slow).")
except Exception as e:
    print("torch not available for CUDA check:", e)

print("-" * 60)
if missing:
    print("RESULT: MISSING PACKAGES ->", ", ".join(missing))
    print("Install with:\n  pip install " + " ".join(missing))
    sys.exit(1)
else:
    print("RESULT: all required libraries are installed.")
    sys.exit(0)
