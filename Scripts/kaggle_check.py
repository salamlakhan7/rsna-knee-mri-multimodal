# RSNA Knee - Day 1 environment & data check
# Paste into ONE Kaggle notebook cell (GPU on, competition data added as Input) and run.
# Then copy the full output back to Claude.

import os, glob, time, random, subprocess, shutil, importlib
import numpy as np
import pandas as pd

print("=" * 60, "\n1. ENVIRONMENT")
try:
    print(subprocess.run(["nvidia-smi", "-L"], capture_output=True, text=True).stdout.strip() or "no nvidia-smi")
except Exception as e:
    print("nvidia-smi failed:", e)
try:
    import torch
    print("torch", torch.__version__, "| cuda available:", torch.cuda.is_available(),
          "| gpus:", torch.cuda.device_count())
except Exception as e:
    print("torch problem:", e)
try:
    import psutil
    print("RAM total GB: %.1f" % (psutil.virtual_memory().total / 1e9))
except Exception:
    pass
print("Disk /kaggle/working free GB: %.1f" % (shutil.disk_usage("/kaggle/working").free / 1e9))
for lib in ["pydicom", "pylibjpeg", "openjpeg", "gdcm", "cv2", "timm", "sklearn"]:
    try:
        m = importlib.import_module(lib)
        print(f"  {lib}: OK", getattr(m, "__version__", ""))
    except Exception:
        print(f"  {lib}: NOT installed")

print("=" * 60, "\n2. DATA PATH")
print("/kaggle/input contains:", os.listdir("/kaggle/input"))
candidates = [
    "/kaggle/input/competitions/rsna-knee-abnormality-detection",
    "/kaggle/input/rsna-knee-abnormality-detection",
]
# also check one level down in every input folder (cheap, no deep walk)
for d in os.listdir("/kaggle/input"):
    candidates.append(f"/kaggle/input/{d}")
    if os.path.isdir(f"/kaggle/input/{d}"):
        for d2 in os.listdir(f"/kaggle/input/{d}"):
            candidates.append(f"/kaggle/input/{d}/{d2}")
ROOT = next((c for c in candidates if os.path.exists(f"{c}/train.csv")), None)
print("train.csv found in:", ROOT)
if ROOT is None:
    raise SystemExit("Competition data not attached. Use 'Add Input' and search for the competition.")
print("ROOT =", ROOT)
print("Files in ROOT:", sorted(os.listdir(ROOT)))

print("=" * 60, "\n3. LABELS & REPORTS")
LABELS = ["ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA", "Lateral OA",
          "PF OA", "Effusion", "Synovitis", "Baker's", "Contusion", "Fracture"]
train = pd.read_csv(f"{ROOT}/train.csv")
print("train shape:", train.shape)
print("columns:", list(train.columns))
present = [c for c in LABELS if c in train.columns]
print("label columns present:", len(present), "of 12")
labeled_mask = train[present].notna().all(axis=1)
print("studies with all 12 labels:", int(labeled_mask.sum()), "| without:", int((~labeled_mask).sum()))
if labeled_mask.sum():
    prev = train.loc[labeled_mask, present].mean().round(3)
    print("prevalence among labeled studies:\n", prev.to_string())
if "Report" in train.columns:
    rep = train["Report"].fillna("")
    print("reports empty:", int((rep.str.len() == 0).sum()),
          "| median length (chars):", int(rep.str.len().median()))
    print("sample report (first 300 chars):\n", rep.iloc[0][:300])

print("=" * 60, "\n4. SERIES METADATA")
ts = pd.read_csv(f"{ROOT}/train_series.csv")
print("train_series shape:", ts.shape, "| studies:", ts.StudyInstanceUID.nunique())
print(ts.Anatomical_Plane.value_counts().to_string())
print("series per study (describe):\n", ts.groupby("StudyInstanceUID").size().describe().round(1).to_string())
print("Fluid_Sensitive x Fat_Suppression:\n", pd.crosstab(ts.Fluid_Sensitive, ts.Fat_Suppression).to_string())
test = pd.read_csv(f"{ROOT}/test.csv")
print("example test.csv rows:", len(test), "(real test has ~1300 studies)")

print("=" * 60, "\n5. DICOM DECODE TEST")
import pydicom
random.seed(0)
sample_series = ts.sample(min(60, len(ts)), random_state=0)
by_syntax, failures, times = {}, [], []
for _, r in sample_series.iterrows():
    folder = f"{ROOT}/train_series/{r.StudyInstanceUID}/{r.SeriesInstanceUID}"
    files = sorted(glob.glob(folder + "/*.dcm"))
    if not files:
        continue
    f = files[len(files) // 2]
    try:
        hdr = pydicom.dcmread(f, stop_before_pixels=True)
        syn = str(hdr.file_meta.TransferSyntaxUID.name)
    except Exception as e:
        failures.append((f, "header: " + str(e)[:80])); continue
    by_syntax.setdefault(syn, [0, 0])
    by_syntax[syn][0] += 1
    t0 = time.time()
    try:
        arr = pydicom.dcmread(f).pixel_array
        times.append(time.time() - t0)
        by_syntax[syn][1] += 1
    except Exception as e:
        failures.append((syn, str(e)[:100]))
print("transfer syntax -> [tested, decoded OK]:")
for k, v in by_syntax.items():
    print("  ", k, v)
print("decode failures:", len(failures))
for fl in failures[:5]:
    print("  ", fl)
if times:
    print("avg single-slice read+decode: %.1f ms" % (1000 * np.mean(times)))

print("=" * 60, "\n6. SERIES READ SPEED (one full series)")
r = ts.iloc[0]
folder = f"{ROOT}/train_series/{r.StudyInstanceUID}/{r.SeriesInstanceUID}"
files = sorted(glob.glob(folder + "/*.dcm"))
t0 = time.time()
shapes = []
for f in files:
    try:
        shapes.append(pydicom.dcmread(f).pixel_array.shape)
    except Exception as e:
        shapes.append("ERR")
print(f"{len(files)} slices in {time.time() - t0:.2f}s | unique shapes: {set(map(str, shapes))}")
print("DONE. Copy everything above and send it back.")
