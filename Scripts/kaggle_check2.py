# RSNA Knee - Day 1 check #2 (label coverage, languages, DICOM formats, decoders)
# Run in a Kaggle notebook with GPU on and Internet ON. Send the full output back to Claude.

import os, sys, time, subprocess
import pandas as pd
import pydicom
from concurrent.futures import ThreadPoolExecutor

ROOT = "/kaggle/input/competitions/rsna-knee-abnormality-detection"
LABELS = ["ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA", "Lateral OA",
          "PF OA", "Effusion", "Synovitis", "Baker's", "Contusion", "Fracture"]
print("CPU cores:", os.cpu_count())

print("=" * 60, "\nA. LABEL COVERAGE")
train = pd.read_csv(f"{ROOT}/train.csv")
print("non-null count per label:\n", train[LABELS].notna().sum().to_string())
print("studies with at least one label:", int(train[LABELS].notna().any(axis=1).sum()))
print("values seen in label columns:", sorted(train[LABELS].stack().unique().tolist()))

print("=" * 60, "\nB. REPORT LANGUAGES")
try:
    from langdetect import detect, DetectorFactory
except ImportError:
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "langdetect"])
    from langdetect import detect, DetectorFactory
DetectorFactory.seed = 0

def lang(t):
    try:
        return detect(str(t)[:500])
    except Exception:
        return "unk"

train["lang"] = train.Report.map(lang)
print(train.lang.value_counts().head(15).to_string())
labeled = train[LABELS].notna().all(axis=1)
print("languages among the fully labeled studies:\n", train.loc[labeled, "lang"].value_counts().to_string())

print("=" * 60, "\nC. INSTALL EXTRA DICOM DECODERS (needs Internet ON)")
r = subprocess.run([sys.executable, "-m", "pip", "install", "-q",
                    "pylibjpeg", "pylibjpeg-libjpeg", "pylibjpeg-openjpeg"],
                   capture_output=True, text=True)
print("pip install return code:", r.returncode)
if r.returncode != 0:
    print(r.stderr[-300:])
# keep wheels for the offline submission notebook later
r2 = subprocess.run([sys.executable, "-m", "pip", "download", "-q",
                     "pylibjpeg", "pylibjpeg-libjpeg", "pylibjpeg-openjpeg", "-d", "/kaggle/working/wheels"],
                    capture_output=True, text=True)
print("wheel download return code:", r2.returncode)

print("=" * 60, "\nD. HEADER SCAN OF ~6000 SERIES (transfer syntax, slices, image size)")
ts = pd.read_csv(f"{ROOT}/train_series.csv")
sub = ts.sample(min(6000, len(ts)), random_state=1)

def probe(row):
    folder = f"{ROOT}/train_series/{row.StudyInstanceUID}/{row.SeriesInstanceUID}"
    try:
        fs = sorted(os.listdir(folder))
        path = f"{folder}/{fs[0]}"
        h = pydicom.dcmread(path, stop_before_pixels=True)
        return dict(n=len(fs), syntax=str(h.file_meta.TransferSyntaxUID.name),
                    size=f"{int(h.Rows)}x{int(h.Columns)}", path=path)
    except Exception as e:
        return dict(n=0, syntax="ERR " + str(e)[:60], size="?", path=folder)

t0 = time.time()
with ThreadPoolExecutor(32) as ex:
    res = pd.DataFrame(list(ex.map(probe, list(sub.itertuples(index=False)))))
print("scan took %.0fs" % (time.time() - t0))
print("transfer syntaxes:\n", res.syntax.value_counts().to_string())
print("slices per series:\n", res.n.describe().round(1).to_string())
print("top image sizes:\n", res["size"].value_counts().head(8).to_string())

print("=" * 60, "\nE. DECODE ONE EXAMPLE OF EACH NON-STANDARD SYNTAX")
for syn in res.syntax.unique():
    if syn == "Explicit VR Little Endian" or syn.startswith("ERR"):
        continue
    p = res[res.syntax == syn].path.iloc[0]
    try:
        a = pydicom.dcmread(p).pixel_array
        print("OK  ", syn, a.shape, a.dtype)
    except Exception as e:
        print("FAIL", syn, str(e)[:150])
print("DONE. Copy everything above and send it back.")
