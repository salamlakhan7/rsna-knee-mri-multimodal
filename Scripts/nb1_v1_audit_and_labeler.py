# NB1 - Report audit + rule-based labeler v1 (en, es, de, tr)
# Paste each "# %%" block into its own Kaggle notebook cell. Internet ON (for langdetect), GPU not needed.
# Goal: (1) understand the reports, (2) build a first labeler, (3) measure it against the 58 gold studies.

# %% 1. Setup
import re, sys, subprocess
import numpy as np, pandas as pd
ROOT = "/kaggle/input/competitions/rsna-knee-abnormality-detection"
LABELS = ["ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA", "Lateral OA",
          "PF OA", "Effusion", "Synovitis", "Baker's", "Contusion", "Fracture"]
train = pd.read_csv(f"{ROOT}/train.csv")
gold_mask = train[LABELS].notna().all(axis=1)
print("studies:", len(train), "| gold (labeled):", int(gold_mask.sum()))

# %% 2. Audit: report length and language
try:
    from langdetect import detect, DetectorFactory
except ImportError:
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "langdetect"])
    from langdetect import detect, DetectorFactory
DetectorFactory.seed = 0
def lang(t):
    try: return detect(str(t)[:500])
    except Exception: return "unk"
train["lang"] = train.Report.map(lang)
train["n_chars"] = train.Report.str.len()
print(train.lang.value_counts().to_string())
print(train.n_chars.describe().round(0).to_string())
print("duplicate StudyInstanceUID:", int(train.StudyInstanceUID.duplicated().sum()))

# %% 3. Read the data (do this by hand: it is the most valuable step)
# Print 3 reports per language next to gold labels when available.
for lg in ["en", "es", "tr", "hr", "el"]:
    print("=" * 70, lg)
    for _, r in train[train.lang == lg].head(3).iterrows():
        pos = [l for l in LABELS if r[l] == 1] if r[LABELS].notna().all() else "(no gold)"
        print(r.Report[:500].replace("\n", " "), "\n   -> gold positives:", pos, "\n")

# %% 4. Rule-based labeler v1 (en, es, de, tr). Extend the patterns after reading more reports.
NEG = re.compile(r"\b(no|not|without|absent|intact|normal|negative|sin|ausencia|keine?n?|ohne|unauff\w+|yok)\b"
                 r"|izlenmem|saptanmam|g\u00f6r\u00fclmem|g\u00f6zlenmem|bulunmam", re.I)
SPLIT = re.compile(r"[.;:\n]| but | however | pero | sin embargo | aber | ancak | fakat ", re.I)
DAMAGE = re.compile(r"tear|torn|rupture|ruptur|partial|sprain|injur|lesion|rotura|ruptura|lesi\u00f3n|"
                    r"riss|verletz|y\u0131rt|r\u00fcpt\u00fcr|kopma|zedelen|hasar", re.I)
OA = re.compile(r"osteoarth|arthrosis|artrosis|arthrose|gonarthrose|gonartroz|osteoartrit|"
                r"cartilage (loss|thinning)|chondromalacia|condromalacia|degenerat", re.I)
MENISCUS = re.compile(r"menisc|menisk", re.I)
MEDIAL = re.compile(r"medial|interno|mediale|\bi\u00e7\b", re.I)
LATERAL = re.compile(r"lateral|externo|laterale|d\u0131\u015f", re.I)

STRUCT = {  # structure patterns that also need a DAMAGE word in the same clause
    "ACL": r"anterior cruciate|\bacl\b|cruzado anterior|\blca\b|vordere[sn]? kreuzband|\u00f6n \u00e7apraz",
    "MCL": r"medial collateral|\bmcl\b|colateral (medial|tibial)|ligamento lateral interno|innenband|"
           r"mediale[sn]? kollateral|i\u00e7 yan ba\u011f",
    "Medial Meniscus": r"medial menisc|menisco (interno|medial)|innenmeniskus|mediale[rn]? meniskus|i\u00e7 menisk|medial menisk",
    "Lateral Meniscus": r"lateral menisc|menisco (externo|lateral)|au(\u00df|ss)enmeniskus|laterale[rn]? meniskus|d\u0131\u015f menisk|lateral menisk",
}
PRESENCE = {  # presence of the word alone is enough
    "PF OA_site": r"patell?o?femor|femoro-?patel|femoropatel|retropatel|rotuliana|patelofemoral",
    "Effusion": r"effusion|derrame|erguss|ef\u00fczyon|eklem s\u0131v\u0131|joint fluid",
    "Synovitis": r"synovitis|sinovitis|sinovit|synovialitis|synovial (thickening|inflamm)",
    "Baker's": r"baker|popliteal cyst|quiste popl\u00edteo|kniekehlenzyste",
    "Contusion": r"contusion|contusi\u00f3n|kontusion|kont\u00fczyon|bone (marrow )?(oedema|edema|bruis)|edema \u00f3seo|"
                 r"knochenmark(\u00f6|oe)dem|kemik ili\u011fi \u00f6dem|kemik \u00f6dem",
    "Fracture": r"fracture|fractura|fraktur|k\u0131r\u0131k",
}
STRUCT = {k: re.compile(v, re.I) for k, v in STRUCT.items()}
PRESENCE = {k: re.compile(v, re.I) for k, v in PRESENCE.items()}

def label_report(text):
    out = {l: 0 for l in LABELS}
    for clause in SPLIT.split(str(text)):
        c = clause.strip()
        if not c or NEG.search(c):
            continue
        for k, rx in STRUCT.items():
            if rx.search(c) and DAMAGE.search(c):
                out[k] = 1
        for k, rx in PRESENCE.items():
            if rx.search(c):
                if k == "PF OA_site":
                    if OA.search(c): out["PF OA"] = 1
                else:
                    out[k] = 1
        if OA.search(c) and not MENISCUS.search(c):
            if MEDIAL.search(c): out["Medial OA"] = 1
            if LATERAL.search(c): out["Lateral OA"] = 1
    return out

pred = pd.DataFrame([label_report(t) for t in train.Report], index=train.index)
print(pred.mean().round(3).to_string())   # predicted prevalence over ALL studies

# %% 5. Evaluate against the 58 gold studies (per finding)
from sklearn.metrics import precision_recall_fscore_support
g = train[gold_mask]
rows = []
for l in LABELS:
    p, r, f, _ = precision_recall_fscore_support(g[l].astype(int), pred.loc[g.index, l], average="binary", zero_division=0)
    rows.append(dict(finding=l, precision=round(p, 2), recall=round(r, 2), f1=round(f, 2),
                     gold_pos=int(g[l].sum()), pred_pos=int(pred.loc[g.index, l].sum())))
res = pd.DataFrame(rows)
print(res.to_string(index=False))
print("mean F1:", res.f1.mean().round(3))

# %% 6. Look at the mistakes (this is where you improve the rules)
for l in ["ACL", "Medial Meniscus", "Effusion"]:   # change the finding to inspect others
    wrong = g[g[l].astype(int) != pred.loc[g.index, l]]
    print("=" * 70, l, "| errors:", len(wrong))
    for _, r in wrong.head(4).iterrows():
        print(f"[{r.lang}] gold={int(r[l])} pred={int(pred.loc[r.name, l])} :: {r.Report[:350]!r}\n")

# %% 7. Save the v1 pseudo-labels (you will overwrite them with better versions later)
out = train[["StudyInstanceUID", "lang"]].join(pred)
out.to_csv("/kaggle/working/pseudo_labels_v1.csv", index=False)
print("saved", out.shape)
