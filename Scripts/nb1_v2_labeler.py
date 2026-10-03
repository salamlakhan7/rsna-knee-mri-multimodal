# NB1 v2 - replace your old blocks 4, 5 and 7 with these (blocks 1-3 stay as they are).
# Needs: `train`, `LABELS`, `gold_mask` from block 1 and `train["lang"]` from block 2.

# %% 4v2. Rule labeler v2
import re, unicodedata
import numpy as np, pandas as pd

COVERED = {"en", "es", "de", "tr"}      # languages the rules were written for

def norm(t):
    t = unicodedata.normalize("NFKC", str(t))      # fixes the Greek micro sign and similar lookalikes
    t = t.replace("\u0130", "i")                   # Turkish dotted capital I
    t = t.lower().replace("\u0307", "")
    return t

NEG = re.compile(r"\b(no|not|without|absent|intact|normal|negative|unremarkable|sin|ausencia|keine?n?|ohne|unauff\w+|yok)\b"
                 r"|izlenmem|saptanmam|g\u00f6r\u00fclmem|g\u00f6zlenmem|bulunmam|normaldir|do\u011fal|"
                 r"no hay|no se observa|sin signos|conservad|l\u00edmites normales", re.I)
SPLIT = re.compile(r"[.;:\n>*]| but | however | pero | sin embargo | aber | ancak | fakat |, and (?=the )", re.I)

TEAR = re.compile(r"\btear|\btorn|rupture|ruptur|avulsion|complete disruption|rotura|ruptura|desgarro|\briss|"
                  r"y\u0131rt|r\u00fcpt\u00fcr|kopma|kopuk|\bk\u0131r\u0131k\w* (bant|ba\u011f)", re.I)
SOFT = re.compile(r"sprain|grade (1|i)\b|interstitial|mucoid|increased signal|signal (change|alteration)|"
                  r"distensi\u00f3n|esguince|zerrung|sinyal art", re.I)
LIG_INJ = re.compile(r"sprain|injur|grade (2|3|ii|iii)\b|esguince|lesi\u00f3n|verletz|zerrung|zedelen|hasar|\bpartial", re.I)

ACL = re.compile(r"anterior cruciate|\bacl\b|cruzado anterior|\blca\b|vordere[sn]? kreuzband|\bvkb\b|\u00f6n \u00e7apraz", re.I)
MCL = re.compile(r"medial collateral|\bmcl\b|colateral (medial|tibial)|\blcm\b|innenband|mediale[sn]? kollateral|"
                 r"i\u00e7 yan ba\u011f|medial kollateral", re.I)
MENISC = re.compile(r"menisc|menisk|men\u00edsc", re.I)
MEDIAL = re.compile(r"medial|interno|mediale|innen|\bi\u00e7\b|medyal|medij", re.I)
LATERAL = re.compile(r"lateral|externo|laterale|au(\u00df|ss)en|d\u0131\u015f|lateral", re.I)
PATELLA = re.compile(r"patell|trochle|rotul|patelof|femoropatel|retropatel|troclea|trochlea|rotula|patella|femoropatellar|"
                     r"femoro-patellar|patelofemoral|patellofemoral", re.I)
CART = re.compile(r"cartilag|chondral|chondr|cart\u00edlago|condral|condr|knorpel|k\u0131k\u0131rdak|hrskavic", re.I)
CDEF = re.compile(r"loss|thinning|defect|lesion|fissur|ulcer|erosion|denud|delaminat|full[- ]thickness|partial[- ]thickness|"
                  r"grade (2|3|4|ii|iii|iv)\b|chondromalacia|chondrosis|degenerat|malacia|\u00falcera|defecto|p\u00e9rdida|"
                  r"adelgaz|fisura|erosi\u00f3n|verlust|defekt|ausd\u00fcnn|kay\u0131p|kayb|incel|dejener|stanjen|lezij|osteophyt|osteofit", re.I)
OAW = re.compile(r"osteoarth|arthrosis|artrosis|arthrose|gonarthrose|gonartroz|osteoartrit|gonartroz|femorotibial osteo", re.I)
COMPT = re.compile(r"compartment|compartimento|kompartiment|kompartment|b\u00f6l\u00fcm|plateau|condyle|c\u00f3ndilo|kondyl|kondil|"
                   r"femorotibial|tibiofemoral|tibial", re.I)

PRES = {
    "Effusion": re.compile(r"effusion|derrame|erguss|ef\u00fczyon|s\u0131v\u0131 miktar\u0131 (hafif )?art|eklem s\u0131v\u0131s\u0131 art|hydrops", re.I),
    "Synovitis": re.compile(r"synovitis|sinovitis|sinovit|synovialitis|synovial (thickening|inflamm|proliferat|hypertroph|reaction)|"
                            r"s\u0131nov\u0131yal (kal\u0131nla|inflam)|pigmented villonodular", re.I),
    "Baker's": re.compile(r"baker|popliteal cyst|poplite\w* cyst|quiste popl\u00edteo|quistes popl|kniekehlenzyste|poplitea[l]? kist", re.I),
    "Contusion": re.compile(r"contusion|contusi\u00f3n|kontusion|kont\u00fczyon|bone (marrow )?(oedema|edema|bruis)|edema \u00f3seo|"
                            r"knochenmark(\u00f6|oe)dem|kemik ili\u011fi \u00f6dem|kemik \u00f6dem|bone bruise|m\u00e9dula \u00f3sea.*edema", re.I),
    "Fracture": re.compile(r"fracture|fractura|fraktur|k\u0131r\u0131\u011f\u0131|k\u0131r\u0131k", re.I),
}

def label_report(text, trace=False):
    """Returns dict label->0/1; with trace=True also dict label->triggering clause."""
    out = {l: 0 for l in LABELS}
    why = {}
    ctx = None                                           # last compartment header seen: medial / lateral / pf
    def setl(k, clause):
        out[k] = 1
        why.setdefault(k, clause[:160])
    for clause in SPLIT.split(norm(text)):
        c = clause.strip()
        if not c:
            continue
        has_side_m, has_side_l, has_pf = MEDIAL.search(c), LATERAL.search(c), PATELLA.search(c)
        if COMPT.search(c) and not CART.search(c) and not TEAR.search(c):      # a header-like clause: remember it
            ctx = "pf" if has_pf else ("medial" if has_side_m else ("lateral" if has_side_l else ctx))
        if NEG.search(c):
            continue
        # ligaments and menisci need a real tear (grade 1 / sprain stays negative)
        if TEAR.search(c) or not SOFT.search(c):
            if ACL.search(c) and TEAR.search(c): setl("ACL", c)
            if MENISC.search(c) and TEAR.search(c):
                if has_side_m: setl("Medial Meniscus", c)
                if has_side_l: setl("Lateral Meniscus", c)
        if MCL.search(c) and (TEAR.search(c) or LIG_INJ.search(c)): setl("MCL", c)
        # cartilage / osteoarthritis per compartment
        cart_hit = (CART.search(c) and CDEF.search(c)) or OAW.search(c)
        if cart_hit and not (MENISC.search(c) and not CART.search(c)):
            side = "pf" if has_pf else ("medial" if has_side_m else ("lateral" if has_side_l else ctx))
            if side == "pf": setl("PF OA", c)
            elif side == "medial": setl("Medial OA", c)
            elif side == "lateral": setl("Lateral OA", c)
        for k, rx in PRES.items():
            if rx.search(c): setl(k, c)
    return (out, why) if trace else out

def why(row_idx, label):
    """Debug helper: show which clause fired a label (or that none did) next to the gold value."""
    r = train.loc[row_idx]
    o, w = label_report(r.Report, trace=True)
    print(f"[{r.lang}] gold={r.get(label)} pred={o[label]}\n  trigger: {w.get(label, '(none)')}\n  report : {str(r.Report)[:600]!r}")

# %% 5v2. Run labeler, mask uncovered languages, evaluate
from sklearn.metrics import precision_recall_fscore_support
pred = pd.DataFrame([label_report(t) for t in train.Report], index=train.index)
covered = train.lang.isin(COVERED)
pred.loc[~covered, :] = np.nan                      # unlabeled, NOT negative
print("covered studies:", int(covered.sum()), "| left unlabeled:", int((~covered).sum()))
print("predicted prevalence (covered only):\n", pred[covered].mean().round(3).to_string())

def evaluate(g, name):
    rows = []
    for l in LABELS:
        y, p = g[l].astype(int), pred.loc[g.index, l].astype(int)
        pr, rc, f, _ = precision_recall_fscore_support(y, p, average="binary", zero_division=0)
        rows.append(dict(finding=l, precision=round(pr, 2), recall=round(rc, 2), f1=round(f, 2),
                         gold_pos=int(y.sum()), pred_pos=int(p.sum())))
    res = pd.DataFrame(rows); print(f"--- {name} (n={len(g)}) mean F1 = {res.f1.mean():.3f}"); return res

g_all = train[gold_mask]
g_cov = g_all[g_all.lang.isin(COVERED)]
print(evaluate(g_cov, "gold, covered languages").to_string(index=False))
# overfitting check: tune on one half, look at the other half
evaluate(g_cov.iloc[::2], "half A (tune here)"); evaluate(g_cov.iloc[1::2], "half B (check here)")
print("gold studies in uncovered languages:", int((~g_all.lang.isin(COVERED)).sum()), "-> evaluate these later with the transformer")

# %% 6v2. Debug examples (edit labels/indices) - look at what fires and why
wrong = g_cov[g_cov["Effusion"].astype(int) != pred.loc[g_cov.index, "Effusion"].astype(int)]
for idx in wrong.index[:4]:
    why(idx, "Effusion")

# %% 7v2. Save v2 pseudo-labels (NaN = not labeled yet)
out = train[["StudyInstanceUID", "lang"]].join(pred)
out["covered"] = covered.values
out.to_csv("/kaggle/working/pseudo_labels_v2.csv", index=False)
print("saved", out.shape)
