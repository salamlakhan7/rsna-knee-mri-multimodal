# Project log

One entry per work session: what I did, what I found, what is next.

## 2026-10-02
- Joined RSNA Knee Abnormality Detection (Kaggle). Environment checks: 2x Tesla T4, 4 CPU, 33.7 GB RAM, ~19.5 GB output limit.
- Data audit: 4,407 train studies, 24,371 series; only 58 studies have the 12 labels; all 4,407 have a report in 9+ languages; test has no reports.
- NB1 v1 rule labeler: mean F1 0.536 vs 58 gold. Problems found: 27% of studies in uncovered languages got false zeros; Greek micro-sign (U+00B5); cartilage/OA missed; grade-1 sprains counted as tears.
- NB1 v2: Unicode NFKC, tear-only ligament/meniscus rules, cartilage+compartment OA rules, uncovered languages left unlabeled (NaN).
  Result on covered gold (n=46): mean F1 0.696 (half A 0.723, half B 0.652).
- Next: NB2 text models (TF-IDF+LR vs multilingual transformer), scored on the same gold studies.
