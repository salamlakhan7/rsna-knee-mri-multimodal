# RSNA Knee MRI: report-supervised abnormality detection

Sprint 03 project (Nolyth: Deep Learning, NLP & Transformers) built on the Kaggle
[RSNA Knee Abnormality Detection](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection) competition.

> Research and learning project. Not a medical device and not for clinical use.

**Live demo:** [KneeScope](https://rsna-knee-mri-multimodal.streamlit.app)

## Problem
Predict 12 knee findings per MRI study (ACL, MCL, medial/lateral meniscus, medial/lateral/PF osteoarthritis,
effusion, synovitis, Baker's cyst, contusion, fracture). Metric: macro-averaged ROC AUC.
Test studies contain images only; training studies also have a free-text radiology report (multilingual).

## Key data facts (from my audit)
- 4,407 training studies, 24,371 series (about 5.5 per study), 3 planes (sagittal, coronal, axial).
- Only **58** studies carry real labels. 4,349 have a report only.
- Reports in at least 9 detected languages (en, es, tr, hr, el, de, bg, nl, fr).
- The test set has no reports, so the final model must be image-only.

## Approach
1. **Report labeler** turns reports into pseudo-labels for unlabeled studies.
   Compare rules vs TF-IDF + logistic regression vs a multilingual transformer, all scored on the 58 gold studies.
2. **Preprocessing** picks up to 3 fluid-sensitive series per study (one per plane), 16 central slices, 192x192.
3. **Image model** compares a small custom CNN with a pretrained CNN (transfer learning), mean vs attention pooling over slices.
4. **Offline Kaggle submission** (Internet off, 9 h limit) and a **Streamlit demo**.
## Status
| Step | Notebook | Status |
|---|---|---|
| Report audit + rule labeler | `notebooks/01_report_audit_and_rule_labeler.ipynb` | done |
| Text models (TF-IDF vs transformer) | `notebooks/02_text_models.ipynb` | done |
| Preprocessing | `notebooks/03_preprocess.ipynb` | done |
| Image models | `notebooks/04_image_models.ipynb` | done |
| Offline inference | `notebooks/05_inference.ipynb` | done |
| Web demo | `demo/` | done |
| Report | `report/` | todo |

## Results
| Component | Metric | Value |
|---|---|---|
| Rule labeler v1 | mean F1 vs 58 gold | 0.536 |
| Rule labeler v2 (en/es/de/tr, n=46 gold) | mean F1 | 0.696 |
| TF-IDF + LR / DistilBERT (covered languages) | mean F1 | 0.629 / 0.616 |
| DistilBERT on the 12 other-language gold studies | mean F1 | 0.299 (rules: 0) |
| Rules + DistilBERT blend | macro AUC on gold | 0.743 |
| CNN scratch / ResNet18 mean / ResNet18 attention | macro AUC on gold | 0.631 / 0.672 / 0.692 |
| Final model on the Kaggle public leaderboard | macro AUC | 0.730 |

Honest notes: gold labels can disagree with report text, so the reports are noisy proxies. With 58 gold studies,
per-finding scores for rare findings are very noisy, and the 95% interval of the final model on gold is 0.642 to 0.742.

## Repository layout
```
notebooks/   Kaggle notebooks exported as .ipynb (outputs cleared)
scripts/     standalone code and environment checks
docs/        project guide
results/     metric tables and figures
demo/        Streamlit app
report/      final write-up
PROJECT_LOG.md   dated work log
```

## How to run
1. Join the competition on Kaggle and accept the rules.
2. Create a Kaggle notebook, add the competition data as input, enable GPU (T4), and paste/import a notebook from `notebooks/`.
3. Pseudo-labels and pip wheels are kept in a private Kaggle dataset (`rsna-knee-pseudo-labels`).
   Competition data is not included in this repository.

## Data and license note
No competition images, DICOMs or full reports are stored here. Follow the competition rules for data use.
