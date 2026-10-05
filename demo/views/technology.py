import streamlit as st
import ui

st.title("Technology"); ui.banner()
with st.expander("Data", expanded=True):
    st.markdown("- 4,407 training studies, 24,371 DICOM series, three imaging planes.\n- 58 studies carry the 12 labels (gold). All 4,407 have a multilingual report; test studies have images only.\n"
                "- Reports in at least nine detected languages (en, es, tr, hr, el, de, bg, nl, fr).")
with st.expander("Report labeler"):
    st.markdown("**Rules (v2):** keyword patterns for English, Spanish, German and Turkish, Unicode normalisation, negation handling, tear-only rules for "
                "ligaments and menisci, and cartilage rules per compartment for osteoarthritis.\n\n**TF-IDF + logistic regression:** character n-grams (3-5).\n\n"
                "**DistilBERT (multilingual):** encoder-only transformer, WordPiece tokenisation, fine-tuned on rule labels, so it can label languages the rules do not cover.")
with st.expander("Image model"):
    st.markdown("A ResNet18 (ImageNet weights, adapted to one channel) encodes each slice. Attention pooling weighs the 16 slices of a plane, the three plane "
                "vectors are concatenated, and a linear layer outputs 12 sigmoid scores. Training used soft labels from the report step and a weighted loss.")
with st.expander("Evaluation"):
    st.markdown("Macro-averaged ROC AUC over the 12 findings. The 58 gold studies were never used for training or model selection. "
                "Confidence intervals come from 1,000 bootstrap resamples.")
with st.expander("Stack"):
    st.markdown("Python, PyTorch, timm, Hugging Face Transformers, scikit-learn, pydicom, OpenCV, Streamlit, SQLAlchemy, SQLite (development) and Neon PostgreSQL (production).")
