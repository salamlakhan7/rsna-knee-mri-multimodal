import os, time, shutil, tempfile
import numpy as np, pandas as pd, streamlit as st
import config, db, ui, mri_pipeline as mri
from report_labeler import label_report, COVERED

user = ui.require_login()
st.title("Live Explore"); ui.banner()
st.caption(f"Signed in as {user['name']}. Every test you run is saved to your history.")
tab_r, tab_m = st.tabs(["Report to findings", "MRI study test"])

# ----------------------------------------------------------------- report tab
with tab_r:
    sample = st.selectbox("Sample report (written for this demo)", ["(none)"] + list(config.SAMPLES))
    s = config.SAMPLES.get(sample)
    text = st.text_area("Radiology report", value=s["text"] if s else "", height=170, key=f"rep_{sample}")
    known = st.multiselect("Known findings (optional, to compare)", config.LABELS,
                           default=s["findings"] if s else [], key=f"kn_{sample}")
    keep = st.checkbox("Also save the report text in my history", value=False)
    if st.button("Analyze report", type="primary", disabled=not text.strip()):
        t0 = time.time(); lang = ui.detect_lang(text)
        out, why = label_report(text, trace=True)
        if lang not in COVERED:
            st.warning(f"Detected language: {lang}. The rules cover English, Spanish, German and Turkish, so findings in other languages may be missed.")
        df = pd.DataFrame({"Finding": config.LABELS, "Detected": ["Yes" if out[l] else "No" for l in config.LABELS],
                           "Triggering clause": [why.get(l, "") for l in config.LABELS],
                           "Labeler F1 on gold": [config.RULE_F1[l] for l in config.LABELS]})
        st.dataframe(df, hide_index=True)
        detected = [l for l in config.LABELS if out[l]]
        res = {"detected": detected}
        if known:
            hit = [l for l in known if out[l]]
            res.update(known=known, matched=hit, extra=[l for l in detected if l not in known])
            st.success(f"Matched {len(hit)} of {len(known)} known findings. Extra detections: {len(res['extra'])}.")
        db.log_run(user["id"], "report", {"language": lang, "n_chars": len(text), "source": sample if s else "pasted",
                                          "text": text[:2000] if keep else None}, res, (time.time() - t0) * 1000)
        st.caption("Saved to your history.")

# ------------------------------------------------------------------ MRI tab
@st.cache_resource(show_spinner="Loading model...")
def get_model(): return mri.load_model(config.WEIGHTS_PATH)

with tab_m:
    st.write("Upload a ZIP with the DICOM files of **one** knee MRI study (all series). Nothing is stored except the scores.")
    zf = st.file_uploader("DICOM study (.zip)", type=["zip"])
    known_m = st.multiselect("Known findings (optional, to compare)", config.LABELS, key="kn_mri")
    if zf and st.button("Run model", type="primary"):
        if not os.path.exists(config.WEIGHTS_PATH):
            st.error("Model weights not found. Place resnet18_attn.pt in the models/ folder."); st.stop()
        t0 = time.time(); tmp = tempfile.mkdtemp()
        try:
            with st.spinner("Reading DICOM files..."):
                mri.extract_zip(zf, tmp); series = mri.scan_series(tmp); picks = mri.pick_series(series)
                if all(p is None for p in picks):
                    st.error("No usable series found (need sagittal, coronal or axial series with 8+ slices)."); st.stop()
                arr, used = mri.build_study(series, picks)
            st.write("Series used"); st.dataframe(pd.DataFrame(used), hide_index=True)
            if len(used) < 3: st.warning("One or more planes are missing; the model was trained with all three, so accuracy will be lower.")
            cols = st.columns(3)
            for i, c in enumerate(cols):
                if picks[i] is not None: c.image(arr[i, mri.K // 2], caption=mri.PLANES[i], clamp=True)
            with st.spinner("Running model..."):
                probs = mri.predict(get_model(), arr)
        except ValueError as e:
            st.error(str(e)); st.stop()
        finally:
            shutil.rmtree(tmp, ignore_errors=True)
        df = pd.DataFrame({"Finding": config.LABELS, "Score": probs.round(3),
                           "Gold AUC": [config.IMAGE_AUC[l] for l in config.LABELS],
                           "Reliability": [config.reliability(config.IMAGE_AUC[l]) for l in config.LABELS]})
        st.dataframe(df, hide_index=True, column_config={"Score": st.column_config.ProgressColumn("Score", min_value=0.0, max_value=1.0, format="%.2f")})
        st.caption("Scores are relative model outputs (macro AUC 0.69 on 58 studies), not probabilities of disease.")
        res = {"scores": {l: float(p) for l, p in zip(config.LABELS, probs)}}
        if known_m:
            others = [p for l, p in zip(config.LABELS, probs) if l not in known_m]
            mk = float(np.mean([p for l, p in zip(config.LABELS, probs) if l in known_m]))
            res.update(known=known_m, mean_known=mk, mean_others=float(np.mean(others)) if others else None)
            st.info(f"Average score of your known findings: {mk:.2f} vs {res['mean_others']:.2f} for the others (higher is better)." if others else f"Average score of known findings: {mk:.2f}")
        db.log_run(user["id"], "mri", {"n_series_used": len(used), "series": used}, res, (time.time() - t0) * 1000)
        st.caption("Saved to your history.")
