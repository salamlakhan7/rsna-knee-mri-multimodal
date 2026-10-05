import streamlit as st
import config, ui

st.title(f":material/orthopedics: {config.APP_NAME}")
st.subheader("Knee MRI abnormality detection supported by multilingual radiology reports")
ui.banner()
st.write("A research project built on the RSNA Knee Abnormality Detection challenge. It predicts 12 knee findings from an MRI "
         "study and turns free-text radiology reports in several languages into structured findings.")
c = st.columns(4)
c[0].metric("Findings predicted", "12"); c[1].metric("Training studies", "4,407")
c[2].metric("Gold-set macro AUC", "0.692"); c[3].metric("Kaggle public score", "0.730")
b = st.columns([1, 1, 4])
b[0].page_link("views/explore.py", label="Live Explore", icon=":material/science:")
b[1].page_link("views/technology.py", label="How it works", icon=":material/memory:")
st.divider(); st.markdown("#### How it works")
steps = [("1. Reports to labels", "Only 58 of 4,407 studies have real labels. A multilingual rule labeler and a DistilBERT classifier turn the reports into labels for the rest."),
         ("2. MRI preprocessing", "Per study: one fluid-sensitive series per plane (sagittal, coronal, axial), 16 central slices each, 192x192 pixels."),
         ("3. Image model", "A pretrained ResNet18 with attention pooling over slices predicts 12 scores per study."),
         ("4. Honest evaluation", "Scored on the 58 gold studies with bootstrap confidence intervals, then on the hidden Kaggle test set.")]
for col, (t, d) in zip(st.columns(4), steps):
    with col.container(border=True):
        st.markdown(f"**{t}**"); st.caption(d)
if not st.session_state.get("user"):
    st.info("Create a free account to test the models and keep a history of your tests.")
    st.page_link("views/access.py", label="Log in or sign up", icon=":material/login:")
