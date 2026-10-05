import streamlit as st
import results_data as R, ui

st.title("Results"); ui.banner()
st.subheader("Report labeling (scored on the 58 gold studies)")
st.dataframe(R.TEXT, hide_index=True)
st.caption("Rules + DistilBERT blend reached macro AUC 0.743. The rules cannot label the 12 studies in other languages; DistilBERT can, weakly.")
st.bar_chart(R.TEXT_F1, color=["#9AA9B2", "#E8A33D", "#0E7C86"])
st.subheader("Image models (gold macro AUC with 95% bootstrap interval)")
st.dataframe(R.IMAGE, hide_index=True)
st.bar_chart(R.IMAGE_AUC, color=["#9AA9B2", "#7BBFC6", "#0E7C86"])
st.caption("Averaging two models did not help (0.685). The Kaggle public leaderboard score of the final model is 0.730.")
st.subheader("Limitations")
st.markdown("- Only 58 gold studies: differences such as 0.672 vs 0.692 are within the noise, and per-finding scores are rough.\n"
            "- Gold labels can disagree with report text, so report-derived labels are a noisy proxy.\n"
            "- Rule labeler covers four languages; other languages rely on the transformer.\n"
            "- The 0.5 threshold is untuned: scores are relative rankings, not probabilities of disease.\n"
            "- Medial meniscus, synovitis, MCL and fracture are the weakest findings.")
