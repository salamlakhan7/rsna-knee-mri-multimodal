import streamlit as st
import config

st.title("Contact")
c = config.CONTACT
st.markdown(f"**{c['name']}**\n\n:material/mail: {c['email']}\n\n:material/code: [GitHub]({c['github']})")
if c["linkedin"]: st.markdown(f":material/work: [LinkedIn]({c['linkedin']})")
st.caption("Built as the Sprint 03 project for Nolyth (Deep Learning, NLP and Transformers).")
