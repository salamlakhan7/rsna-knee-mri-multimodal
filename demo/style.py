"""Presentation only: palette A (calm clinical). No application logic here."""
import streamlit as st

CSS = """
<style>
:root { --teal:#0E7C86; --navy:#0B2545; --line:#D3E4E7; }
h1, h2, h3, h4 { color: var(--navy); letter-spacing: -0.01em; }
.block-container { padding-top: 2.4rem; max-width: 1240px; }
[data-testid="stMetric"] { background:#fff; border:1px solid var(--line); border-radius:14px;
    padding:14px 18px; box-shadow:0 1px 3px rgba(11,37,69,.06); }
[data-testid="stMetricValue"] { color: var(--teal); }
div.stButton > button, div[data-testid="stFormSubmitButton"] > button { border-radius:10px; font-weight:600; }
div[class*="st-key-hero"] { background:linear-gradient(120deg,#0B2545 0%,#0E7C86 100%);
    border-radius:20px; padding:2.2rem 2.4rem; margin-bottom:1.2rem; }
div[class*="st-key-hero"] h1, div[class*="st-key-hero"] h2, div[class*="st-key-hero"] h3,
div[class*="st-key-hero"] p, div[class*="st-key-hero"] span { color:#fff !important; }
div[class*="st-key-hero"] a { background:rgba(255,255,255,.16); border-radius:10px; }
div[class*="st-key-card"] { background:#fff; border:1px solid var(--line); border-top:4px solid var(--teal);
    border-radius:14px; padding:1rem 1.1rem; height:100%; }
div[class*="st-key-brand"] { background:linear-gradient(160deg,#0B2545,#0E7C86); border-radius:18px; padding:1.6rem; }
div[class*="st-key-brand"] h3, div[class*="st-key-brand"] p, div[class*="st-key-brand"] li,
div[class*="st-key-brand"] span { color:#fff !important; }
</style>
"""

def inject():
    st.markdown(CSS, unsafe_allow_html=True)