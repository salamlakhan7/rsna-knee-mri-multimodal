import streamlit as st
import config

def current_user(): return st.session_state.get("user")

def require_login():
    u = current_user()
    if not u:
        st.title("Live Explore")
        st.info("Please log in or create an account to use the live tools. Your tests are saved to your own history.")
        st.page_link("views/access.py", label="Log in or sign up", icon=":material/login:")
        st.stop()
    return u

def require_admin():
    u = require_login()
    if u["role"] != "admin":
        st.error("Admin access required."); st.stop()
    return u

def banner(): st.caption(f":material/info: {config.DISCLAIMER}")

def detect_lang(text):
    try:
        from langdetect import detect, DetectorFactory
        DetectorFactory.seed = 0
        return detect(text[:500])
    except Exception:
        return "unk"

def run_summary(r):
    if r["test_type"] == "report":
        det = r["results"].get("detected", [])
        return f"Report ({r['input_info'].get('language', '?')}, {r['input_info'].get('n_chars', 0)} chars): {len(det)} finding(s) detected"
    sc = r["results"].get("scores", {})
    top = sorted(sc.items(), key=lambda kv: -kv[1])[:2]
    return "MRI study: top scores " + ", ".join(f"{k} {v:.2f}" for k, v in top) if top else "MRI study"
