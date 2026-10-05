import streamlit as st
import config, style

st.set_page_config(page_title=config.APP_NAME, page_icon=":material/orthopedics:", layout="wide")
style.inject()
user = st.session_state.get("user")

home = st.Page("views/home.py", title="Home", icon=":material/home:", default=True, url_path="home")
tech = st.Page("views/technology.py", title="Technology", icon=":material/memory:", url_path="technology")
results = st.Page("views/results.py", title="Results", icon=":material/bar_chart:", url_path="results")
explore = st.Page("views/explore.py", title="Live Explore", icon=":material/science:", url_path="explore")
history = st.Page("views/history.py", title="My History", icon=":material/history:", url_path="history")
admin = st.Page("views/admin.py", title="Admin", icon=":material/admin_panel_settings:", url_path="admin")
contact = st.Page("views/contact.py", title="Contact", icon=":material/mail:", url_path="contact")
access = st.Page("views/access.py", title="Log in / Sign up", icon=":material/login:", url_path="access")
account = st.Page("views/account.py", title="Account", icon=":material/account_circle:", url_path="account")

pages = [home, tech, results, explore]
if user:
    pages.append(history)
    if user["role"] == "admin": pages.append(admin)
pages += [contact, account if user else access]
st.navigation(pages, position="top").run()
