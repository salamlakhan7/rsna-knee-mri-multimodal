import streamlit as st
import ui

u = ui.require_login()
st.title("Account")
st.write(f"**{u['name']}**  \n{u['email']}  \nRole: {u['role']}")
if st.button("Log out"):
    st.session_state.pop("user", None); st.switch_page("views/home.py")
