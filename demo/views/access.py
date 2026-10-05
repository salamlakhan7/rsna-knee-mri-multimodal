import streamlit as st
import auth, config

left, right = st.columns([2, 3], gap="large")
with left:
    with st.container(key="brand"):
        st.markdown(f"### :material/orthopedics: {config.APP_NAME}")
        st.write("Test the models and keep your own history of every test.")
        st.markdown("- Report to findings in four languages\n- MRI study scoring\n- Private, per-user history")
with right:
    st.title("Log in or sign up")
    t_in, t_up = st.tabs(["Log in", "Sign up"])
    with t_in:
        with st.form("login"):
            email = st.text_input("Email"); pw = st.text_input("Password", type="password")
            go = st.form_submit_button("Log in", type="primary")
        if go:
            user, err = auth.authenticate(email, pw)
            if user:
                st.session_state["user"] = user; st.switch_page("views/explore.py")
            else:
                st.error(err)
    with t_up:
        with st.form("signup"):
            name = st.text_input("Full name"); em = st.text_input("Email", key="su_email")
            p1 = st.text_input("Password (8+ characters, a letter and a digit)", type="password", key="su_p1")
            p2 = st.text_input("Confirm password", type="password", key="su_p2")
            go2 = st.form_submit_button("Create account", type="primary")
        if go2:
            user, errs = auth.register(name, em, p1, p2)
            if user:
                st.session_state["user"] = user; st.switch_page("views/explore.py")
            else:
                for e in errs: st.error(e)