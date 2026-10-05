import streamlit as st, pandas as pd
import db, ui

ui.require_admin()
st.title("Admin"); st.caption("Users and their test history")
users = db.all_users()
df = pd.DataFrame([{"Name": x["name"], "Email": x["email"], "Role": x["role"], "Joined": x["created_at"].strftime("%Y-%m-%d"),
                    "Last login": x["last_login"].strftime("%Y-%m-%d %H:%M") if x["last_login"] else "-",
                    "Tests": x["n_tests"]} for x in users])
c = st.columns(3); c[0].metric("Users", len(users)); c[1].metric("Total tests", int(df["Tests"].sum()) if len(df) else 0)
st.dataframe(df, hide_index=True)
if users:
    pick = st.selectbox("View a user's history", [x["email"] for x in users])
    uid = next(x["id"] for x in users if x["email"] == pick)
    runs = db.user_runs(uid)
    st.write(f"{len(runs)} test(s)")
    for r in runs:
        with st.expander(f"{r['created_at']:%Y-%m-%d %H:%M} | {ui.run_summary(r)}"):
            st.json({"input": r["input_info"], "results": r["results"]})
