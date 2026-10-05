import streamlit as st, pandas as pd
import db, ui

u = ui.require_login()
st.title("My History"); st.caption(f"Tests run by {u['name']}")
runs = db.user_runs(u["id"])
if not runs:
    st.info("No tests yet. Try Live Explore."); st.page_link("views/explore.py", label="Open Live Explore"); st.stop()
df = pd.DataFrame([{"Time (UTC)": r["created_at"].strftime("%Y-%m-%d %H:%M"), "Type": r["test_type"],
                    "Summary": ui.run_summary(r), "Seconds": round(r["duration_ms"] / 1000, 1)} for r in runs])
st.dataframe(df, hide_index=True)
st.download_button("Download history (CSV)", df.to_csv(index=False), "my_history.csv", "text/csv")
st.subheader("Details")
for r in runs[:25]:
    with st.expander(f"{r['created_at']:%Y-%m-%d %H:%M} | {ui.run_summary(r)}"):
        st.json({"input": r["input_info"], "results": r["results"]})
