import streamlit as st

exec(Path("pages/3_HW3.py").read_text(), {"st": st})


st.set_page_config(page_title="Homework Manager", layout="wide")

st.title("Homework Manager")
st.write("Use the sidebar to choose an assignment page (HW1, HW2, HW3).")

st.sidebar.success("Select a homework page above 👆")

st.info(
    "If Streamlit Cloud is failing, check that your requirements.txt is in the repo root "
    "and you committed/pushed it."
)









