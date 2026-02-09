import streamlit as st
from pathlib import Path

st.set_page_config(page_title="IST 488 Homework", layout="centered")
st.title("Homework Manager")

choice = st.sidebar.radio(
    "Select assignment",
    ["HW 1", "HW 2", "HW 3"]
)

if choice == "HW 1":
    exec(Path("HW/HW1.py").read_text(), {})
elif choice == "HW 2":
    exec(Path("HW/HW2.py").read_text(), {})
else:
    exec(Path("HW/HW3.py").read_text(), {})







