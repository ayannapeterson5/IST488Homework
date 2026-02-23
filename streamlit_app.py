import streamlit as st

choice = st.sidebar.selectbox(
    "Choose a HW",
    ["HW1", "HW2", "HW3", "HW4", "HW5"],
    index=4
)


if choice == "HW1":
    exec(open("HW/HW1.py").read())
elif choice == "HW2":
    exec(open("HW/HW2.py").read())
elif choice == "HW3":
    exec(open("HW/HW3.py").read())
elif choice == "HW4":
    exec(open("HW/HW4.py").read())
elif choice == "HW5":
    exec(open("HW/HW5.py").read())


