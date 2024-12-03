import streamlit as st

HORIZONTAL_RED = "firstStreamlitapp/cvs health logo (1).png"
ICON_RED = "firstStreamlitapp/cvs health logo (1).png"
HORIZONTAL_BLUE = "firstStreamlitapp/cvs health logo (1).png"
ICON_BLUE = "firstStreamlitapp/cvs health logo (1).png"

options = [HORIZONTAL_RED, ICON_RED, HORIZONTAL_BLUE, ICON_BLUE]
sidebar_logo = st.selectbox("Sidebar logo", options, 0)
main_body_logo = st.selectbox("Main body logo", options, 1)

st.logo(sidebar_logo, icon_image=main_body_logo)
st.sidebar.markdown("Hi!")
