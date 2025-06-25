import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import streamlit as st
from src.predict import predict_email

st.set_page_config(page_title="Spam Mail Detection", layout="centered")
st.title("Spam Mail Detection")

content = st.text_area("Please enter mail's content: ", height=70)

if st.button("Predict"):
    if content.strip() == "":
        st.warning("Please enter email before predicting!!!")
    else:
        result = predict_email(content)
        if result == "Spam":
            st.error("This is SPAM!!!")
        else:
            st.success("Safe email!!!")