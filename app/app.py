import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import streamlit as st

from nb_classifier.predict import predict as nb_predict
from vector_db.pipeline import predict_pipeline

st.set_page_config(page_title="Spam Mail Detection", layout="centered")
st.title("Spam Mail Classification")

content = st.text_area("Please enter text content:", height=68)

model_option = st.radio("Choose model:", ("Naive Bayes (NB)", "Vector Database (KNN)") )

if st.button("Predict"):
    if content.strip() == "":
        st.warning("Please enter email before predicting!!!")
    else:
        if model_option == "Naive Bayes (NB)":
            result = nb_predict([content])[0]
            st.write("**Model: Naive Bayes**")
        else:
            result = predict_pipeline([content], k=3)[0]
            st.write("**Model: Vector Database (KNN)**")
        if result is None:
            st.error("Prediction failed. Please check your models and data.")
        elif result.lower() == "spam":
            st.error("This is SPAM!!!")
        else:
            st.success("Safe email!!!")