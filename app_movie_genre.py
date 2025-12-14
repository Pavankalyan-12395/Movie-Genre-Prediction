import streamlit as st
import joblib

st.set_page_config(page_title="Movie Genre Predictor", layout="centered")

st.title("🎬 Movie Genre Prediction App")
st.write("Enter a movie plot and get the predicted genre.")

# Load trained model
@st.cache_resource
def load_model():
    return joblib.load("model_movie.pkl")

model = load_model()

# User input
plot = st.text_area("Enter Movie Plot / Overview", height=200)

# Prediction button
if st.button("Predict Genre"):
    if plot.strip() == "":
        st.warning("Please enter a movie plot.")
    else:
        prediction = model.predict([plot])[0]
        st.success(f"Predicted Genre: **{prediction.title()}**")
