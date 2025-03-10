# heart_disease_app.py

import streamlit as st
import numpy as np
import joblib

# Load model and scaler
@st.cache_resource
def load_model():
    model = joblib.load("heart_disease_model.pkl")
    scaler = joblib.load("scaler.pkl")
    return model, scaler

model, scaler = load_model()

st.title("💓 Heart Disease Prediction App")
st.write("### Enter patient details to predict the risk of heart disease:")

# Input form
age = st.slider("Age", 20, 100, 50)
sex = st.selectbox("Sex", [0, 1], format_func=lambda x: "Female" if x == 0 else "Male")
cp = st.selectbox("Chest Pain Type (cp)", [0, 1, 2, 3])
trestbps = st.slider("Resting Blood Pressure (trestbps)", 80, 200, 120)
chol = st.slider("Serum Cholesterol (chol)", 100, 400, 200)
fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dl (fbs)", [0, 1])
restecg = st.selectbox("Resting ECG Results (restecg)", [0, 1, 2])
thalach = st.slider("Max Heart Rate (thalach)", 60, 220, 150)
exang = st.selectbox("Exercise Induced Angina (exang)", [0, 1])
oldpeak = st.slider("ST depression (oldpeak)", 0.0, 6.0, 1.0)
slope = st.selectbox("Slope of the ST segment (slope)", [0, 1, 2])
ca = st.selectbox("Major Vessels Colored (ca)", [0, 1, 2, 3])
thal = st.selectbox("Thalassemia (thal)", [0, 1, 2, 3])

# Combine inputs
input_data = np.array([[age, sex, cp, trestbps, chol, fbs, restecg, thalach,
                        exang, oldpeak, slope, ca, thal]])

# Scale input
scaled_input = scaler.transform(input_data)

# Predict
if st.button("Predict"):
    prediction = model.predict(scaled_input)[0]  # 1 = Heart disease, 0 = No heart disease

    if prediction == 1:
        st.error("✅ No heart disease detected.")
    else:
        st.success("⚠️ High risk of heart disease detected.")