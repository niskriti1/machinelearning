import streamlit as st
import numpy as np
import pickle

# Load the trained model
with open('model.pkl', 'rb') as file:
    model = pickle.load(file)

st.title("🩺 Diabetes Prediction App")

st.write("Enter patient details to predict the likelihood of diabetes.")

# Gender input (label encoded: Male=0, Female=1, Others=2)
gender = st.selectbox("Gender", ["Male", "Female", "Others"])
gender_map = {"Male": 0, "Female": 1, "Others": 2}
gender_encoded = gender_map[gender]

# Age
age = st.number_input("Age", min_value=0, step=1)

# Hypertension
hypertension = st.selectbox("Hypertension", ["No", "Yes"])
hypertension_binary = 1 if hypertension == "Yes" else 0

# Heart Disease
heart_disease = st.selectbox("Heart Disease", ["No", "Yes"])
heart_disease_binary = 1 if heart_disease == "Yes" else 0

# Smoking History (label encoded)
smoking_history = st.selectbox("Smoking History", ["never", "current", "not current", "ever", "former", "no info"])
smoking_map = {
    "never": 0,
    "current": 1,
    "not current": 2,
    "ever": 3,
    "former": 4,
    "no info": 5
}
smoking_encoded = smoking_map[smoking_history]

# BMI
bmi = st.number_input("Body Mass Index(bmi)", min_value=0.0, step=0.1)

# HbA1c
hba1c = st.number_input("Hemoglobin A1c(HbA1c) Level", min_value=0.0, step=0.1)

# Blood Glucose
glucose = st.number_input("Blood Glucose Level", min_value=0.0, step=0.1)

# Predict
if st.button("Predict"):
    input_data = np.array([[gender_encoded, age, hypertension_binary, heart_disease_binary,
                            smoking_encoded, bmi, hba1c, glucose]])
    prediction = model.predict(input_data)

    st.subheader("Prediction Result")
    if prediction[0] == 1:
        st.success("You have diabetes")
    else:
        st.success("You don't have diabetes")
