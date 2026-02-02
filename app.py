import streamlit as st
import pandas as pd
import joblib
import json
import os
numeric_columns = ['Age', 'RestingBP', 'Cholesterol', 'MaxHR', 'Oldpeak']

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

columns_path = os.path.join(BASE_DIR, "columns.json")

with open(columns_path, "r") as f:
    columns = json.load(f)


model = joblib.load("lr_heart.pkl")
scaler = joblib.load("scaler.pkl")





st.title("Heart Disease Prediction by Shobhit ❤️")



st.markdown("Provide the following details")
age= st.slider("Age",min_value=18,max_value=120,value=40)
sex= st.selectbox("Sex",options=['M','F'])
ChestPainType= st.selectbox("Chest Pain Type",options=['ATA','NAP','ASY','TA'])
restingbp= st.number_input("Resting Blood Pressure",min_value=80,max_value=200,value=120)
cholesterol= st.number_input("Cholesterol",min_value=100,max_value=600,value=200)
fastingbs= st.selectbox("Fasting Blood Sugar > 120 mg/dl",options=['Yes','No'])
restingecg= st.selectbox("Resting ECG",options=['Normal','ST','LVH'])
maxhr= st.slider("Maximum Heart Rate Achieved",min_value=60,max_value=220,value=150)
exerciseAngina= st.selectbox("Exercise Induced Angina",options=['Y','N'])
oldpeak= st.number_input("Oldpeak",min_value=0.0,max_value=10.0,value=1.0,step=0.1)
stslope= st.selectbox("ST Slope",options=['Up','Flat','Down'])

if st.button("Predict"):
    raw_input = {
        'Age': age,
        'Sex': 1 if sex == 'F' else 0,
        'RestingBP': restingbp,
        'Cholesterol': cholesterol,
        'FastingBS': 1 if fastingbs == 'Yes' else 0,
        'MaxHR': maxhr,
        'ExerciseAngina': 1 if exerciseAngina == 'Y' else 0,
        'Oldpeak': oldpeak
    }

    input_df = pd.DataFrame([raw_input])

    # One-hot placeholders
    for col in columns:
        if col.startswith("ChestPainType_"):
            input_df[col] = 1 if col == f"ChestPainType_{ChestPainType}" else 0
        elif col.startswith("RestingECG_"):
            input_df[col] = 1 if col == f"RestingECG_{restingecg}" else 0
        elif col.startswith("ST_Slope_"):
            input_df[col] = 1 if col == f"ST_Slope_{stslope}" else 0

    # Fill missing columns
    for col in columns:
        if col not in input_df.columns:
            input_df[col] = 0

    # Reorder
    input_df = input_df[columns]

    # Scale & predict
    input_df[numeric_columns] = scaler.transform(input_df[numeric_columns])
    prediction = model.predict(input_df)[0]

    

    if prediction == 1:
        st.error("⚠️ High Risk of Heart Disease 💔")
    else:
        st.success("✅ Low Risk of Heart Disease ❤️")
