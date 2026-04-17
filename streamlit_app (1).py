
import joblib
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Diabetes Logistic Regression", layout="centered")

model = joblib.load("diabetes_logreg.joblib")

numeric_features = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']
categorical_features = []
cat_values = {}

st.title("Diabetes Prediction")

with st.form("input_form"):
    inputs = {}
    st.subheader("Numeric inputs")
    for col in numeric_features:
        inputs[col] = st.number_input(col, value=0.0)
    if categorical_features:
        st.subheader("Categorical inputs")
        for col in categorical_features:
            options = cat_values.get(col, [])
            if options:
                inputs[col] = st.selectbox(col, options)
            else:
                inputs[col] = st.text_input(col, value="")
    submitted = st.form_submit_button("Predict")

if submitted:
    df_input = pd.DataFrame([inputs])
    proba = model.predict_proba(df_input)[:, 1][0]
    pred = int(proba >= 0.5)
    st.write(f"Prediction: {pred}")
    st.write(f"Probability of positive class: {proba:.3f}")
