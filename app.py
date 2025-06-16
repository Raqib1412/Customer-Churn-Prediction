import streamlit as st
import pandas as pd
import joblib


model = joblib.load('model/churn_model.pkl')
feature_columns = joblib.load('model/feature_columns.pkl')

st.set_page_config(page_title="Customer Churn Predictor", layout="centered")
st.title("📊 Customer Churn Prediction App")

st.markdown("Fill in the customer details to predict churn probability:")


gender = st.selectbox("Gender", ["Male", "Female"])
SeniorCitizen = st.selectbox("Senior Citizen", [0, 1])
Partner = st.selectbox("Partner", ["Yes", "No"])
Dependents = st.selectbox("Dependents", ["Yes", "No"])
tenure = st.slider("Tenure (months)", 0, 72, 12)
PhoneService = st.selectbox("Phone Service", ["Yes", "No"])
MultipleLines = st.selectbox("Multiple Lines", ["No", "Yes", "No phone service"])
InternetService = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
OnlineSecurity = st.selectbox("Online Security", ["Yes", "No", "No internet service"])
OnlineBackup = st.selectbox("Online Backup", ["Yes", "No", "No internet service"])
DeviceProtection = st.selectbox("Device Protection", ["Yes", "No", "No internet service"])
TechSupport = st.selectbox("Tech Support", ["Yes", "No", "No internet service"])
StreamingTV = st.selectbox("Streaming TV", ["Yes", "No", "No internet service"])
StreamingMovies = st.selectbox("Streaming Movies", ["Yes", "No", "No internet service"])
Contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
PaperlessBilling = st.selectbox("Paperless Billing", ["Yes", "No"])
PaymentMethod = st.selectbox("Payment Method", [
    "Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"
])
MonthlyCharges = st.number_input("Monthly Charges", min_value=0.0, value=70.0)
TotalCharges = st.number_input("Total Charges", min_value=0.0, value=2000.0)


input_dict = {
    "SeniorCitizen": SeniorCitizen,
    "tenure": tenure,
    "MonthlyCharges": MonthlyCharges,
    "TotalCharges": TotalCharges,
    "gender_" + gender: 1,
    "Partner_" + Partner: 1,
    "Dependents_" + Dependents: 1,
    "PhoneService_" + PhoneService: 1,
    "MultipleLines_" + MultipleLines: 1,
    "InternetService_" + InternetService: 1,
    "OnlineSecurity_" + OnlineSecurity: 1,
    "OnlineBackup_" + OnlineBackup: 1,
    "DeviceProtection_" + DeviceProtection: 1,
    "TechSupport_" + TechSupport: 1,
    "StreamingTV_" + StreamingTV: 1,
    "StreamingMovies_" + StreamingMovies: 1,
    "Contract_" + Contract: 1,
    "PaperlessBilling_" + PaperlessBilling: 1,
    "PaymentMethod_" + PaymentMethod: 1
}


input_df = pd.DataFrame([input_dict])
for col in feature_columns:
    if col not in input_df.columns:
        input_df[col] = 0
input_df = input_df[feature_columns]  

if st.button("Predict Churn"):
    prediction = model.predict(input_df)[0]
    prob = model.predict_proba(input_df)[0][1]
    st.success(f"Churn Prediction: {'Yes' if prediction == 1 else 'No'}")
    st.info(f"Churn Probability: {prob:.2%}")
