import streamlit as st
import pickle
import numpy as np

model = pickle.load(open("loan_model.pkl", "rb"))

st.title("🏦 Loan Approval Prediction System")

Gender = st.selectbox("Gender", ["Male", "Female"])
Married = st.selectbox("Married", ["Yes", "No"])
Dependents = st.selectbox("Dependents", ["0", "1", "2", "3+"])
Education = st.selectbox("Education", ["Graduate", "Not Graduate"])
Self_Employed = st.selectbox("Self Employed", ["Yes", "No"])
ApplicantIncome = st.number_input("Applicant Income", min_value=0)
CoapplicantIncome = st.number_input("Co-applicant Income", min_value=0)
LoanAmount = st.number_input("Loan Amount", min_value=0)
Loan_Amount_Term = st.number_input("Loan Amount Term", min_value=0)
Credit_History = st.selectbox("Credit History", ["Good", "Bad"])
Property_Area = st.selectbox("Property Area", ["Urban", "Semiurban", "Rural"])

# Encoding (same as training)
Gender = 1 if Gender == "Male" else 0
Married = 1 if Married == "Yes" else 0
Dependents = 3 if Dependents == "3+" else int(Dependents)
Education = 1 if Education == "Graduate" else 0
Self_Employed = 1 if Self_Employed == "Yes" else 0
Credit_History = 1 if Credit_History == "Good" else 0

Property_Area_map = {"Urban": 2, "Semiurban": 1, "Rural": 0}
Property_Area = Property_Area_map[Property_Area]

if st.button("Predict Loan Status"):
    input_data = np.array([[Gender, Married, Dependents, Education,
                            Self_Employed, ApplicantIncome,
                            CoapplicantIncome, LoanAmount,
                            Loan_Amount_Term, Credit_History,
                            Property_Area]])

    prediction = model.predict(input_data)

    if prediction[0] == 1:
        st.success("✅ Loan Approved")
    else:
        st.error("❌ Loan Rejected")

