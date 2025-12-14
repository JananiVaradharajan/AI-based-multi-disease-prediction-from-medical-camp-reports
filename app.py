import streamlit as st
import pandas as pd
import numpy as np
import joblib

# ---------------------------------------------------------
# CONFIG
# ---------------------------------------------------------
st.set_page_config(page_title="Multi-Disease Risk Predictor", layout="wide")

# Disease Labels
label_cols = [
    "Diabetes", "Anemia", "CKD", "LiverDisease",
    "Hyperlipidemia", "ThyroidDisorder", "InfectionSuspected"
]

# Columns used during training
required_columns = [
    "Age", "Gender", "BP_Systolic", "BP_Diastolic", "Glucose",
    "Creatinine", "Urea", "Hemoglobin", "WBC", "Platelets",
    "SGOT", "SGPT", "Bilirubin", "TotalCholesterol", "LDL",
    "HDL", "Triglycerides", "TSH", "VitaminD", "CRP",
    "SymptomsScore"
]

# ---------------------------------------------------------
# LOAD MODEL & SCALER
# ---------------------------------------------------------
@st.cache_resource
def load_model():
    model = joblib.load("multi_disease_model.pkl")
    scaler = joblib.load("scaler.pkl")
    return model, scaler

model, scaler = load_model()

# ---------------------------------------------------------
# UI TITLE
# ---------------------------------------------------------
st.title("🩺 Multi-Disease AI Risk Prediction System")
st.write("Upload a patient CSV file and get sorted risk predictions with disease labels.")

uploaded_file = st.file_uploader("📤 Upload patient CSV file", type=["csv"])

# ---------------------------------------------------------
# PROCESS FILE
# ---------------------------------------------------------
if uploaded_file is not None:
    st.success("File uploaded successfully!")

    df = pd.read_csv(uploaded_file)
    st.subheader("📄 Uploaded Data Preview")
    st.dataframe(df.head())

    # Convert Gender values
    if "Gender" in df.columns:
        df["Gender"] = df["Gender"].map({"M": 1, "F": 0})

    # Copy original df for output
    output_df = df.copy()

    # Keep only required columns for prediction
    df = df[[col for col in df.columns if col in required_columns]]

    # Fill missing columns automatically
    for col in required_columns:
        if col not in df.columns:
            if col == "BP_Systolic":
                st.warning("Missing BP_Systolic → Filling with 120")
                df[col] = 120
            elif col == "BP_Diastolic":
                st.warning("Missing BP_Diastolic → Filling with 80")
                df[col] = 80
            else:
                st.warning(f"Missing {col} → Filling with 0")
                df[col] = 0

    # Ensure correct order
    df = df[required_columns]

    # Scale features
    scaled = scaler.transform(df)

    # ---------------------------------------------------------
    # PREDICTION
    # ---------------------------------------------------------
    st.subheader("🤖 Predicting Risk Scores...")

    predictions = {d: [] for d in label_cols}

    for row in scaled:
        row = row.reshape(1, -1)
        for i, disease in enumerate(label_cols):
            prob = model.estimators_[i].predict_proba(row)[0][1]
            predictions[disease].append(round(float(prob), 4))

    # Add risk scores to output
    for disease in label_cols:
        output_df[f"{disease}_Risk"] = predictions[disease]

    # ---------------------------------------------------------
    # ADD Predicted Diseases column
    # ---------------------------------------------------------
    def get_predicted_diseases(row):
        predicted = [d for d in label_cols if row[f"{d}_Risk"] >= 0.5]
        return ", ".join(predicted) if predicted else "None"

    output_df["Predicted_Diseases"] = output_df.apply(get_predicted_diseases, axis=1)

    # ---------------------------------------------------------
    # SORT HIGH RISK → LOW RISK
    # ---------------------------------------------------------
    output_df["Max_Risk"] = output_df[[f"{d}_Risk" for d in label_cols]].max(axis=1)
    output_df = output_df.sort_values(by="Max_Risk", ascending=False)

    # OPTIONAL: Remove Max_Risk from display
    # output_df = output_df.drop(columns=["Max_Risk"])

    st.subheader("📊 Sorted Results: High Risk → Low Risk")
    st.dataframe(output_df)

    # ---------------------------------------------------------
    # DOWNLOAD UPDATED CSV
    # ---------------------------------------------------------
    csv_data = output_df.to_csv(index=False).encode('utf-8')

    st.download_button(
        label="📥 Download Sorted CSV with Predicted Diseases",
        data=csv_data,
        file_name="sorted_patient_risk_predictions.csv",
        mime="text/csv"
    )

    st.success("✔ Prediction Completed & Sorted Successfully!")
