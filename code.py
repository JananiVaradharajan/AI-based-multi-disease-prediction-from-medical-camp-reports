import pandas as pd
import numpy as np
import joblib

# Disease Labels
label_cols = [
    "Diabetes", "Anemia", "CKD", "LiverDisease",
    "Hyperlipidemia", "ThyroidDisorder", "InfectionSuspected"
]

# Columns the model was trained on
required_columns = [
    "Age", "Gender", "BP_Systolic", "BP_Diastolic", "Glucose",
    "Creatinine", "Urea", "Hemoglobin", "WBC", "Platelets",
    "SGOT", "SGPT", "Bilirubin", "TotalCholesterol", "LDL",
    "HDL", "Triglycerides", "TSH", "VitaminD", "CRP",
    "SymptomsScore"
]

def predict_from_csv(input_csv, output_csv):
    model = joblib.load("multi_disease_model.pkl")
    scaler = joblib.load("scaler.pkl")

    df = pd.read_csv(input_csv)

    # Convert gender
    df["Gender"] = df["Gender"].map({"M": 1, "F": 0})

    # DROP EXTRA COLUMNS (like PhoneNumber)
    df = df[[col for col in df.columns if col in required_columns or col == "PatientID"]]

    # ADD MISSING COLUMNS
    for col in required_columns:
        if col not in df.columns:
            print(f"⚠ Missing Column Found: {col} → Adding with default 0")
            df[col] = 0

    # Order columns correctly
    features = df[required_columns]

    # Scale features
    scaled = scaler.transform(features)

    # Predict risk probability
    results = {d: [] for d in label_cols}

    for row in scaled:
        row = row.reshape(1, -1)
        for i, disease in enumerate(label_cols):
            prob = model.estimators_[i].predict_proba(row)[0][1]
            results[disease].append(round(float(prob), 4))

    # Add predictions into output
    output_df = df.copy()
    for disease in label_cols:
        output_df[f"{disease}_Risk"] = results[disease]

    # SAVE CSV
    output_df.to_csv(output_csv, index=False)
    print("\n🎉 Prediction completed!")
    print(f"Output saved to: {output_csv}")


if __name__ == "__main__":
    predict_from_csv("patient input.csv", "output_risk_prediction.csv")
