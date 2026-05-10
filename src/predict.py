import joblib
import pandas as pd
from preprocess import transform_input

def load_model(model_path):
    artifacts = joblib.load(model_path)
    return artifacts['model'], artifacts['encoder']

def predict(input_dict, model_path=r'C:\Users\HP\OneDrive\Desktop\diabetic_prediction\model\diabetic_model_xgboost.pkl'):
    model, encoder = load_model(model_path)

    df = pd.DataFrame([input_dict])
    df = transform_input(df, encoder)

    prediction  = model.predict(df)[0]
    probability = model.predict_proba(df)[0][1]

    return {
        'prediction': int(prediction),
        'label': 'Will be Readmitted within 30 days' if prediction == 1 else 'Will NOT be Readmitted within 30 days',
        'probability': round(float(probability) * 100, 2)
    }

if __name__ == "__main__":
    sample_input = {
    'gender': 'Female',           # 'Female' or 'Male'
    'age': '[50-60)',              # age range in this exact
    'admission_type_id': 1,       # number 1-8
    'discharge_disposition_id': 1, # number 1-28
    'admission_source_id': 7,     # number 1-25
    'time_in_hospital': 5,        # number of days stayed
    'num_lab_procedures': 40,     # number of lab tests done
    'num_procedures': 1,          # number of procedures done
    'num_medications': 15,        # number of medications given
    'number_outpatient': 0,       # outpatient visits in past year
    'number_emergency': 0,        # emergency visits in past year
    'number_inpatient': 0,        # inpatient visits in past year
    'diag_1': '250.01',           # primary diagnosis ICD code
    'diag_2': '401',              # secondary diagnosis ICD code
    'diag_3': '272',              # third diagnosis ICD code
    'number_diagnoses': 9,        # total number of diagnoses
    'metformin': 'No',            # 'No', 'Steady', 'Up', 'Down'
    'repaglinide': 'No',
    'nateglinide': 'No',
    'chlorpropamide': 'No',
    'glimepiride': 'No',
    'glipizide': 'No',
    'glyburide': 'No',
    'pioglitazone': 'No',
    'rosiglitazone': 'No',
    'acarbose': 'No',
    'miglitol': 'No',
    'tolazamide': 'No',
    'insulin': 'Up',              # 'No', 'Steady', 'Up', 'Down'
    'glyburide-metformin': 'No',
    'change': 'Ch',               # 'Ch' (changed) or 'No'
    'diabetesMed': 'Yes'          # 'Yes' or 'No'
} # your patient data

    result = predict(sample_input)
    print(f"Prediction  : {result['label']}")
    print(f"Probability : {result['probability']}%")