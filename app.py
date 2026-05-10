import streamlit as st
import pandas as pd
import joblib
import uuid
from sqlalchemy import text
from src.preprocess import transform_input

# --- Page Config ---
st.set_page_config(page_title="Diabetes Readmission Predictor", layout="wide", page_icon="🏥")

# --- Database Connection ---
# Ensure your secrets.toml is configured for 'postgresql'
conn = st.connection('postgresql', type='sql')

# --- Session State Initialization ---
if 'logged_in' not in st.session_state:
    st.session_state.logged_in = False
    st.session_state.user_id = None
    st.session_state.username = None

# --- Authentication Logic ---
def auth_page():
    st.title("🩺 Diabetic Patient Predictor")
    tab1, tab2 = st.tabs(['Login', 'Register'])

    with tab2:
        st.subheader("Create Account")
        new_un = st.text_input("Username", key="reg_un")
        new_addr = st.text_area("Address", key="reg_addr")
        
        if st.button("Register"):
            if new_un:
                u_id = str(uuid.uuid4())
                try:
                    with conn.session as s:
                        # Fixed table name to match login: 'diaetic_users'
                        s.execute(
                            text("INSERT INTO diaetic_users(user_id, username, address) VALUES(:id, :un, :ad)"),
                            {"id": u_id, "un": new_un, "ad": new_addr}
                        )
                        s.commit()
                    st.success(f"Registered! Your ID: {u_id}. Please switch to Login.")
                except Exception as e:
                    st.error(f"Registration Error: {e}")
            else:
                st.warning("Please enter a username.")

    with tab1:
        st.subheader("Login")
        un = st.text_input("Enter Username", key='log_un')
        if st.button("Login"):
            # Using parameterized query to prevent SQL Injection
            res = conn.query("SELECT user_id FROM diaetic_users WHERE username = :un", 
                             params={"un": un}, ttl=0)
            
            if not res.empty:
                st.session_state.logged_in = True
                st.session_state.user_id = res.iloc[0]['user_id']
                st.session_state.username = un
                st.rerun()
            else:
                st.error("Username Not Found. Please register first.")

# --- Load Model Function ---
@st.cache_resource
def load_prediction_artifacts(model_path):
    # Standardizing path handling
    artifacts = joblib.load(model_path)
    return artifacts['model'], artifacts['encoder']

# --- Main App Logic ---
if not st.session_state.logged_in:
    auth_page()
else:
    # Sidebar Logout
    with st.sidebar:
        st.write(f"Logged in as: **{st.session_state.username}**")
        if st.button("Logout"):
            st.session_state.logged_in = False
            st.rerun()

    st.title("🏥 Patient Readmission Prediction")
    st.markdown("Predicts whether a diabetic patient will be readmitted within 30 days based on clinical data.")

    # --- Loading Assets ---
    MODEL_PATH = r'model/diabetic_model_xgboost.pkl' 
    try:
        model, encoder = load_prediction_artifacts(MODEL_PATH)
    except Exception as e:
        st.error(f"Critical Error: Could not load model artifacts. {e}")
        st.stop()

    # --- Input Form ---
    with st.form("prediction_form"):
        col1, col2, col3 = st.columns(3)

        with col1:
            st.subheader("Demographics")
            gender = st.selectbox("Gender", ["Female", "Male"])
            age = st.selectbox("Age Range", ['[0-10)', '[10-20)', '[20-30)', '[30-40)', '[40-50)', 
                                            '[50-60)', '[60-70)', '[70-80)', '[80-90)', '[90-100)'])
            time_in_hospital = st.number_input("Days in Hospital", min_value=1, max_value=14, value=5)
            
        with col2:
            st.subheader("Clinical Metrics")
            num_lab_procedures = st.number_input("Lab Procedures", value=40)
            num_procedures = st.number_input("Procedures", value=1)
            num_medications = st.number_input("Medications", value=15)
            number_diagnoses = st.number_input("Total Diagnoses", value=9)

        with col3:
            st.subheader("History")
            number_outpatient = st.number_input("Outpatient Visits (Past Year)", value=0)
            number_emergency = st.number_input("Emergency Visits (Past Year)", value=0)
            number_inpatient = st.number_input("Inpatient Visits (Past Year)", value=0)
            admission_type_id = st.number_input("Admission Type ID (1-8)", value=1)

        st.divider()
        
        col4, col5 = st.columns(2)
        with col4:
            st.subheader("Diagnosis Codes (ICD)")
            diag_1 = st.text_input("Primary Diagnosis", value="250.01")
            diag_2 = st.text_input("Secondary Diagnosis", value="401")
            diag_3 = st.text_input("Additional Diagnosis", value="272")

        with col5:
            st.subheader("Medication Changes")
            insulin = st.selectbox("Insulin", ['No', 'Steady', 'Up', 'Down'])
            metformin = st.selectbox("Metformin", ['No', 'Steady', 'Up', 'Down'])
            diabetesMed = st.selectbox("On Diabetes Medication?", ['Yes', 'No'])
            change = st.selectbox("Change in Meds?", ['Ch', 'No'])

        submit = st.form_submit_button("Predict Readmission Risk")

    # --- Prediction & Database Storage ---
    if submit:
        # 1. Prepare Input Data
        input_data = {
            'gender': gender, 'age': age, 'admission_type_id': admission_type_id,
            'discharge_disposition_id': 1, 'admission_source_id': 7, 
            'time_in_hospital': time_in_hospital, 'num_lab_procedures': num_lab_procedures,
            'num_procedures': num_procedures, 'num_medications': num_medications,
            'number_outpatient': number_outpatient, 'number_emergency': number_emergency,
            'number_inpatient': number_inpatient, 'diag_1': diag_1, 'diag_2': diag_2,
            'diag_3': diag_3, 'number_diagnoses': number_diagnoses, 'metformin': metformin,
            'repaglinide': 'No', 'nateglinide': 'No', 'chlorpropamide': 'No',
            'glimepiride': 'No', 'glipizide': 'No', 'glyburide': 'No',
            'pioglitazone': 'No', 'rosiglitazone': 'No', 'acarbose': 'No',
            'miglitol': 'No', 'tolazamide': 'No', 'insulin': insulin,
            'glyburide-metformin': 'No', 'change': change, 'diabetesMed': diabetesMed
        }

        # 2. Transform and Predict
        df = pd.DataFrame([input_data])
        processed_df = transform_input(df, encoder)
        
        prediction = model.predict(processed_df)[0]
        probability = model.predict_proba(processed_df)[0][1]

        # 3. Display Results
        st.divider()
        if prediction == 1:
            st.error("### Result: High Risk of Readmission")
        else:
            st.success("### Result: Low Risk of Readmission")
        
        st.metric(label="Readmission Probability", value=f"{round(float(probability) * 100, 2)}%")
        st.progress(float(probability))

        # 4. Save to Database
        try:
            with conn.session as s:
                s.execute(
                    text("""
                        INSERT INTO diabetics_prediction (
                            user_id, gender, age, time_in_hospital, prediction, probability
                        ) 
                        VALUES (:uid, :gen, :age, :time, :pred, :prob)
                    """),
                    {
                        "uid": st.session_state.user_id,
                        "gen": gender,
                        "age": age,
                        "time": time_in_hospital,
                        "pred": int(prediction),
                        "prob": float(probability)
                    }
                )
                s.commit()
            st.caption("✅ Prediction saved to history.")
        except Exception as e:
            st.warning(f"Result displayed, but failed to save to database: {e}")