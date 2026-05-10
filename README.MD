\# 🏥 Diabetic Patient Readmission Predictor

\*\*APP\*\*

\[!\[Open in Streamlit](https://static.streamlit.io/badges/streamlit\_badge\_black\_white.svg)](https://diabeticsprediction-qz2w6yrgv4nkdqzt7qrn8n.streamlit.app/)



A machine learning web application that predicts whether a diabetic patient will be

readmitted to the hospital within 30 days based on clinical and demographic indicators.



\## 📌 Objective



To assist healthcare providers in identifying high-risk diabetic patients by analyzing

factors like diagnosis codes, medication changes, lab procedures, hospital stay duration,

and prior visit history — enabling proactive clinical intervention.



\## ✨ Key Features



\- \*\*Risk Prediction\*\*: Uses an XGBoost model trained on real-world diabetic patient data to classify readmission risk.

\- \*\*User Accounts\*\*: Secure registration and login using unique UUIDs.

\- \*\*Persistent History\*\*: Predictions are saved to a PostgreSQL database (Neon.tech) so they survive app restarts.

\- \*\*Clinical Input Form\*\*: Full patient data entry form covering demographics, diagnosis codes, medication changes, and visit history.

\- \*\*Probability Score\*\*: Displays readmission probability as both a percentage and visual progress bar.



\## 🛠️ How it was Built



The project follows a modular pipeline architecture for clean code and easy maintenance:



\- \*\*Frontend\*\*: Streamlit with a clinical data entry form and prediction results display.

\- \*\*Backend Database\*\*: PostgreSQL hosted on Neon.tech via SQLAlchemy.

\- \*\*Preprocessing\*\*: Custom Python scripts for encoding, ICD code mapping, and age group transformation.

\- \*\*Model\*\*: XGBoost classifier saved as a joblib artifact with label encoder.



\## 🗂️ Project Structure



```

diabetic\_readmission/

│

├── data/

│   └── diabetic\_data.csv           # raw dataset

│

├── model/

│   └── diabetic\_model\_xgboost.pkl  # trained model + encoder

│

├── notebook/

│   └── diabetic\_readmission.ipynb  # EDA and experimentation

│

├── src/

│   ├── preprocess.py               # data preprocessing and feature engineering

│   ├── train.py                    # model training script

│   └── predict.py                  # prediction logic

│

├── app.py                          # streamlit web app

├── README.md

└── requirements.txt

```



\## ⚙️ Tech Stack



\- \*\*Python\*\*

\- \*\*XGBoost\*\* — Gradient Boosted Decision Trees classifier

\- \*\*Scikit-learn\*\* — preprocessing, label encoding, evaluation metrics

\- \*\*Streamlit\*\* — web app UI

\- \*\*Pandas \& NumPy\*\* — data manipulation

\- \*\*Joblib\*\* — model serialization

\- \*\*PostgreSQL (Neon.tech)\*\* — persistent data storage

\- \*\*SQLAlchemy\*\* — database ORM



\## 🧠 ML Approach



\- \*\*Algorithm\*\*: XGBoost Classifier (Gradient Boosting)

\- \*\*Metric\*\*: ROC-AUC Score (`predict\_proba`)

\- \*\*Class Imbalance Handling\*\*: `scale\_pos\_weight` parameter in XGBoost

\- \*\*Preprocessing\*\*:

&#x20; - ICD-9 diagnosis code mapping to clinical categories (e.g., Circulatory, Diabetes, Respiratory)

&#x20; - Age group encoding (`\[0-10)` → numerical midpoints)

&#x20; - Label encoding for categorical features (gender, medications, medication changes)



\## 📊 Dataset



\- \*\*Source\*\*: UCI ML Repository — Diabetes 130-US Hospitals (1999–2008)

\- \*\*Rows\*\*: \~100,000 patient encounters

\- \*\*Features\*\*: gender, age, admission type, time in hospital, lab procedures, medications,

&#x20; diagnosis codes, outpatient/emergency/inpatient visits, medication changes, and more



\## 📈 Prediction Categories



| Prediction | Risk Level | Description |

|---|---|---|

| 1 | 🔴 High Risk | Patient likely to be readmitted within 30 days |

| 0 | 🟢 Low Risk | Patient unlikely to be readmitted within 30 days |

