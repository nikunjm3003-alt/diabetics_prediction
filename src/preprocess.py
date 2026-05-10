import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from imblearn.over_sampling import SMOTE


def map_diag(code):
    try:
        code = str(code)
        if code.startswith('V') or code.startswith('E'):
            return 'Other'
        c = float(code)
        if 390 <= c <= 459 or c == 785: return 'Circulatory'
        if 460 <= c <= 519 or c == 786: return 'Respiratory'
        if 520 <= c <= 579 or c == 787: return 'Digestive'
        if 250 <= c <= 250.99:          return 'Diabetes'
        if 800 <= c <= 999:             return 'Injury'
        if 710 <= c <= 739:             return 'Musculoskeletal'
        if 580 <= c <= 629 or c == 788: return 'Genitourinary'
        if 140 <= c <= 239:             return 'Neoplasms'
        return 'Other'
    except:
        return 'Other'


def transform_input(df, encoder):
    age_mapping = {
        '[0-10)': 1, '[10-20)': 2, '[20-30)': 3, '[30-40)': 4,
        '[40-50)': 5, '[50-60)': 6, '[60-70)': 7,
        '[70-80)': 8, '[80-90)': 9, '[90-100)': 10
    }
    df['age']    = df['age'].map(age_mapping)
    df['gender'] = df['gender'].map({'Female': 0, 'Male': 1})
    df['diag_1'] = df['diag_1'].apply(map_diag)
    df['diag_2'] = df['diag_2'].apply(map_diag)
    df['diag_3'] = df['diag_3'].apply(map_diag)

    categorical_cols = df.select_dtypes(include='object').columns.tolist()
    encoded_cols = encoder.get_feature_names_out(categorical_cols).tolist()
    df[encoded_cols] = encoder.transform(df[categorical_cols])
    df.drop(columns=categorical_cols, inplace=True)

    # ── fix: align columns to exactly what the model was trained on ──
    expected_cols = [
        'gender', 'age', 'admission_type_id', 'discharge_disposition_id',
        'admission_source_id', 'time_in_hospital', 'num_lab_procedures',
        'num_procedures', 'num_medications', 'number_outpatient',
        'number_emergency', 'number_inpatient', 'number_diagnoses',
        'diag_1_Circulatory', 'diag_1_Diabetes', 'diag_1_Digestive',
        'diag_1_Genitourinary', 'diag_1_Injury', 'diag_1_Musculoskeletal',
        'diag_1_Neoplasms', 'diag_1_Other', 'diag_1_Respiratory',
        'diag_2_Circulatory', 'diag_2_Diabetes', 'diag_2_Digestive',
        'diag_2_Genitourinary', 'diag_2_Injury', 'diag_2_Musculoskeletal',
        'diag_2_Neoplasms', 'diag_2_Other', 'diag_2_Respiratory',
        'diag_3_Circulatory', 'diag_3_Diabetes', 'diag_3_Digestive',
        'diag_3_Genitourinary', 'diag_3_Injury', 'diag_3_Musculoskeletal',
        'diag_3_Neoplasms', 'diag_3_Other', 'diag_3_Respiratory',
        'metformin_Down', 'metformin_No', 'metformin_Steady', 'metformin_Up',
        'repaglinide_Down', 'repaglinide_No', 'repaglinide_Steady', 'repaglinide_Up',
        'nateglinide_Down', 'nateglinide_No', 'nateglinide_Steady', 'nateglinide_Up',
        'chlorpropamide_Down', 'chlorpropamide_No', 'chlorpropamide_Steady', 'chlorpropamide_Up',
        'glimepiride_Down', 'glimepiride_No', 'glimepiride_Steady', 'glimepiride_Up',
        'glipizide_Down', 'glipizide_No', 'glipizide_Steady', 'glipizide_Up',
        'glyburide_Down', 'glyburide_No', 'glyburide_Steady', 'glyburide_Up',
        'pioglitazone_Down', 'pioglitazone_No', 'pioglitazone_Steady', 'pioglitazone_Up',
        'rosiglitazone_Down', 'rosiglitazone_No', 'rosiglitazone_Steady', 'rosiglitazone_Up',
        'acarbose_Down', 'acarbose_No', 'acarbose_Steady', 'acarbose_Up',
        'miglitol_Down', 'miglitol_No', 'miglitol_Steady', 'miglitol_Up',
        'tolazamide_No', 'tolazamide_Steady', 'tolazamide_Up',
        'insulin_Down', 'insulin_No', 'insulin_Steady', 'insulin_Up',
        'glyburide-metformin_Down', 'glyburide-metformin_No',
        'glyburide-metformin_Steady', 'glyburide-metformin_Up',
        'change_Ch', 'change_No', 'diabetesMed_No', 'diabetesMed_Yes'
    ]

    for col in expected_cols:
        if col not in df.columns:
            df[col] = 0

    df = df[expected_cols]

    return df


def load_and_preprocess(filepath):

    df = pd.read_csv(filepath)

    # too many null values
    df.drop(['max_glu_serum', 'A1Cresult', 'weight', 'payer_code', 'medical_specialty'], inplace=True, axis=1)

    # not required
    df.drop(['encounter_id', 'patient_nbr', 'race'], inplace=True, axis=1)

    # columns having only 'No' as value
    useless_cols = ['examide', 'citoglipton', 'troglitazone',
                    'tolbutamide', 'acetohexamide',
                    'glimepiride-pioglitazone', 'metformin-rosiglitazone',
                    'metformin-pioglitazone', 'glipizide-metformin']
    df.drop(useless_cols, inplace=True, axis=1)

    # exclude invalid gender rows
    df = df[df['gender'] != 'Unknown/Invalid']

    # target column
    df['readmitted'] = (df['readmitted'] == '<30').astype(int)

    # split X and Y
    X = df.drop('readmitted', axis=1)
    Y = df['readmitted']

    # apply all mappings on full X BEFORE splitting
    X['age']    = X['age'].map({
        '[0-10)': 1, '[10-20)': 2, '[20-30)': 3, '[30-40)': 4,
        '[40-50)': 5, '[50-60)': 6, '[60-70)': 7,
        '[70-80)': 8, '[80-90)': 9, '[90-100)': 10
    })
    X['gender'] = X['gender'].map({'Female': 0, 'Male': 1})
    X['diag_1'] = X['diag_1'].apply(map_diag)
    X['diag_2'] = X['diag_2'].apply(map_diag)
    X['diag_3'] = X['diag_3'].apply(map_diag)

    # split AFTER mapping
    X_train, X_test, Y_train, Y_test = train_test_split(
        X, Y, random_state=42, test_size=0.2, stratify=Y
    )

    # fit encoder on mapped train data
    categorical_cols = X_train.select_dtypes(include='object').columns.tolist()
    encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
    encoder.fit(X_train[categorical_cols])

    # encode both splits
    encoded_cols = encoder.get_feature_names_out(categorical_cols).tolist()
    X_train[encoded_cols] = encoder.transform(X_train[categorical_cols])
    X_test[encoded_cols]  = encoder.transform(X_test[categorical_cols])

    X_train.drop(columns=categorical_cols, inplace=True)
    X_test.drop(columns=categorical_cols, inplace=True)

    # SMOTE
    smote = SMOTE(random_state=42)
    X_train_resampled, Y_train_resampled = smote.fit_resample(X_train, Y_train)

    print("Y_train_resampled value counts:", pd.Series(Y_train_resampled).value_counts().to_dict())
    print("X_train_resampled shape:", X_train_resampled.shape)
    print("X_test shape:", X_test.shape)

    return X_train_resampled, X_test, Y_train_resampled, Y_test, encoder