import joblib
import os
from sklearn.metrics import classification_report, roc_auc_score
from xgboost import XGBClassifier
from preprocess import load_and_preprocess

def train_and_save(data_path, model_save_path):
    print("Loading and Preprocessing data")
    X_train_resampled, X_test, Y_train_resampled, Y_test, encoder = load_and_preprocess(data_path)

    print("Training The Model")
    model = XGBClassifier(
    random_state=42,
    n_estimators=300,        # more trees
    max_depth=6,             # deeper trees
    learning_rate=0.05,      # slower learning = better generalization
    scale_pos_weight=8,
    subsample=0.8,           # use 80% of data per tree
    colsample_bytree=0.8,    # use 80% of features per tree
    eval_metric='aucpr'
)
    model.fit(X_train_resampled, Y_train_resampled)

    preds = model.predict(X_test)
    proba = model.predict_proba(X_test)[:, 1]   # ← probabilities not predictions

    roc_auc = roc_auc_score(Y_test, proba)       # ← fixed
    report  = classification_report(Y_test, preds)

    print(f"\nTraining Complete!")
    print(f"   ROC_AUC : {roc_auc:.4f}")
    print(f"   Report  : \n{report}")

    os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
    joblib.dump({'model': model, 'encoder': encoder}, model_save_path)
    print(f"\nModel Saved To : {model_save_path}")


if __name__ == "__main__":
    train_and_save(
        data_path=r'C:\Users\HP\OneDrive\Desktop\diabetic_preddiction\data\diabetic_data.csv',
        model_save_path=r'C:\Users\HP\OneDrive\Desktop\diabetic_preddiction\model\diabetic_model_xgboost.pkl'
    )