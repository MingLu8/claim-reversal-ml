
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from src.data_generation import generate_claims
from src.features import get_features_and_target
from src.model import build_model
import joblib
import os

def main():
    df = generate_claims(n=1000, seed=42)
    X, y = get_features_and_target(df)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = build_model(X_train)
    model.fit(X_train, y_train)
    os.makedirs("models", exist_ok=True)
    model_path = os.path.join("models", "claim_reversal_model.joblib")
    joblib.dump(model, model_path)
    
    pred = model.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, pred)
    print(f"Test AUC: {auc:.4f}")
    
    loaded_model = joblib.load(model_path)
    loaded_pred = loaded_model.predict_proba(X_test)[:, 1]
    difference = (pred - loaded_pred).max()
    print(f"Max difference in predictions: {difference}")
    
if __name__ == "__main__":
    main()