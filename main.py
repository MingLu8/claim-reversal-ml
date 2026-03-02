
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from src.data_generation import generate_claims
from src.features import get_features_and_target
from src.model import build_model
import joblib
import os

def main():
    # 1. Generate data
    df = generate_claims(n=10000)
    print(df.dtypes)
    print(df.head())
    # 2. Split features and label
    X, y = get_features_and_target(df)

    # 3. Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # 4. Build model
    model = build_model(X_train)

    # 5. Train
    model.fit(X_train, y_train)
    os.makedirs("models", exist_ok=True)
    model_path = "models/reversal_model.joblib"
    joblib.dump(model, model_path)
    
    loaded_model = joblib.load(model_path)
    preds_loaded  = loaded_model.predict_proba(X_test)[:, 1]
    
    X_transformed = model.named_steps["preprocessor"].transform(X_train)
    print("Transformed shape:", X_transformed.shape)
    # 6. Predict probabilities
    preds = model.predict_proba(X_test)[:, 1]
    difference = (preds - preds_loaded).max()
    
    print("Max prediction difference after reload:", difference)
    
    # 7. Evaluate
    auc = roc_auc_score(y_test, preds)

    print(f"ROC-AUC: {auc:.4f}")

if __name__ == "__main__":
    main()