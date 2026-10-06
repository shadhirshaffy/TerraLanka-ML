from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

PROJECT_DIR = Path(__file__).resolve().parent
INPUT_PATH = PROJECT_DIR / "data" / "clean_lands.csv"
MODEL_DIR = PROJECT_DIR / "models"


def train_colombo_model():
    if not INPUT_PATH.exists():
        raise FileNotFoundError(f"Cleaned dataset not found at {INPUT_PATH}. Run preprocess.py first.")

    df = pd.read_csv(INPUT_PATH)

    df_colombo = df[df["district"].astype(str).str.strip().str.casefold() == "colombo"].copy()
    print(f"Filtered records for Colombo: {len(df_colombo)}")

    if df_colombo.empty:
        raise ValueError("No rows remain after filtering for district == 'Colombo'.")

    X = df_colombo[["land_extent_perches"]]
    y_raw = df_colombo["price_per_perch_lkr"]
    y = np.log1p(y_raw)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    y_test_actual = y_raw.loc[X_test.index]

    model = Ridge(alpha=1.0)
    model.fit(X_train, y_train)

    preds_log = model.predict(X_test)
    preds_actual = np.expm1(preds_log)

    mae = mean_absolute_error(y_test_actual, preds_actual)
    rmse = np.sqrt(mean_squared_error(y_test_actual, preds_actual))
    mape = np.mean(np.abs((y_test_actual - preds_actual) / y_test_actual)) * 100
    r2 = r2_score(y_test_actual, preds_actual)

    print("\nModel Evaluation Metrics")
    print(f"MAE: LKR {mae:,.2f}")
    print(f"RMSE: LKR {rmse:,.2f}")
    print(f"MAPE: {mape:.2f}%")
    print(f"R^2 Score: {r2:.3f}")

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    artifact = {
        "model": model,
        "feature_columns": ["land_extent_perches"],
        "target": "price_per_perch_lkr",
        "district": "Colombo",
        "random_state": 42,
    }
    model_path = MODEL_DIR / "colombo_land_model.pkl"
    joblib.dump(artifact, model_path)
    print(f"\nSaved model artifact to {model_path}")


if __name__ == "__main__":
    train_colombo_model()