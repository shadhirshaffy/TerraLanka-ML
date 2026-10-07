from pathlib import Path
from datetime import datetime, timezone
import json

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import (
    GradientBoostingRegressor,
    HistGradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


PROJECT_DIR = Path(__file__).resolve().parent
INPUT_PATH = PROJECT_DIR / "data" / "clean_lands.csv"
MODEL_DIR = PROJECT_DIR / "models"
EVALUATION_DIR = PROJECT_DIR / "data" / "evaluation"
METADATA_PATH = MODEL_DIR / "colombo_land_model_metadata.json"

NUMERIC_FEATURES = [
    "land_extent_perches",
    "log_land_extent_perches",
    "distance_to_colombo_km",
    "kw_commercial",
    "kw_residential",
    "kw_main_road",
    "kw_bare_land",
    "kw_lake",
    "kw_corner",
    "kw_approved",
    "kw_prime",
]
CATEGORICAL_FEATURES = ["district", "city", "land_type", "price_basis"]
TEXT_FEATURE = "title_clean"
TARGET = "price_per_perch_lkr"


def evaluate_predictions(y_true, y_pred):
    return {
        "mae": mean_absolute_error(y_true, y_pred),
        "rmse": np.sqrt(mean_squared_error(y_true, y_pred)),
        "mape": np.mean(np.abs((y_true - y_pred) / y_true)) * 100,
        "r2": r2_score(y_true, y_pred),
    }


def metrics_to_row(model_name, metrics, baseline_metrics=None):
    row = {"model": model_name, **metrics}
    if baseline_metrics:
        row["beats_baseline_mae"] = metrics["mae"] < baseline_metrics["mae"]
        row["mae_improvement_lkr"] = baseline_metrics["mae"] - metrics["mae"]
        row["mae_improvement_pct"] = (
            row["mae_improvement_lkr"] / baseline_metrics["mae"] * 100
        )
        row["mape_improvement_pct_points"] = baseline_metrics["mape"] - metrics["mape"]
    return row


def write_csv_with_fallback(df, output_path):
    try:
        df.to_csv(output_path, index=False)
        return output_path
    except PermissionError:
        fallback_path = output_path.with_name(
            f"{output_path.stem}_{datetime.now().strftime('%Y%m%d_%H%M%S')}{output_path.suffix}"
        )
        df.to_csv(fallback_path, index=False)
        return fallback_path


def print_metrics(name, metrics):
    print(f"\n{name}")
    print(f"MAE: LKR {metrics['mae']:,.2f}")
    print(f"RMSE: LKR {metrics['rmse']:,.2f}")
    print(f"MAPE: {metrics['mape']:.2f}%")
    print(f"R^2 Score: {metrics['r2']:.3f}")


def build_preprocessor():
    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", min_frequency=2)),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, NUMERIC_FEATURES),
            ("categorical", categorical_pipeline, CATEGORICAL_FEATURES),
            ("title_text", TfidfVectorizer(max_features=80, ngram_range=(1, 2)), TEXT_FEATURE),
        ],
        remainder="drop",
        sparse_threshold=0.0,
    )


def build_model_candidates():
    return {
        "ridge": Ridge(alpha=1.0),
        "random_forest": RandomForestRegressor(
            n_estimators=300,
            min_samples_leaf=3,
            random_state=42,
            n_jobs=-1,
        ),
        "gradient_boosting": GradientBoostingRegressor(
            n_estimators=250,
            learning_rate=0.05,
            max_depth=3,
            random_state=42,
        ),
        "hist_gradient_boosting": HistGradientBoostingRegressor(
            max_iter=250,
            learning_rate=0.05,
            l2_regularization=0.1,
            random_state=42,
        ),
    }


def build_model_pipeline(regressor):
    return Pipeline(
        steps=[
            ("preprocessor", build_preprocessor()),
            ("regressor", regressor),
        ]
    )


def prepare_dataset():
    if not INPUT_PATH.exists():
        raise FileNotFoundError(f"Cleaned dataset not found at {INPUT_PATH}. Run preprocess.py first.")

    df = pd.read_csv(INPUT_PATH)
    required_columns = set(NUMERIC_FEATURES + CATEGORICAL_FEATURES + [TEXT_FEATURE, TARGET])
    missing_columns = required_columns.difference(df.columns)
    if missing_columns:
        raise ValueError(
            f"Cleaned dataset is missing columns: {', '.join(sorted(missing_columns))}. "
            "Run preprocess.py again."
        )

    df = df.dropna(subset=[TARGET]).copy()
    df[TEXT_FEATURE] = df[TEXT_FEATURE].fillna("")
    df[CATEGORICAL_FEATURES] = df[CATEGORICAL_FEATURES].fillna("Unknown")

    if len(df) < 30:
        raise ValueError(f"Only {len(df)} usable rows found; collect more listings before training.")

    return df


def split_features_target(df, test_size=0.2, random_state=42):
    X = df[NUMERIC_FEATURES + CATEGORICAL_FEATURES + [TEXT_FEATURE]]
    y_raw = df[TARGET].astype(float)
    y = np.log1p(y_raw)
    return train_test_split(
        X, y, y_raw, test_size=test_size, random_state=random_state
    )


def evaluate_city_median_baseline(X_train, X_test, y_raw_train, y_raw_test):
    city_medians = y_raw_train.groupby(X_train["city"]).median()
    global_median = y_raw_train.median()
    baseline_preds = X_test["city"].map(city_medians).fillna(global_median)
    return evaluate_predictions(y_raw_test, baseline_preds), baseline_preds


def train_land_price_model():
    df = prepare_dataset()
    print(f"Training with {len(df)} cleaned records.")
    print(f"Cities represented: {df['city'].nunique()}")

    X_train, X_test, y_train, y_test, y_raw_train, y_raw_test = train_test_split(
        df[NUMERIC_FEATURES + CATEGORICAL_FEATURES + [TEXT_FEATURE]],
        np.log1p(df[TARGET].astype(float)),
        df[TARGET].astype(float),
        test_size=0.2,
        random_state=42,
    )

    baseline_metrics, _ = evaluate_city_median_baseline(
        X_train, X_test, y_raw_train, y_raw_test
    )
    print_metrics("City Median Baseline", baseline_metrics)

    results = {}
    fitted_models = {}
    evaluation_rows = [metrics_to_row("city_median_baseline", baseline_metrics)]
    for name, regressor in build_model_candidates().items():
        pipeline = build_model_pipeline(regressor)
        pipeline.fit(X_train, y_train)
        pred_log = pipeline.predict(X_test)
        pred_actual = np.expm1(pred_log)
        metrics = evaluate_predictions(y_raw_test, pred_actual)
        results[name] = metrics
        fitted_models[name] = pipeline
        evaluation_rows.append(metrics_to_row(name, metrics, baseline_metrics))
        print_metrics(f"{name.replace('_', ' ').title()} Model", metrics)

    best_model_name = min(results, key=lambda key: results[key]["mae"])
    best_pipeline = fitted_models[best_model_name]
    best_metrics = results[best_model_name]

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    artifact = {
        "model": best_pipeline,
        "model_name": best_model_name,
        "numeric_features": NUMERIC_FEATURES,
        "categorical_features": CATEGORICAL_FEATURES,
        "text_feature": TEXT_FEATURE,
        "feature_columns": NUMERIC_FEATURES + CATEGORICAL_FEATURES + [TEXT_FEATURE],
        "target": TARGET,
        "target_transform": "log1p",
        "metrics": best_metrics,
        "baseline_metrics": baseline_metrics,
        "training_rows": len(df),
        "random_state": 42,
    }
    model_path = MODEL_DIR / "colombo_land_model.pkl"
    joblib.dump(artifact, model_path)

    metadata = {
        "trained_at_utc": datetime.now(timezone.utc).isoformat(),
        "model_path": str(model_path),
        "model_name": best_model_name,
        "training_rows": len(df),
        "city_count": int(df["city"].nunique()),
        "target": TARGET,
        "target_transform": "log1p",
        "numeric_features": NUMERIC_FEATURES,
        "categorical_features": CATEGORICAL_FEATURES,
        "text_feature": TEXT_FEATURE,
        "feature_columns": NUMERIC_FEATURES + CATEGORICAL_FEATURES + [TEXT_FEATURE],
        "metrics": best_metrics,
        "baseline_metrics": baseline_metrics,
        "beats_city_median_baseline_mae": best_metrics["mae"] < baseline_metrics["mae"],
        "model_candidates": list(build_model_candidates().keys()),
    }
    with METADATA_PATH.open("w", encoding="utf-8") as metadata_file:
        json.dump(metadata, metadata_file, indent=2)

    EVALUATION_DIR.mkdir(parents=True, exist_ok=True)
    evaluation_output = EVALUATION_DIR / "model_comparison.csv"
    evaluation_output = write_csv_with_fallback(
        pd.DataFrame(evaluation_rows).sort_values("mae"), evaluation_output
    )

    print(f"\nBest model: {best_model_name}")
    print(f"Saved model artifact to {model_path}")
    print(f"Saved model metadata to {METADATA_PATH}")
    print(f"Saved model comparison to {evaluation_output}")


if __name__ == "__main__":
    train_land_price_model()
