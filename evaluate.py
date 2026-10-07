from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from train import (
    CATEGORICAL_FEATURES,
    MODEL_DIR,
    NUMERIC_FEATURES,
    PROJECT_DIR,
    TARGET,
    TEXT_FEATURE,
    build_model_candidates,
    build_model_pipeline,
    evaluate_city_median_baseline,
    evaluate_predictions,
    metrics_to_row,
    prepare_dataset,
    print_metrics,
    split_features_target,
    write_csv_with_fallback,
)


EVALUATION_DIR = PROJECT_DIR / "data" / "evaluation"
MODEL_PATH = MODEL_DIR / "colombo_land_model.pkl"


def build_prediction_report(y_true, baseline_preds, model_predictions):
    report = pd.DataFrame(
        {
            "actual_price_per_perch_lkr": y_true,
            "baseline_prediction_lkr": baseline_preds,
            "model_prediction_lkr": model_predictions,
        }
    )
    report["baseline_abs_error_lkr"] = (
        report["actual_price_per_perch_lkr"] - report["baseline_prediction_lkr"]
    ).abs()
    report["model_abs_error_lkr"] = (
        report["actual_price_per_perch_lkr"] - report["model_prediction_lkr"]
    ).abs()
    return report


def evaluate_all_models():
    df = prepare_dataset()
    X_train, X_test, y_train, y_test, y_raw_train, y_raw_test = split_features_target(
        df
    )

    baseline_metrics, baseline_preds = evaluate_city_median_baseline(
        X_train, X_test, y_raw_train, y_raw_test
    )
    print_metrics("City Median Baseline", baseline_metrics)

    rows = [metrics_to_row("city_median_baseline", baseline_metrics)]
    fitted_models = {}
    prediction_outputs = {}

    for name, regressor in build_model_candidates().items():
        pipeline = build_model_pipeline(regressor)
        pipeline.fit(X_train, y_train)
        predictions = np.expm1(pipeline.predict(X_test))
        metrics = evaluate_predictions(y_raw_test, predictions)
        rows.append(metrics_to_row(name, metrics, baseline_metrics))
        fitted_models[name] = pipeline
        prediction_outputs[name] = predictions
        print_metrics(f"{name.replace('_', ' ').title()} Model", metrics)

    results = pd.DataFrame(rows).sort_values("mae")
    best_model_name = results.loc[results["model"] != "city_median_baseline", "model"].iloc[0]
    best_metrics = results.loc[results["model"] == best_model_name].iloc[0].to_dict()

    if not bool(best_metrics["beats_baseline_mae"]):
        raise RuntimeError(
            f"Best model '{best_model_name}' did not beat the city-median baseline on MAE."
        )

    EVALUATION_DIR.mkdir(parents=True, exist_ok=True)
    comparison_path = EVALUATION_DIR / "model_comparison.csv"
    predictions_path = EVALUATION_DIR / "holdout_predictions.csv"

    comparison_path = write_csv_with_fallback(results, comparison_path)
    prediction_report = build_prediction_report(
        y_raw_test.reset_index(drop=True),
        baseline_preds.reset_index(drop=True),
        pd.Series(prediction_outputs[best_model_name]).reset_index(drop=True),
    )
    predictions_path = write_csv_with_fallback(prediction_report, predictions_path)

    print(f"\nBest model: {best_model_name}")
    print(f"Saved evaluation comparison to {comparison_path}")
    print(f"Saved holdout predictions to {predictions_path}")

    if MODEL_PATH.exists():
        artifact = joblib.load(MODEL_PATH)
        print(f"Current saved artifact model: {artifact.get('model_name', 'unknown')}")


if __name__ == "__main__":
    evaluate_all_models()
