"""Evaluate daily sales forecasts and save regression metrics."""

import json
from pathlib import Path

import joblib
import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error


def main() -> None:
    test_data = pd.read_csv("data/processed/test.csv", parse_dates=["date"])
    features = test_data.drop(columns=["date", "sales"])
    actual = test_data["sales"].to_numpy()
    model = joblib.load("models/model.pkl")
    predictions = model.predict(features)
    mape_denominator = np.maximum(np.abs(actual), 1e-8)

    metrics = {
        "MAE": float(mean_absolute_error(actual, predictions)),
        "RMSE": float(np.sqrt(mean_squared_error(actual, predictions))),
        "MAPE": float(np.mean(np.abs((actual - predictions) / mape_denominator)) * 100),
    }
    Path("metrics.json").write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    print("Test metrics:")
    for metric_name, value in metrics.items():
        print(f"  {metric_name}: {value:.4f}")
    print("Saved metrics to metrics.json")


if __name__ == "__main__":
    main()