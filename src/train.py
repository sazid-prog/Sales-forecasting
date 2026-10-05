"""Train and save the configured sales forecasting model."""

from pathlib import Path

import joblib
import pandas as pd
import yaml
from sklearn.ensemble import RandomForestRegressor


def main() -> None:
    with Path("params.yaml").open("r", encoding="utf-8") as parameter_file:
        parameters = yaml.safe_load(parameter_file)

    train_data = pd.read_csv("data/processed/train.csv", parse_dates=["date"])
    features = train_data.drop(columns=["date", "sales"])
    target = train_data["sales"]
    train_parameters = parameters["train"]
    model = RandomForestRegressor(
        n_estimators=train_parameters["n_estimators"],
        max_depth=train_parameters["max_depth"],
        random_state=train_parameters["seed"],
        n_jobs=-1,
    )
    model.fit(features, target)

    model_path = Path("models/model.pkl")
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, model_path)
    print(f"Trained on {len(train_data)} dates using {features.shape[1]} features.")
    print(f"Training R-squared: {model.score(features, target):.4f}")
    print(f"Saved model to {model_path}")


if __name__ == "__main__":
    main()