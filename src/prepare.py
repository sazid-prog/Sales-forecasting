"""Build lag features and split daily sales chronologically."""

from pathlib import Path

import pandas as pd
import yaml


def main() -> None:
    with Path("params.yaml").open("r", encoding="utf-8") as parameter_file:
        parameters = yaml.safe_load(parameter_file)

    prepare_parameters = parameters["prepare"]
    sales = pd.read_csv("data/raw/sales.csv", parse_dates=["date"])
    if sales["date"].isna().any() or sales["sales"].isna().any():
        raise ValueError("Input data must have valid dates and sales values.")
    sales["sales"] = pd.to_numeric(sales["sales"], errors="raise")

    # Combine duplicate dates before calculating calendar and lag features.
    sales = sales.groupby("date", as_index=False)["sales"].sum().sort_values("date")
    sales["month"] = sales["date"].dt.month
    sales["dayofweek"] = sales["date"].dt.dayofweek
    sales["weekofyear"] = sales["date"].dt.isocalendar().week.astype(int)

    for lag_days in prepare_parameters["lag_days"]:
        sales[f"sales_lag_{lag_days}"] = sales["sales"].shift(lag_days)

    rolling_window = prepare_parameters["rolling_window"]
    sales[f"sales_rolling_mean_{rolling_window}"] = (
        sales["sales"].shift(1).rolling(window=rolling_window).mean()
    )
    sales = sales.dropna().reset_index(drop=True)

    split_index = int(len(sales) * (1 - prepare_parameters["test_size"]))
    if split_index <= 0 or split_index >= len(sales):
        raise ValueError("test_size must leave at least one row in train and test sets.")

    processed_directory = Path("data/processed")
    processed_directory.mkdir(parents=True, exist_ok=True)
    sales.iloc[:split_index].to_csv(processed_directory / "train.csv", index=False)
    sales.iloc[split_index:].to_csv(processed_directory / "test.csv", index=False)
    print(
        f"Prepared {len(sales)} daily rows: {split_index} train, "
        f"{len(sales) - split_index} test."
    )
    print(f"Test dates: {sales.iloc[split_index]['date'].date()} to {sales.iloc[-1]['date'].date()}")


if __name__ == "__main__":
    main()