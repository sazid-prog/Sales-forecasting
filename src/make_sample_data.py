"""Create three years of deterministic synthetic daily sales."""

from pathlib import Path

import numpy as np
import pandas as pd


def main() -> None:
    random_generator = np.random.default_rng(42)
    dates = pd.date_range(start="2021-01-01", periods=1095, freq="D")
    day_number = np.arange(len(dates))
    yearly_seasonality = 20 * np.sin(2 * np.pi * day_number / 365.25)
    weekend_effect = np.where(dates.dayofweek >= 5, 15, 0)
    noise = random_generator.normal(loc=0, scale=8, size=len(dates))

    sales = 100 + 0.05 * day_number + yearly_seasonality + weekend_effect + noise
    sample_data = pd.DataFrame({"date": dates, "sales": np.round(sales, 2)})

    output_path = Path("data/raw/sales.csv")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sample_data.to_csv(output_path, index=False)
    print(f"Generated {len(sample_data)} daily rows at {output_path}")


if __name__ == "__main__":
    main()
