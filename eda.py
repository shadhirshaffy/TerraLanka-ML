from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_DIR = Path(__file__).resolve().parent
INPUT_PATH = PROJECT_DIR / "data" / "clean_lands.csv"
OUTPUT_DIR = PROJECT_DIR / "data"


def main():
    if not INPUT_PATH.exists():
        raise FileNotFoundError(
            f"Cleaned dataset not found: {INPUT_PATH}. Run preprocess.py first."
        )

    df = pd.read_csv(INPUT_PATH)
    required_columns = {
        "price_per_perch_lkr",
        "land_extent_perches",
        "location",
    }
    missing_columns = required_columns.difference(df.columns)
    if missing_columns:
        raise ValueError(
            f"Dataset is missing required columns: {', '.join(sorted(missing_columns))}"
        )

    df["price_per_perch_lkr"] = pd.to_numeric(
        df["price_per_perch_lkr"], errors="coerce"
    )
    df["land_extent_perches"] = pd.to_numeric(
        df["land_extent_perches"], errors="coerce"
    )
    df = df.dropna(subset=["price_per_perch_lkr", "land_extent_perches"])

    prices = df.loc[df["price_per_perch_lkr"] > 0, "price_per_perch_lkr"]
    if prices.empty:
        raise ValueError("No positive price_per_perch_lkr values to analyze.")

    log_prices = np.log1p(prices)
    print(f"Loaded {len(df)} usable records from {INPUT_PATH}")

    print("\nTarget distribution: price per perch (LKR)")
    print(prices.describe(percentiles=[0.25, 0.5, 0.75, 0.9, 0.95]).to_string())
    print(f"Skewness: {prices.skew():.3f}")

    print("\nLog-transformed target: log1p(price per perch)")
    print(log_prices.describe(percentiles=[0.25, 0.5, 0.75, 0.9, 0.95]).to_string())
    print(f"Skewness: {log_prices.skew():.3f}")

    if "district" in df and df["district"].nunique(dropna=True) > 1:
        df["analysis_district"] = df["district"].astype("string").str.strip()
    else:
        df["analysis_district"] = (
            df["location"].astype("string").str.split(",", n=1).str[0].str.strip()
        )
        if "district" in df:
            print(
                "\nNote: The district column does not distinguish locations; "
                "using the location prefix for the district comparison."
            )

    district_summary = (
        df.dropna(subset=["analysis_district"])
        .groupby("analysis_district")["price_per_perch_lkr"]
        .agg(listings="count", mean_price="mean", median_price="median")
        .sort_values("mean_price", ascending=False)
    )
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    district_output = OUTPUT_DIR / "eda_district_pricing.csv"
    district_summary.to_csv(district_output)
    print("\nDistrict pricing comparison (sorted by mean price per perch)")
    print(district_summary.to_string(float_format=lambda value: f"{value:,.2f}"))
    print(f"Saved district comparison to {district_output}")

    size_price = df[["land_extent_perches", "price_per_perch_lkr"]].dropna()
    print("\nLand extent vs. price per perch")
    print(
        "Pearson correlation: "
        f"{size_price['land_extent_perches'].corr(size_price['price_per_perch_lkr'], method='pearson'):.3f}"
    )
    ranked_size_price = size_price.rank(method="average")
    print(
        "Spearman correlation: "
        f"{ranked_size_price['land_extent_perches'].corr(ranked_size_price['price_per_perch_lkr'], method='pearson'):.3f}"
    )

    size_bins = [0, 5, 10, 20, np.inf]
    size_labels = ["0-<5", "5-<10", "10-<20", "20+"]
    size_price = size_price.loc[size_price["land_extent_perches"] > 0].copy()
    size_price["extent_group_perches"] = pd.cut(
        size_price["land_extent_perches"],
        bins=size_bins,
        labels=size_labels,
        right=False,
    )
    size_summary = (
        size_price.groupby("extent_group_perches", observed=False)["price_per_perch_lkr"]
        .agg(listings="count", mean_price="mean", median_price="median")
    )
    size_output = OUTPUT_DIR / "eda_size_price_groups.csv"
    size_summary.to_csv(size_output)
    print("\nPrice per perch by land-extent group (LKR)")
    print(size_summary.to_string(float_format=lambda value: f"{value:,.2f}"))
    print(f"Saved size/price comparison to {size_output}")


if __name__ == "__main__":
    main()
