import math
from pathlib import Path
import sys

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from preprocess import (  # noqa: E402
    compute_price_columns,
    extract_city,
    infer_land_type,
    parse_land_extent,
    parse_price,
)


def test_parse_price_per_perch():
    value, basis = parse_price("Rs 1,590,000 per perch")
    assert value == 1_590_000
    assert basis == "per_perch"


def test_parse_price_total():
    value, basis = parse_price("Rs 25,000,000 total price")
    assert value == 25_000_000
    assert basis == "total"


def test_parse_land_extent_acres_and_perches():
    assert parse_land_extent("1 acre 20 perches land") == 180


def test_parse_land_extent_uses_detail_fields():
    assert parse_land_extent("Veloria - Bokundara", "Colombo, Bokundara", "6 perches onwards") == 6


def test_city_and_land_type_extraction():
    title = "10 Perches Residential Bare Land For Sale Maharagama"
    location = "Colombo, Land For Sale"
    assert extract_city(title, location) == "Maharagama"
    assert infer_land_type(title, location) == "Residential"


def test_compute_price_columns_keeps_per_perch_prices():
    df = pd.DataFrame(
        [
            {
                "numeric_price": 1_590_000,
                "price_basis": "per_perch",
                "land_extent_perches": 8,
            },
            {
                "numeric_price": 25_000_000,
                "price_basis": "total",
                "land_extent_perches": 10,
            },
        ]
    )

    result = compute_price_columns(df)
    assert result.loc[0, "price_per_perch_lkr"] == 1_590_000
    assert result.loc[0, "total_price_lkr"] == 12_720_000
    assert result.loc[1, "price_per_perch_lkr"] == 2_500_000
    assert result.loc[1, "total_price_lkr"] == 25_000_000
    assert not math.isnan(result.loc[1, "price_per_perch_lkr"])
