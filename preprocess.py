import re
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_DIR = Path(__file__).resolve().parent
RAW_PATH = PROJECT_DIR / "raw_ikman_lands.csv"
CLEAN_PATH = PROJECT_DIR / "data" / "clean_lands.csv"

COLOMBO_CENTER_LAT = 6.9271
COLOMBO_CENTER_LON = 79.8612

CITY_COORDINATES = {
    "angoda": (6.9418, 79.9286),
    "athurugiriya": (6.8730, 79.9990),
    "battaramulla": (6.9006, 79.9220),
    "boralesgamuwa": (6.8427, 79.9006),
    "colombo": (6.9271, 79.8612),
    "dehiwala": (6.8513, 79.8656),
    "homagama": (6.8440, 80.0024),
    "kaduwela": (6.9350, 79.9840),
    "kesbewa": (6.7957, 79.9386),
    "kohuwala": (6.8678, 79.8840),
    "kolonnawa": (6.9329, 79.8848),
    "kotte": (6.8905, 79.9015),
    "maharagama": (6.8480, 79.9265),
    "malabe": (6.9061, 79.9696),
    "moratuwa": (6.7730, 79.8816),
    "mount lavinia": (6.8394, 79.8631),
    "nawala": (6.8901, 79.8877),
    "nugegoda": (6.8649, 79.8997),
    "padukka": (6.8422, 80.0901),
    "piliyandala": (6.8018, 79.9227),
    "rajagiriya": (6.9094, 79.8957),
    "ratmalana": (6.8195, 79.8801),
    "thalawathugoda": (6.8736, 79.9415),
    "wellampitiya": (6.9385, 79.8996),
}

KEYWORD_PATTERNS = {
    "kw_commercial": r"\bcommercial\b",
    "kw_residential": r"\bresidential\b",
    "kw_main_road": r"\bmain\s+road\b|\broad\s+front\b|\broadside\b",
    "kw_bare_land": r"\bbare\s+land\b",
    "kw_lake": r"\blake\b|\bwaterfront\b",
    "kw_corner": r"\bcorner\b",
    "kw_approved": r"\bapproved\b|\bapproval\b",
    "kw_prime": r"\bprime\b|\bhighly\s+residential\b",
}


def normalize_text(value):
    if not isinstance(value, str):
        return ""
    return re.sub(r"\s+", " ", value.strip().lower())


def infer_land_type(title, location):
    text = f"{normalize_text(title)} {normalize_text(location)}"
    if re.search(r"\bcommercial\b|\bshop\b|\boffice\b|\bbusiness\b", text):
        return "Commercial"
    if re.search(r"\bindustrial\b|\bfactory\b|\bwarehouse\b", text):
        return "Industrial"
    if re.search(r"\bagricultural\b|\bagriculture\b|\bpaddy\b|\bcoconut\b|\bcinnamon\b", text):
        return "Agricultural"
    if re.search(r"\bresidential\b|\bhouse\b|\bhousing\b|\bhome\b|\bvilla\b", text):
        return "Residential"
    if re.search(r"\bbare\s+land\b|\bland\s+for\s+sale\b", text):
        return "Bare Land"
    return "Unknown"


def parse_price(raw_price_str):
    text = normalize_text(raw_price_str).replace(",", "")
    numbers = re.findall(r"\d+(?:\.\d+)?", text)
    if not numbers:
        return np.nan, "unknown"

    value = float(numbers[0])
    if "million" in text or re.search(r"\bmn\b", text):
        value *= 1_000_000
    elif "lakh" in text or "lakhs" in text:
        value *= 100_000

    if re.search(r"\bper\s*perch\b|\bper\s*p\b|\bp\.?p\b", text):
        return value, "per_perch"
    if "total" in text:
        return value, "total"
    return value, "unknown"


def parse_land_extent(*values):
    text = normalize_text(" ".join(str(value) for value in values if isinstance(value, str)))
    if not text:
        return np.nan

    total_perches = 0.0
    found = False

    for match in re.finditer(r"(\d+(?:\.\d+)?)\s*(?:acre|acres|acr|ac\b)", text):
        total_perches += float(match.group(1)) * 160.0
        found = True

    for match in re.finditer(r"(\d+(?:\.\d+)?)\s*(?:perch|perches|p\b)", text):
        total_perches += float(match.group(1))
        found = True

    return total_perches if found and total_perches > 0 else np.nan


def extract_district(location):
    text = normalize_text(location)
    if not text:
        return "unknown"
    return text.split(",", 1)[0].strip() or "unknown"


def extract_city(title, location):
    title_text = normalize_text(title)
    location_text = normalize_text(location)
    combined = f"{title_text} {location_text}"

    for city in sorted(CITY_COORDINATES, key=len, reverse=True):
        if re.search(rf"\b{re.escape(city)}\b", combined):
            return city.title()

    district = extract_district(location_text)
    return district.title() if district != "unknown" else "Unknown"


def haversine_km(lat1, lon1, lat2, lon2):
    radius_km = 6371.0
    lat1_rad, lon1_rad, lat2_rad, lon2_rad = map(
        np.radians, [lat1, lon1, lat2, lon2]
    )
    dlat = lat2_rad - lat1_rad
    dlon = lon2_rad - lon1_rad
    a = (
        np.sin(dlat / 2.0) ** 2
        + np.cos(lat1_rad) * np.cos(lat2_rad) * np.sin(dlon / 2.0) ** 2
    )
    return 2 * radius_km * np.arcsin(np.sqrt(a))


def estimate_distance_to_colombo(city):
    coords = CITY_COORDINATES.get(normalize_text(city))
    if not coords:
        return np.nan
    return haversine_km(COLOMBO_CENTER_LAT, COLOMBO_CENTER_LON, coords[0], coords[1])


def add_keyword_features(df):
    text = (
        df["title"].fillna("").astype(str)
        + " "
        + df["location"].fillna("").astype(str)
    ).str.lower()

    for column, pattern in KEYWORD_PATTERNS.items():
        df[column] = text.str.contains(pattern, regex=True, na=False).astype(int)
    return df


def compute_price_columns(df):
    df["price_per_perch_lkr"] = np.nan
    df["total_price_lkr"] = np.nan

    per_perch_mask = df["price_basis"] == "per_perch"
    total_mask = df["price_basis"] == "total"
    unknown_mask = df["price_basis"] == "unknown"

    df.loc[per_perch_mask, "price_per_perch_lkr"] = df.loc[
        per_perch_mask, "numeric_price"
    ]
    df.loc[per_perch_mask, "total_price_lkr"] = (
        df.loc[per_perch_mask, "numeric_price"]
        * df.loc[per_perch_mask, "land_extent_perches"]
    )

    df.loc[total_mask, "total_price_lkr"] = df.loc[total_mask, "numeric_price"]
    df.loc[total_mask, "price_per_perch_lkr"] = (
        df.loc[total_mask, "numeric_price"]
        / df.loc[total_mask, "land_extent_perches"]
    )

    unknown_per_perch = unknown_mask & (df["numeric_price"] < 15_000_000)
    unknown_total = unknown_mask & ~unknown_per_perch

    df.loc[unknown_per_perch, "price_per_perch_lkr"] = df.loc[
        unknown_per_perch, "numeric_price"
    ]
    df.loc[unknown_per_perch, "total_price_lkr"] = (
        df.loc[unknown_per_perch, "numeric_price"]
        * df.loc[unknown_per_perch, "land_extent_perches"]
    )

    df.loc[unknown_total, "total_price_lkr"] = df.loc[unknown_total, "numeric_price"]
    df.loc[unknown_total, "price_per_perch_lkr"] = (
        df.loc[unknown_total, "numeric_price"]
        / df.loc[unknown_total, "land_extent_perches"]
    )
    return df


def clean_dataset(input_path=RAW_PATH, output_path=CLEAN_PATH):
    input_path = Path(input_path)
    output_path = Path(output_path)
    if not input_path.exists():
        raise FileNotFoundError(f"Raw dataset not found at {input_path}. Run scraper.py first.")

    print("Loading raw dataset...")
    df = pd.read_csv(input_path)
    initial_count = len(df)

    expected_columns = {"title", "raw_price", "location"}
    missing_columns = expected_columns.difference(df.columns)
    if missing_columns:
        raise ValueError(f"Raw dataset is missing columns: {', '.join(sorted(missing_columns))}")

    df = df.dropna(how="all").drop_duplicates(
        subset=["title", "raw_price", "location"]
    )
    print(f"Dropped {initial_count - len(df)} duplicate or empty records.")

    df["title_clean"] = df["title"].fillna("").astype(str).map(normalize_text)
    df["district"] = df["location"].map(extract_district).str.title()
    df["city"] = [extract_city(title, location) for title, location in zip(df["title"], df["location"])]
    df["land_type"] = [
        infer_land_type(title, location) for title, location in zip(df["title"], df["location"])
    ]
    df["land_extent_perches"] = [
        parse_land_extent(title, location) for title, location in zip(df["title"], df["location"])
    ]

    parsed_prices = df["raw_price"].apply(parse_price)
    df["numeric_price"] = [result[0] for result in parsed_prices]
    df["price_basis"] = [result[1] for result in parsed_prices]

    df = compute_price_columns(df)
    df = add_keyword_features(df)
    df["distance_to_colombo_km"] = df["city"].map(estimate_distance_to_colombo)
    df["distance_to_colombo_km"] = df["distance_to_colombo_km"].fillna(
        df["distance_to_colombo_km"].median()
    )
    df["log_land_extent_perches"] = np.log1p(df["land_extent_perches"])

    before_filter = len(df)
    df = df.dropna(
        subset=[
            "price_per_perch_lkr",
            "land_extent_perches",
            "total_price_lkr",
            "numeric_price",
        ]
    )
    df = df[(df["land_extent_perches"] > 0.5) & (df["land_extent_perches"] < 500)]
    df = df[
        (df["price_per_perch_lkr"] > 10_000)
        & (df["price_per_perch_lkr"] < 100_000_000)
    ]
    print(f"Filtered out {before_filter - len(df)} anomalous or incomplete records.")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"Successfully saved {len(df)} cleaned records to {output_path}.")
    print(
        df[
            [
                "city",
                "district",
                "land_type",
                "land_extent_perches",
                "price_basis",
                "price_per_perch_lkr",
                "distance_to_colombo_km",
            ]
        ].head()
    )


if __name__ == "__main__":
    clean_dataset()
