# Data Files

This folder contains generated data artifacts for TerraLanka ML.

## Files

- `clean_lands.csv` - cleaned training dataset produced by `python preprocess.py`
- `raw_ikman_lands_checkpoint.csv` - scraper checkpoint written during longer scraping runs
- `eda_district_pricing.csv` - district/city pricing summary from EDA
- `eda_size_price_groups.csv` - land-size pricing summary from EDA
- `evaluation/model_comparison.csv` - holdout metrics for baseline and model candidates
- `evaluation/holdout_predictions.csv` - actual vs predicted holdout rows for the selected model

The raw source file currently lives at the project root:

- `raw_ikman_lands.csv`

## Important Columns In `clean_lands.csv`

- `title` - listing title from the source site
- `raw_price` - original price text
- `location` - original location text
- `district` - parsed district/location prefix
- `city` - parsed suburb/city, usually inferred from title
- `land_type` - inferred category such as Residential, Commercial, Bare Land, Agricultural, or Unknown
- `land_extent_perches` - normalized land extent in perches
- `price_basis` - `per_perch`, `total`, or `unknown`
- `price_per_perch_lkr` - model target
- `total_price_lkr` - normalized total estimated/listed price
- `distance_to_colombo_km` - approximate suburb distance proxy when known
- `kw_*` columns - binary keyword features

## Cleaning Rules

Price handling is critical:

- `Rs X per perch` keeps `X` as `price_per_perch_lkr`
- `Rs X total price` computes `price_per_perch_lkr = X / land_extent_perches`

Rows with missing target values, invalid land extent, or extreme prices are removed during preprocessing.
