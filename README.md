# TerraLanka ML Land Price Estimator

TerraLanka ML estimates land price per perch for Sri Lankan land listings, with a current focus on Colombo-area listings. The project includes scraping, preprocessing, model training, evaluation, and a FastAPI web app.

## Project Structure

- `scrape.py` - scrapes land listings into `raw_ikman_lands.csv`
- `preprocess.py` - cleans raw listings and builds model features
- `train.py` - trains and compares multiple regression models
- `evaluate.py` - reruns holdout evaluation and baseline checks
- `app.py` - FastAPI app and browser UI
- `data/` - cleaned datasets, checkpoints, and evaluation reports
- `models/` - trained model artifact and metadata
- `tests/` - focused preprocessing tests
- `archive/` - older experimental scraper files kept for reference

## Setup

```powershell
python -m pip install -r requirements.txt
```

## Workflow

1. Scrape listings:

```powershell
python scrape.py --pages 50
```

2. Clean and feature-engineer the dataset:

```powershell
python preprocess.py
```

3. Train and compare models:

```powershell
python train.py
```

4. Run evaluation reports:

```powershell
python evaluate.py
```

5. Start the app:

```powershell
python -m uvicorn app:app --host 127.0.0.1 --port 8001
```

Open `http://127.0.0.1:8001`.

## Model Features

The model uses:

- numeric features: land extent, log land extent, distance to Colombo, listing keyword flags
- categorical features: district, city, land type, price basis
- text features: cleaned title/description keywords through TF-IDF

The target is `log1p(price_per_perch_lkr)`, converted back to LKR with `expm1` for predictions.

## Evaluation

The project compares each model against a city-median baseline:

```text
Predict median price per perch for each city.
```

Reports are saved to:

- `data/evaluation/model_comparison.csv`
- `data/evaluation/holdout_predictions.csv`

The selected model must beat the city-median baseline on MAE.

## App Output

The app returns:

- estimated price per perch
- estimated total land value
- low / expected / high confidence band
- number of cleaned training listings for the selected city
- warning when location data is sparse

## Tests

```powershell
python -m pytest
```

The tests focus on preprocessing correctness, especially `per perch` vs `total price` handling.
