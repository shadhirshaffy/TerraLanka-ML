from pathlib import Path
import re
import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

# Initialize FastAPI app
app = FastAPI(
    title="TerraLanka ML - Land Price Prediction API",
    description="API for predicting land prices per perch in Colombo, Sri Lanka.",
    version="1.0.0"
)
app.mount("/assets", StaticFiles(directory=Path(__file__).resolve().parent / "assets"), name="assets")

# Load trained model artifact
PROJECT_DIR = Path(__file__).resolve().parent
MODEL_PATH = PROJECT_DIR / "models" / "colombo_land_model.pkl"
CLEAN_DATA_PATH = PROJECT_DIR / "data" / "clean_lands.csv"

if not MODEL_PATH.exists():
    raise RuntimeError(f"Model artifact not found at {MODEL_PATH}. Please train the model first using train_colombo.py.")

artifact = joblib.load(MODEL_PATH)
model = artifact["model"]
feature_columns = artifact["feature_columns"]
model_name = str(artifact.get("model_name", "trained model")).replace("_", " ").title()
training_rows = int(artifact.get("training_rows", 0) or 0)
model_metrics = artifact.get("metrics", {})
model_mae = float(model_metrics.get("mae", 0) or 0)

if CLEAN_DATA_PATH.exists():
    city_counts = (
        pd.read_csv(CLEAN_DATA_PATH)["city"]
        .fillna("Unknown")
        .astype(str)
        .str.casefold()
        .value_counts()
        .to_dict()
    )
else:
    city_counts = {}

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
        return 0.0
    return haversine_km(6.9271, 79.8612, coords[0], coords[1])


def build_prediction_features(request):
    city = request.city or "Colombo"
    district = request.district or "Colombo"
    land_type = request.land_type or "Unknown"
    details = " ".join(
        value
        for value in [
            request.title or "",
            request.description or "",
            "main road" if request.has_main_road_access else "",
            "residential" if request.is_residential else "",
            "commercial" if request.is_commercial else "",
        ]
        if value
    )
    title_clean = normalize_text(details or city)
    keyword_text = f"{title_clean} {normalize_text(city)} {normalize_text(district)} {normalize_text(land_type)}"

    row = {
        "land_extent_perches": request.land_extent_perches,
        "log_land_extent_perches": np.log1p(request.land_extent_perches),
        "distance_to_colombo_km": estimate_distance_to_colombo(city),
        "district": district.title(),
        "city": city.title(),
        "land_type": land_type.title(),
        "price_basis": "unknown",
        "title_clean": title_clean,
    }
    for column, pattern in KEYWORD_PATTERNS.items():
        row[column] = int(bool(re.search(pattern, keyword_text)))

    return pd.DataFrame([{column: row.get(column, 0) for column in feature_columns}])


def build_confidence_band(price_per_perch, land_extent_perches):
    if model_mae > 0:
        low_per_perch = max(price_per_perch - model_mae, price_per_perch * 0.65, 0)
        high_per_perch = price_per_perch + model_mae
    else:
        low_per_perch = price_per_perch * 0.75
        high_per_perch = price_per_perch * 1.25

    return {
        "low_price_per_perch_lkr": round(low_per_perch, 2),
        "expected_price_per_perch_lkr": round(price_per_perch, 2),
        "high_price_per_perch_lkr": round(high_per_perch, 2),
        "low_total_price_lkr": round(low_per_perch * land_extent_perches, 2),
        "expected_total_price_lkr": round(price_per_perch * land_extent_perches, 2),
        "high_total_price_lkr": round(high_per_perch * land_extent_perches, 2),
    }


def build_location_warning(city):
    listing_count = city_counts.get(normalize_text(city), 0)
    if listing_count == 0:
        return (
            listing_count,
            "We do not yet have cleaned examples for this city, so this estimate is based on broader market patterns.",
        )
    if listing_count < 5:
        return (
            listing_count,
            f"We found only {listing_count} cleaned examples for this city. Use the estimate as a rough guide and compare nearby listings.",
        )
    return listing_count, None

# Define request schema
class LandPredictionRequest(BaseModel):
    land_extent_perches: float
    city: str | None = "Colombo"
    district: str | None = "Colombo"
    land_type: str | None = "Unknown"
    has_main_road_access: bool = False
    is_residential: bool = False
    is_commercial: bool = False
    title: str | None = None
    description: str | None = None

# Define response schema
class LandPredictionResponse(BaseModel):
    district: str
    city: str
    land_type: str
    land_extent_perches: float
    estimated_price_per_perch_lkr: float
    estimated_total_price_lkr: float
    confidence_band: dict[str, float]
    training_listings_for_city: int
    model_name: str
    model_training_rows: int
    model_mae_lkr: float
    confidence_note: str
    warning: str | None = None

@app.get("/")
def home() -> HTMLResponse:
    return HTMLResponse(
        """
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>TerraLanka | Land Price Estimator</title>
  <style>
    :root {
      --ink: #f8fafc;
      --muted: #cbd5e1;
      --canvas: #020617;
      --paper: rgba(255,255,255,.1);
      --field: rgba(255,255,255,.1);
      --green: #22d3ee;
      --green-dark: #67e8f9;
      --gold: #a5f3fc;
      --line: rgba(255,255,255,.2);
      --panel: rgba(255,255,255,.1);
      --glow: 0 0 34px rgba(34,211,238,.28);
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      min-height: 100vh;
      color: var(--ink);
      background:
        radial-gradient(circle at 18% 12%, rgba(59,130,246,.55), transparent 30%),
        radial-gradient(circle at 85% 6%, rgba(168,85,247,.55), transparent 28%),
        radial-gradient(circle at 52% 86%, rgba(34,211,238,.32), transparent 34%),
        linear-gradient(135deg, #020617 0%, #1e1b4b 46%, #312e81 100%);
      background-attachment: fixed;
      font-family: Inter, ui-sans-serif, system-ui, -apple-system, sans-serif;
    }
    .shell { width: min(1120px, 92vw); margin: 0 auto; padding: 26px 0 48px; }
    .topbar { display: flex; justify-content: space-between; align-items: center; margin-bottom: 34px; padding: 16px 18px; border: 1px solid var(--line); border-radius: 18px; background: rgba(255,255,255,.08); backdrop-filter: blur(18px); box-shadow: 0 18px 60px rgba(0,0,0,.22); }
    .brand { display: flex; align-items: center; gap: 12px; font-weight: 800; letter-spacing: -0.03em; }
    .brand-logo {
      width: 48px; height: 48px; object-fit: contain;
      filter: drop-shadow(0 0 18px rgba(34,211,238,.28));
    }
    .tag { color: #a5f3fc; font-size: 13px; font-weight: 800; }
    .hero { max-width: 760px; margin-bottom: 24px; }
    .eyebrow { color: var(--gold); font-size: 12px; font-weight: 800; letter-spacing: .16em; text-transform: uppercase; }
    h1 { margin: 9px 0 10px; font-size: clamp(34px, 5vw, 56px); line-height: 1; letter-spacing: -0.04em; }
    .hero p { max-width: 620px; margin: 0; color: var(--muted); font-size: 16px; line-height: 1.55; }
    .grid { display: grid; grid-template-columns: minmax(0, 1.08fr) minmax(360px, .92fr); gap: 18px; align-items: start; }
    .card { padding: 24px; background: var(--paper); border: 1px solid var(--line); border-radius: 24px; box-shadow: 0 24px 70px rgba(0,0,0,.28); backdrop-filter: blur(20px); }
    .card h2 { margin: 0 0 8px; font-size: 22px; letter-spacing: -.03em; }
    .card-copy { margin: 0 0 26px; color: var(--muted); line-height: 1.5; }
    .form-section { padding: 22px 0; border-top: 1px solid var(--line); }
    .form-section:first-of-type { padding-top: 0; border-top: 0; }
    .section-title { margin: 0 0 14px; font-size: 15px; font-weight: 900; letter-spacing: .01em; }
    .section-note { margin: -6px 0 16px; color: var(--muted); font-size: 13px; line-height: 1.45; }
    label { display: block; margin-bottom: 9px; font-size: 13px; font-weight: 800; }
    .input-wrap { position: relative; }
    input, select, textarea {
      width: 100%; padding: 14px 54px 14px 14px; color: var(--ink);
      background: var(--field); border: 1px solid var(--line); border-radius: 16px;
      font: inherit; font-size: 18px; outline: none; transition: .2s;
    }
    option { color: #0f172a; }
    textarea { min-height: 92px; resize: vertical; padding-right: 16px; }
    input:focus, select:focus, textarea:focus { border-color: var(--green); box-shadow: 0 0 0 4px rgba(34,211,238,.16), var(--glow); }
    .unit { position: absolute; top: 50%; right: 16px; color: var(--muted); font-size: 13px; transform: translateY(-50%); }
    .split { display: grid; grid-template-columns: 1fr 1fr; gap: 14px; }
    .field { min-width: 0; }
    .checks { display: grid; gap: 10px; margin-top: 16px; }
    .check {
      display: flex; align-items: center; gap: 10px; padding: 12px 13px;
      background: var(--field); border: 1px solid var(--line); border-radius: 16px;
      color: var(--ink); font-size: 14px; font-weight: 700;
    }
    .check input { width: 18px; height: 18px; padding: 0; accent-color: var(--green); }
    details { padding: 18px 0 0; border-top: 1px solid var(--line); }
    summary { cursor: pointer; color: var(--green); font-weight: 900; }
    .advanced-grid { display: grid; gap: 16px; margin-top: 16px; }
    button {
      width: 100%; margin-top: 22px; padding: 15px; color: #020617; background: var(--green);
      border: 0; border-radius: 16px; font: inherit; font-weight: 900; cursor: pointer; box-shadow: var(--glow);
      transition: background .2s, transform .2s;
    }
    button:hover { background: var(--green-dark); transform: translateY(-1px); }
    button:disabled { cursor: wait; opacity: .65; transform: none; }
    .actions { display: grid; grid-template-columns: 1fr 1fr; gap: 10px; margin-top: 10px; }
    .secondary-button {
      margin-top: 0; color: #cffafe; background: rgba(255,255,255,.08);
      border: 1px solid rgba(255,255,255,.18); box-shadow: none;
    }
    .secondary-button:hover { background: rgba(255,255,255,.14); }
    .hint { margin: 13px 0 0; color: var(--muted); font-size: 12px; }
    .recent { display: none; margin-top: 14px; padding: 12px; color: #cbd5e1; background: rgba(255,255,255,.07); border: 1px solid rgba(255,255,255,.12); border-radius: 16px; font-size: 13px; line-height: 1.45; }
    .recent strong { color: #fff; }
    .results { position: sticky; top: 24px; display: grid; align-content: center; min-height: 360px; background: var(--panel); color: white; box-shadow: 0 24px 70px rgba(0,0,0,.3), var(--glow); }
    .results h2 { color: #dcecdf; }
    .results-copy { color: #b9d1c3; }
    .placeholder { color: #b9d1c3; line-height: 1.6; }
    .result-top { padding-bottom: 18px; }
    .estimate-label { color: #b9d1c3; font-size: 12px; font-weight: 800; text-transform: uppercase; letter-spacing: .08em; }
    .estimate-value { margin-top: 7px; color: white; font-size: clamp(36px, 5vw, 52px); line-height: 1; font-weight: 900; letter-spacing: -.04em; }
    .sub-estimate { margin-top: 8px; color: #d7eadc; font-size: 14px; font-weight: 700; }
    .metric { padding: 17px 0; border-top: 1px solid rgba(255,255,255,.18); }
    .metric-label { color: #b9d1c3; font-size: 12px; font-weight: 700; text-transform: uppercase; letter-spacing: .08em; }
    .metric-value { margin-top: 5px; font-size: clamp(25px, 4vw, 36px); font-weight: 800; letter-spacing: -.04em; }
    .trust-row { display: flex; flex-wrap: wrap; gap: 8px; margin: 10px 0 4px; }
    .badge { display: inline-flex; align-items: center; padding: 7px 10px; border-radius: 999px; background: rgba(34,211,238,.12); color: #cffafe; border: 1px solid rgba(103,232,249,.28); font-size: 12px; font-weight: 800; }
    .confidence-head { display: flex; align-items: center; justify-content: space-between; gap: 12px; }
    .tooltip { position: relative; display: inline-flex; align-items: center; justify-content: center; width: 20px; height: 20px; margin-left: 6px; border-radius: 999px; background: rgba(255,255,255,.12); color: #cffafe; font-size: 12px; cursor: help; }
    .tooltip:hover::after {
      content: attr(data-tip); position: absolute; right: 0; bottom: calc(100% + 8px);
      width: min(260px, 70vw); padding: 10px 12px; color: #e2e8f0;
      background: rgba(15,23,42,.96); border: 1px solid rgba(255,255,255,.16);
      border-radius: 12px; font-size: 12px; line-height: 1.35; text-transform: none;
      letter-spacing: 0; z-index: 5; box-shadow: 0 18px 40px rgba(0,0,0,.28);
    }
    .confidence-badge { padding: 6px 10px; border-radius: 999px; background: rgba(34,211,238,.14); color: #cffafe; border: 1px solid rgba(103,232,249,.24); font-size: 11px; font-weight: 900; text-transform: uppercase; letter-spacing: .08em; white-space: nowrap; }
    .band { display: grid; grid-template-columns: repeat(3, 1fr); gap: 8px; margin-top: 14px; }
    .band-item { padding: 12px; background: rgba(255,255,255,.08); border: 1px solid rgba(255,255,255,.14); border-radius: 16px; }
    .band-item.expected { background: rgba(34,211,238,.16); outline: 1px solid rgba(103,232,249,.24); box-shadow: var(--glow); }
    .band-label { color: #b9d1c3; font-size: 11px; font-weight: 800; text-transform: uppercase; letter-spacing: .08em; }
    .band-value { margin-top: 4px; color: white; font-size: 15px; font-weight: 800; }
    .loading-state { padding: 22px 0; color: #d7eadc; font-weight: 800; }
    .error-state { padding: 13px; color: #ffd4ce; background: rgba(181,63,53,.18); border-radius: 8px; font-size: 13px; font-weight: 800; }
    .note { margin-top: 10px; color: #cfe5d6; font-size: 13px; line-height: 1.45; }
    .model-panel { margin-top: 14px; padding: 13px; background: rgba(255,255,255,.08); border: 1px solid rgba(255,255,255,.14); border-radius: 16px; }
    .model-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 10px; margin-top: 10px; }
    .model-stat { padding: 10px; background: rgba(255,255,255,.07); border-radius: 14px; }
    .model-stat strong { display: block; color: white; font-size: 15px; }
    .model-stat span { color: #b9d1c3; font-size: 11px; font-weight: 800; text-transform: uppercase; letter-spacing: .08em; }
    .warning { margin-top: 14px; padding: 12px; color: #fef3c7; background: rgba(251,191,36,.12); border: 1px solid rgba(251,191,36,.24); border-radius: 16px; font-size: 13px; line-height: 1.4; }
    .disclaimer { margin-top: 14px; padding: 12px; color: #cbd5e1; background: rgba(255,255,255,.06); border: 1px solid rgba(255,255,255,.12); border-radius: 16px; font-size: 12px; line-height: 1.45; }
    .result-actions { display: grid; grid-template-columns: 1fr 1fr; gap: 10px; margin-top: 14px; }
    .error { margin-top: 14px; color: #b53f35; font-size: 13px; font-weight: 700; }
    .results .error { color: #ffd4ce; }
    @media (max-width: 720px) {
      .topbar { margin-bottom: 42px; }
      .hero { margin-bottom: 20px; }
      .grid { grid-template-columns: 1fr; }
      .results { position: static; min-height: 288px; }
      .split { grid-template-columns: 1fr; }
      .band { grid-template-columns: 1fr; }
      .model-grid { grid-template-columns: 1fr; }
      .card { padding: 23px; }
    }
  </style>
</head>
<body>
  <main class="shell">
    <nav class="topbar">
      <div class="brand"><img class="brand-logo" src="/assets/terralanka-ml-logo.png" alt="TerraLanka ML logo"> TerraLanka ML</div>
      <span class="tag">COLOMBO · SRI LANKA</span>
    </nav>
    <section class="hero">
      <div class="eyebrow">Data-informed property decisions</div>
      <h1>Estimate your land's value.</h1>
      <p>Enter the land extent below to get an instant estimate based on our Colombo land-price model.</p>
    </section>
    <section class="grid">
      <form class="card" id="prediction-form">
        <h2>Estimate setup</h2>
        <p class="card-copy">Add the basic property details first. Optional listing signals can improve the estimate when you know them.</p>

        <section class="form-section">
          <h3 class="section-title">Location</h3>
          <div class="split">
            <div class="field">
              <label for="district">Property district</label>
              <div class="input-wrap">
                <select id="district" name="district" required>
                  <option value="Colombo" selected>Colombo</option>
                  <option value="Gampaha">Gampaha</option>
                  <option value="Kalutara">Kalutara</option>
                  <option value="Badulla">Badulla</option>
                </select>
              </div>
            </div>
            <div class="field">
              <label for="city">Nearest city or suburb</label>
              <div class="input-wrap">
                <input id="city" name="city" type="text" list="city-options" placeholder="Maharagama" required>
                <datalist id="city-options">
                  <option value="Colombo">
                  <option value="Maharagama">
                  <option value="Nugegoda">
                  <option value="Piliyandala">
                  <option value="Battaramulla">
                  <option value="Dehiwala">
                  <option value="Rajagiriya">
                  <option value="Thalawathugoda">
                  <option value="Malabe">
                  <option value="Boralesgamuwa">
                  <option value="Mount Lavinia">
                  <option value="Kohuwala">
                  <option value="Athurugiriya">
                </datalist>
              </div>
            </div>
          </div>
        </section>

        <section class="form-section">
          <h3 class="section-title">Land details</h3>
          <div class="split">
            <div class="field">
              <label for="extent">Land extent</label>
              <div class="input-wrap">
                <input id="extent" name="extent" type="number" min="0.1" step="0.1" placeholder="10" required>
                <span class="unit">perches</span>
              </div>
            </div>
            <div class="field">
              <label for="land-type">Primary land use</label>
              <div class="input-wrap">
                <select id="land-type" name="land-type" required>
                  <option value="Unknown">Unknown</option>
                  <option value="Residential">Residential</option>
                  <option value="Commercial">Commercial</option>
                  <option value="Bare Land">Bare land</option>
                  <option value="Agricultural">Agricultural</option>
                  <option value="Industrial">Industrial</option>
                </select>
              </div>
            </div>
          </div>
        </section>

        <section class="form-section">
          <h3 class="section-title">Listing signals</h3>
          <p class="section-note">Select any details that appear in the listing or are known about the property.</p>
          <div class="checks">
            <label class="check"><input id="main-road" type="checkbox">Access from a main road</label>
            <label class="check"><input id="residential" type="checkbox">Suitable for residential use</label>
            <label class="check"><input id="commercial" type="checkbox">Suitable for commercial use</label>
          </div>
        </section>

        <details>
          <summary>Add optional listing text</summary>
          <div class="advanced-grid">
            <div>
              <label for="title">Key selling points</label>
              <div class="input-wrap">
                <input id="title" name="title" type="text" placeholder="approved bare land near main road">
              </div>
            </div>
            <div>
              <label for="description">Extra notes</label>
              <div class="input-wrap">
                <textarea id="description" name="description" placeholder="Road width, plan approvals, nearby landmarks, access notes"></textarea>
              </div>
            </div>
          </div>
        </details>

        <button id="submit" type="submit">Estimate land value →</button>
        <div class="actions">
          <button id="example-button" class="secondary-button" type="button">Use sample</button>
          <button id="reset-button" class="secondary-button" type="button">Reset form</button>
        </div>
        <p class="hint">Use this as an early guide. Always compare with recent local listings before making decisions.</p>
        <div id="recent-summary" class="recent"></div>
        <div class="error" id="form-error" role="alert"></div>
      </form>
      <section class="card results" aria-live="polite">
        <h2>Valuation summary</h2>
        <p class="card-copy results-copy">Your estimate will appear here after you submit the property details.</p>
        <div id="metrics" class="placeholder">Expected value, price per perch, range, and data coverage will be shown here.</div>
      </section>
    </section>
  </main>
  <script>
    const form = document.getElementById("prediction-form");
    const input = document.getElementById("extent");
    const districtInput = document.getElementById("district");
    const cityInput = document.getElementById("city");
    const landTypeInput = document.getElementById("land-type");
    const mainRoadInput = document.getElementById("main-road");
    const residentialInput = document.getElementById("residential");
    const commercialInput = document.getElementById("commercial");
    const titleInput = document.getElementById("title");
    const descriptionInput = document.getElementById("description");
    const button = document.getElementById("submit");
    const exampleButton = document.getElementById("example-button");
    const resetButton = document.getElementById("reset-button");
    const recentSummary = document.getElementById("recent-summary");
    const error = document.getElementById("form-error");
    const metrics = document.getElementById("metrics");
    let latestEstimateText = "";
    const money = value => new Intl.NumberFormat("en-LK", {
      style: "currency", currency: "LKR", maximumFractionDigits: 0
    }).format(value);
    const number = value => new Intl.NumberFormat("en-LK", {
      maximumFractionDigits: 0
    }).format(value);
    const compactMoney = value => {
      const abs = Math.abs(value);
      if (abs >= 1_000_000) return `LKR ${(value / 1_000_000).toFixed(1)}M`;
      if (abs >= 100_000) return `LKR ${(value / 100_000).toFixed(1)}L`;
      return money(value);
    };

    exampleButton.addEventListener("click", () => {
      districtInput.value = "Colombo";
      cityInput.value = "Maharagama";
      input.value = "10";
      landTypeInput.value = "Residential";
      mainRoadInput.checked = true;
      residentialInput.checked = true;
      commercialInput.checked = false;
      titleInput.value = "approved bare land near main road";
      descriptionInput.value = "Residential land with good access and nearby amenities.";
      error.textContent = "";
    });

    resetButton.addEventListener("click", () => {
      form.reset();
      districtInput.value = "Colombo";
      landTypeInput.value = "Unknown";
      error.textContent = "";
      latestEstimateText = "";
      recentSummary.style.display = "none";
      metrics.className = "placeholder";
      metrics.textContent = "Expected value, price per perch, range, and data coverage will be shown here.";
    });

    form.addEventListener("submit", async event => {
      event.preventDefault();
      error.textContent = "";
      const extent = Number(input.value);
      if (!Number.isFinite(extent) || extent <= 0) {
        error.textContent = "Enter a land extent greater than zero.";
        input.focus();
        return;
      }
      button.disabled = true;
      button.textContent = "Estimating…";
      metrics.className = "";
      metrics.innerHTML = `<div class="loading-state">Estimating value for ${cityInput.value || "Colombo"}...</div>`;
      try {
        const response = await fetch("/predict", {
          method: "POST",
          headers: {"Content-Type": "application/json"},
          body: JSON.stringify({
            land_extent_perches: extent,
            city: cityInput.value || "Colombo",
            district: districtInput.value || "Colombo",
            land_type: landTypeInput.value || "Unknown",
            has_main_road_access: mainRoadInput.checked,
            is_residential: residentialInput.checked,
            is_commercial: commercialInput.checked,
            title: titleInput.value || cityInput.value || "Colombo",
            description: descriptionInput.value || ""
          })
        });
        const payload = await response.json();
        if (!response.ok) throw new Error(payload.detail || "Unable to calculate estimate.");
        latestEstimateText = [
          `TerraLanka ML estimate for ${payload.city}`,
          `Land extent: ${number(payload.land_extent_perches)} perches`,
          `Expected total value: ${money(payload.estimated_total_price_lkr)}`,
          `Price per perch: ${money(payload.estimated_price_per_perch_lkr)}`,
          `Likely range: ${money(payload.confidence_band.low_total_price_lkr)} - ${money(payload.confidence_band.high_total_price_lkr)}`,
        ].join("\\n");
        recentSummary.style.display = "block";
        recentSummary.innerHTML = `Recent estimate: <strong>${payload.city}</strong>, ${number(payload.land_extent_perches)} perches · <strong>${compactMoney(payload.estimated_total_price_lkr)}</strong>`;
        metrics.className = "";
        metrics.innerHTML = `
          <div class="trust-row">
            <span class="badge">${payload.city}</span>
            <span class="badge">${payload.land_type}</span>
            <span class="badge">${payload.training_listings_for_city} city listings</span>
          </div>
          <div class="result-top">
            <div class="estimate-label">Expected market value</div>
            <div class="estimate-value">${compactMoney(payload.estimated_total_price_lkr)}</div>
            <div class="sub-estimate">${money(payload.estimated_total_price_lkr)} total · ${money(payload.estimated_price_per_perch_lkr)} per perch · ${number(payload.land_extent_perches)} perches</div>
          </div>
          <div class="metric"><div class="confidence-head"><div class="metric-label">Likely value range <span class="tooltip" data-tip="The range is based on the model's typical holdout error. It is a guide, not a formal confidence interval.">?</span></div><div class="confidence-badge">Guide only</div></div>
          <div class="band">
            <div class="band-item"><div class="band-label">Low</div><div class="band-value">${compactMoney(payload.confidence_band.low_total_price_lkr)}</div></div>
            <div class="band-item expected"><div class="band-label">Expected</div><div class="band-value">${compactMoney(payload.confidence_band.expected_total_price_lkr)}</div></div>
            <div class="band-item"><div class="band-label">High</div><div class="band-value">${compactMoney(payload.confidence_band.high_total_price_lkr)}</div></div>
          </div>
          <p class="note">${payload.confidence_note}</p></div>
          ${payload.warning ? `<div class="warning">${payload.warning}</div>` : `<div class="warning">This location has ${payload.training_listings_for_city} cleaned examples in the training data.</div>`}
          <div class="model-panel">
            <div class="metric-label">Estimate quality</div>
            <div class="model-grid">
              <div class="model-stat"><strong>${payload.model_name}</strong><span>Prediction model</span></div>
              <div class="model-stat"><strong>${number(payload.model_training_rows)}</strong><span>Cleaned listings</span></div>
              <div class="model-stat"><strong>${money(payload.model_mae_lkr)}</strong><span>Typical error</span></div>
              <div class="model-stat"><strong>${payload.training_listings_for_city}</strong><span>Local examples</span></div>
            </div>
          </div>
          <div class="disclaimer">This estimate is generated from listing data and should be used for screening and comparison only. It is not a professional valuation.</div>`;
        metrics.insertAdjacentHTML("beforeend", `
          <div class="result-actions">
            <button id="copy-result" class="secondary-button" type="button">Copy result</button>
            <button id="clear-result" class="secondary-button" type="button">Clear result</button>
          </div>
        `);
        document.getElementById("copy-result").addEventListener("click", async () => {
          await navigator.clipboard.writeText(latestEstimateText);
          document.getElementById("copy-result").textContent = "Copied";
          setTimeout(() => {
            const copyButton = document.getElementById("copy-result");
            if (copyButton) copyButton.textContent = "Copy result";
          }, 1400);
        });
        document.getElementById("clear-result").addEventListener("click", () => {
          metrics.className = "placeholder";
          metrics.textContent = "Expected value, price per perch, range, and data coverage will be shown here.";
        });
      } catch (requestError) {
        error.textContent = requestError.message;
        metrics.className = "";
        metrics.innerHTML = `<div class="error-state">${requestError.message}</div>`;
      } finally {
        button.disabled = false;
        button.textContent = "Estimate land value →";
      }
    });
  </script>
</body>
</html>
        """
    )

@app.post("/predict", response_model=LandPredictionResponse)
def predict_land_price(request: LandPredictionRequest):
    if request.land_extent_perches <= 0:
        raise HTTPException(status_code=400, detail="Land extent in perches must be greater than zero.")
    
    input_data = build_prediction_features(request)
    
    # Predict log price and convert back using expm1
    pred_log = model.predict(input_data)
    pred_price_per_perch = float(np.expm1(pred_log[0]))
    
    total_estimated_price = pred_price_per_perch * request.land_extent_perches
    confidence_band = build_confidence_band(
        pred_price_per_perch, request.land_extent_perches
    )
    training_listings_for_city, warning = build_location_warning(
        request.city or "Colombo"
    )
    confidence_note = (
        "The range shows a lower, expected, and higher value based on recent model error. "
        "A wider range means the property needs closer manual comparison."
    )

    return {
        "district": (request.district or "Colombo").title(),
        "city": (request.city or "Colombo").title(),
        "land_type": (request.land_type or "Unknown").title(),
        "land_extent_perches": request.land_extent_perches,
        "estimated_price_per_perch_lkr": round(pred_price_per_perch, 2),
        "estimated_total_price_lkr": round(total_estimated_price, 2),
        "confidence_band": confidence_band,
        "training_listings_for_city": training_listings_for_city,
        "model_name": model_name,
        "model_training_rows": training_rows,
        "model_mae_lkr": round(model_mae, 2),
        "confidence_note": confidence_note,
        "warning": warning,
    }
