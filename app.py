from pathlib import Path
import re
import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

# Initialize FastAPI app
app = FastAPI(
    title="TerraLanka ML - Land Price Prediction API",
    description="API for predicting land prices per perch in Colombo, Sri Lanka.",
    version="1.0.0"
)

# Load trained model artifact
PROJECT_DIR = Path(__file__).resolve().parent
MODEL_PATH = PROJECT_DIR / "models" / "colombo_land_model.pkl"
CLEAN_DATA_PATH = PROJECT_DIR / "data" / "clean_lands.csv"

if not MODEL_PATH.exists():
    raise RuntimeError(f"Model artifact not found at {MODEL_PATH}. Please train the model first using train_colombo.py.")

artifact = joblib.load(MODEL_PATH)
model = artifact["model"]
feature_columns = artifact["feature_columns"]
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
            "No cleaned training listings were found for this city. The estimate relies on nearby/general patterns.",
        )
    if listing_count < 5:
        return (
            listing_count,
            f"Only {listing_count} cleaned training listings were found for this city, so treat the estimate as approximate.",
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
      --ink: #17332b;
      --muted: #6a7e75;
      --cream: #f6f2e9;
      --paper: #fffdf8;
      --green: #1f6b4f;
      --green-dark: #124c39;
      --gold: #d5a84a;
      --line: #dce5dc;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      min-height: 100vh;
      color: var(--ink);
      background:
        radial-gradient(circle at 10% 0%, #dcecdf 0, transparent 34%),
        var(--cream);
      font-family: Inter, ui-sans-serif, system-ui, -apple-system, sans-serif;
    }
    .shell { width: min(1080px, 92vw); margin: 0 auto; padding: 34px 0 58px; }
    .topbar { display: flex; justify-content: space-between; align-items: center; margin-bottom: 58px; }
    .brand { display: flex; align-items: center; gap: 11px; font-weight: 800; letter-spacing: -0.03em; }
    .brand-mark {
      display: grid; place-items: center; width: 38px; height: 38px;
      color: white; background: var(--green); border-radius: 12px; font-size: 20px;
    }
    .tag { color: var(--green); font-size: 13px; font-weight: 700; }
    .hero { max-width: 710px; margin-bottom: 31px; }
    .eyebrow { color: var(--gold); font-size: 12px; font-weight: 800; letter-spacing: .16em; text-transform: uppercase; }
    h1 { margin: 11px 0 14px; font-size: clamp(38px, 6vw, 70px); line-height: .98; letter-spacing: -0.065em; }
    .hero p { max-width: 580px; margin: 0; color: var(--muted); font-size: 17px; line-height: 1.6; }
    .grid { display: grid; grid-template-columns: minmax(0, 1.02fr) minmax(0, .98fr); gap: 20px; }
    .card { padding: 28px; background: rgba(255,253,248,.9); border: 1px solid rgba(220,229,220,.9); border-radius: 24px; box-shadow: 0 18px 50px rgba(23,51,43,.08); }
    .card h2 { margin: 0 0 8px; font-size: 22px; letter-spacing: -.03em; }
    .card-copy { margin: 0 0 26px; color: var(--muted); line-height: 1.5; }
    label { display: block; margin-bottom: 9px; font-size: 13px; font-weight: 800; }
    .input-wrap { position: relative; }
    input, select, textarea {
      width: 100%; padding: 17px 68px 17px 16px; color: var(--ink);
      background: var(--paper); border: 1px solid var(--line); border-radius: 13px;
      font: inherit; font-size: 18px; outline: none; transition: .2s;
    }
    textarea { min-height: 92px; resize: vertical; padding-right: 16px; }
    input:focus, select:focus, textarea:focus { border-color: var(--green); box-shadow: 0 0 0 4px rgba(31,107,79,.12); }
    .unit { position: absolute; top: 50%; right: 16px; color: var(--muted); font-size: 13px; transform: translateY(-50%); }
    .split { display: grid; grid-template-columns: 1fr 1fr; gap: 14px; }
    .checks { display: grid; gap: 10px; margin-top: 16px; }
    .check {
      display: flex; align-items: center; gap: 10px; padding: 12px 13px;
      background: var(--paper); border: 1px solid var(--line); border-radius: 13px;
      color: var(--ink); font-size: 14px; font-weight: 700;
    }
    .check input { width: 18px; height: 18px; padding: 0; accent-color: var(--green); }
    button {
      width: 100%; margin-top: 22px; padding: 16px; color: white; background: var(--green);
      border: 0; border-radius: 13px; font: inherit; font-weight: 800; cursor: pointer;
      transition: background .2s, transform .2s;
    }
    button:hover { background: var(--green-dark); transform: translateY(-1px); }
    button:disabled { cursor: wait; opacity: .65; transform: none; }
    .hint { margin: 13px 0 0; color: var(--muted); font-size: 12px; }
    .results { display: grid; align-content: center; min-height: 288px; background: var(--green); color: white; }
    .results h2 { color: #dcecdf; }
    .results-copy { color: #b9d1c3; }
    .placeholder { color: #b9d1c3; line-height: 1.6; }
    .metric { padding: 17px 0; border-top: 1px solid rgba(255,255,255,.18); }
    .metric-label { color: #b9d1c3; font-size: 12px; font-weight: 700; text-transform: uppercase; letter-spacing: .08em; }
    .metric-value { margin-top: 5px; font-size: clamp(25px, 4vw, 36px); font-weight: 800; letter-spacing: -.04em; }
    .band { display: grid; grid-template-columns: repeat(3, 1fr); gap: 8px; margin-top: 14px; }
    .band-item { padding: 12px; background: rgba(255,255,255,.1); border-radius: 12px; }
    .band-label { color: #b9d1c3; font-size: 11px; font-weight: 800; text-transform: uppercase; letter-spacing: .08em; }
    .band-value { margin-top: 4px; color: white; font-size: 15px; font-weight: 800; }
    .warning { margin-top: 14px; padding: 12px; color: #fff3c4; background: rgba(213,168,74,.16); border-radius: 12px; font-size: 13px; line-height: 1.4; }
    .error { margin-top: 14px; color: #b53f35; font-size: 13px; font-weight: 700; }
    .results .error { color: #ffd4ce; }
    @media (max-width: 720px) {
      .topbar { margin-bottom: 42px; }
      .grid { grid-template-columns: 1fr; }
      .split { grid-template-columns: 1fr; }
      .band { grid-template-columns: 1fr; }
      .card { padding: 23px; }
    }
  </style>
</head>
<body>
  <main class="shell">
    <nav class="topbar">
      <div class="brand"><span class="brand-mark">⌂</span> TerraLanka ML</div>
      <span class="tag">COLOMBO · SRI LANKA</span>
    </nav>
    <section class="hero">
      <div class="eyebrow">Data-informed property decisions</div>
      <h1>Estimate your land's value.</h1>
      <p>Enter the land extent below to get an instant estimate based on our Colombo land-price model.</p>
    </section>
    <section class="grid">
      <form class="card" id="prediction-form">
        <h2>Property details</h2>
        <p class="card-copy">The model currently supports Colombo land listings.</p>
        <label for="extent">Land extent</label>
        <div class="input-wrap">
          <input id="extent" name="extent" type="number" min="0.1" step="0.1" placeholder="e.g. 10" required>
          <span class="unit">perches</span>
        </div>
        <div class="split">
          <div>
            <label for="district" style="margin-top:16px;">District</label>
            <div class="input-wrap">
              <input id="district" name="district" type="text" value="Colombo">
            </div>
          </div>
          <div>
            <label for="city" style="margin-top:16px;">City or suburb</label>
            <div class="input-wrap">
              <input id="city" name="city" type="text" placeholder="e.g. Maharagama">
            </div>
          </div>
        </div>
        <label for="land-type" style="margin-top:16px;">Land type</label>
        <div class="input-wrap">
          <select id="land-type" name="land-type">
            <option value="Unknown">Unknown</option>
            <option value="Residential">Residential</option>
            <option value="Commercial">Commercial</option>
            <option value="Bare Land">Bare land</option>
            <option value="Agricultural">Agricultural</option>
            <option value="Industrial">Industrial</option>
          </select>
        </div>
        <div class="checks">
          <label class="check"><input id="main-road" type="checkbox">Main road access</label>
          <label class="check"><input id="residential" type="checkbox">Residential use</label>
          <label class="check"><input id="commercial" type="checkbox">Commercial use</label>
        </div>
        <label for="title" style="margin-top:16px;">Listing keywords</label>
        <div class="input-wrap">
          <input id="title" name="title" type="text" placeholder="e.g. residential bare land main road">
        </div>
        <label for="description" style="margin-top:16px;">Description</label>
        <div class="input-wrap">
          <textarea id="description" name="description" placeholder="Optional notes about road width, approvals, nearby landmarks, or other listing details"></textarea>
        </div>
        <button id="submit" type="submit">Calculate estimate →</button>
        <p class="hint">Estimates are indicative and should not replace an independent valuation.</p>
        <div class="error" id="form-error" role="alert"></div>
      </form>
      <section class="card results" aria-live="polite">
        <h2>Your estimate</h2>
        <p class="card-copy results-copy">A prediction will appear here after you enter the land size.</p>
        <div id="metrics" class="placeholder">Your estimated price per perch and total property value will be shown here.</div>
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
    const error = document.getElementById("form-error");
    const metrics = document.getElementById("metrics");
    const money = value => new Intl.NumberFormat("en-LK", {
      style: "currency", currency: "LKR", maximumFractionDigits: 0
    }).format(value);

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
      button.textContent = "Calculating…";
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
        metrics.className = "";
        metrics.innerHTML = `
          <div class="metric"><div class="metric-label">Estimated price per perch</div>
          <div class="metric-value">${money(payload.estimated_price_per_perch_lkr)}</div></div>
          <div class="metric"><div class="metric-label">Estimated total value</div>
          <div class="metric-value">${money(payload.estimated_total_price_lkr)}</div></div>
          <div class="metric"><div class="metric-label">Confidence band</div>
          <div class="band">
            <div class="band-item"><div class="band-label">Low</div><div class="band-value">${money(payload.confidence_band.low_total_price_lkr)}</div></div>
            <div class="band-item"><div class="band-label">Expected</div><div class="band-value">${money(payload.confidence_band.expected_total_price_lkr)}</div></div>
            <div class="band-item"><div class="band-label">High</div><div class="band-value">${money(payload.confidence_band.high_total_price_lkr)}</div></div>
          </div></div>
          ${payload.warning ? `<div class="warning">${payload.warning}</div>` : `<div class="warning">Based on ${payload.training_listings_for_city} cleaned listings for this city.</div>`}`;
      } catch (requestError) {
        error.textContent = requestError.message;
      } finally {
        button.disabled = false;
        button.textContent = "Calculate estimate →";
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

    return {
        "district": (request.district or "Colombo").title(),
        "city": (request.city or "Colombo").title(),
        "land_type": (request.land_type or "Unknown").title(),
        "land_extent_perches": request.land_extent_perches,
        "estimated_price_per_perch_lkr": round(pred_price_per_perch, 2),
        "estimated_total_price_lkr": round(total_estimated_price, 2),
        "confidence_band": confidence_band,
        "training_listings_for_city": training_listings_for_city,
        "warning": warning,
    }
