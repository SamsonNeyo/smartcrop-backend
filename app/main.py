from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from .ml_model import PredictionRequest, get_top_crops, get_crop_prediction, explanations
from pydantic import BaseModel
from pathlib import Path
from dotenv import load_dotenv
from .openai_client import chat_with_openai, OpenAIError
import time
from .forecast_model import build_forecast, ensure_models
from .soil_mapping import detect_soil_zone, list_soil_zones, get_sub_county_profile, get_crop_catalog
from datetime import datetime
from functools import lru_cache
import copy

ENV_PATH = Path(__file__).resolve().parents[2] / ".env"
load_dotenv(ENV_PATH)

app = FastAPI(title="Luwero Crop Prediction API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/predict")
async def predict_crop(request: PredictionRequest):
    season_value = "First" if str(request.season).strip().lower() in {"1", "first", "first season"} else "Second"
    recommendations = _add_planning_to_recommendations(get_top_crops(request), season_value)
    top = recommendations[0] if recommendations else {"crop": None, "confidence_score": 0.0}
    season_context = _season_context(season_value)
    return {
        "recommended_crop": top["crop"],
        "confidence": top["confidence_score"],
        "recommendations": recommendations,
        "inputs": request.dict(),
        "season_advice": {
            **season_context,
            "focus": "Plan early planting and harvest logistics to reduce losses.",
        },
    }


@app.post("/predict/search")
async def predict_crop_search(request: PredictionRequest, crop: str):
    result = get_crop_prediction(request, crop)
    if not result:
        raise HTTPException(status_code=404, detail="Crop not found in recommendations.")
    return {
        "result": result,
        "query": crop,
        "inputs": request.dict()
    }

@app.get("/")
async def root():
    return {"message": "Luwero Crop Prediction API is running"}


class ChatRequest(BaseModel):
    message: str


class SoilDetectionRequest(BaseModel):
    latitude: float
    longitude: float


class SubCountyPredictionRequest(BaseModel):
    sub_county: str
    season: str | int | None = None


def _season_from_date() -> str:
    month = datetime.now().month
    if 3 <= month <= 6:
        return "First"
    if 8 <= month <= 12:
        return "Second"
    return "First"


def _season_defaults(season: str) -> dict:
    if season == "First":
        return {"temperature": 26.0, "rainfall": 180.0}
    return {"temperature": 25.0, "rainfall": 120.0}


@lru_cache(maxsize=128)
def _cached_sub_county_response(sub_county_key: str, season_value: str) -> dict | None:
    profile = get_sub_county_profile(sub_county_key)
    if not profile:
        return None
    defaults = profile.get("first") if season_value == "First" else profile.get("second")
    if not defaults:
        defaults = _season_defaults(season_value)
    soil_type = profile["soil_type"]

    preferred_crops = profile.get("preferred_crops") or ["Maize", "Beans", "Cassava", "Groundnuts"]

    base_score = 0.93
    recommendations = []
    for crop in preferred_crops[:8]:
        score = max(base_score, 0.45)
        recommendations.append(
            {
                "crop": crop,
                "confidence": round(score * 100, 1),
                "confidence_score": round(score, 4),
                "source": "sub_county_profile",
                "explanation": explanations.get(crop, "Commonly grown in this sub-county."),
            }
        )
        base_score -= 0.07

    recommendations = _add_planning_to_recommendations(recommendations, season_value)
    top = recommendations[0] if recommendations else {"crop": None, "confidence_score": 0.0}
    season_context = _season_context(season_value)
    return {
        "recommended_crop": top["crop"],
        "confidence": top["confidence_score"],
        "recommendations": recommendations,
        "season_advice": {
            **season_context,
            "harvest_readiness": "Arrange labor, storage, and buyers at least 2-3 weeks before expected harvest.",
            "focus": "Use sub-county crop priorities and season timing to schedule planting and harvest decisions.",
        },
        "inputs": {
            "sub_county": profile["sub_county"],
            "season": season_value,
            "soil_type": soil_type,
            "temperature": defaults["temperature"],
            "rainfall": defaults["rainfall"],
        },
    }


@app.get("/forecast")
async def forecast(steps: int = 6):
    if steps < 1 or steps > 24:
        raise HTTPException(status_code=400, detail="steps must be between 1 and 24")
    return build_forecast(steps=steps)


@app.post("/detect-soil")
async def detect_soil(request: SoilDetectionRequest):
    return detect_soil_zone(request.latitude, request.longitude)


@app.get("/soil-zones")
async def soil_zones():
    return {"soil_zones": list_soil_zones()}


@app.get("/crop-catalog")
async def crop_catalog():
    return get_crop_catalog()


@app.post("/predict/sub-county")
async def predict_by_sub_county(request: SubCountyPredictionRequest):
    season = request.season if request.season is not None else _season_from_date()
    season_value = "First" if str(season).strip().lower() in {"1", "first", "first season"} else "Second"
    cache_key = request.sub_county.strip().lower()
    cached = _cached_sub_county_response(cache_key, season_value)
    if not cached:
        raise HTTPException(status_code=404, detail="Sub-county not found in mapping table.")
    return copy.deepcopy(cached)


@app.post("/chat")
async def chat(request: ChatRequest, http_request: Request):
    try:
        if not request.message.strip():
            raise HTTPException(status_code=400, detail="Message is required.")
        client_ip = http_request.client.host if http_request.client else "unknown"
        now = time.time()
        recent = [t for t in _RATE_LIMIT.get(client_ip, []) if now - t < _RATE_LIMIT_WINDOW]
        if len(recent) >= _RATE_LIMIT_MAX:
            raise HTTPException(status_code=429, detail="Too many requests. Please wait a few seconds.")
        recent.append(now)
        _RATE_LIMIT[client_ip] = recent
        cache_key = f"{client_ip}:{request.message.strip().lower()}"
        answer = await chat_with_openai(request.message, cache_key=cache_key)
        return {"answer": answer}
    except OpenAIError as e:
        raise HTTPException(status_code=e.status_code, detail=e.detail)

# For testing purposes, you can run this app with Uvicorn:
# uvicorn app.main:app --reload --host
# cmd /c "python -m uvicorn app.main:app --reload --host 0.0.0.0 --port 8000"
# Make sure to replace <IP_ADDRESS> with your actual local IP address if testing from a mobile device.
_RATE_LIMIT: dict[str, list[float]] = {}
_RATE_LIMIT_WINDOW = 10  # seconds
_RATE_LIMIT_MAX = 1
ensure_models()

CROP_CALENDAR = {
    "Maize": {"duration_days": "95-130", "harvest_first": "June - July", "harvest_second": "November - December"},
    "Beans": {"duration_days": "70-100", "harvest_first": "May - June", "harvest_second": "October - November"},
    "Cassava": {"duration_days": "300-540", "harvest_first": "From January onward", "harvest_second": "From March onward"},
    "Groundnuts": {"duration_days": "90-120", "harvest_first": "June - July", "harvest_second": "November - December"},
    "Banana": {"duration_days": "270-420", "harvest_first": "Harvest all year, peaks from July", "harvest_second": "Harvest all year, peaks from December"},
    "Rice": {"duration_days": "110-150", "harvest_first": "July - August", "harvest_second": "December - January"},
    "Sweet Potatoes": {"duration_days": "90-130", "harvest_first": "June - July", "harvest_second": "November - December"},
    "Sorghum": {"duration_days": "105-140", "harvest_first": "July - August", "harvest_second": "December - January"},
    "Millet": {"duration_days": "95-120", "harvest_first": "June - July", "harvest_second": "November - December"},
    "Soybeans": {"duration_days": "95-130", "harvest_first": "June - July", "harvest_second": "November - December"},
    "Tomatoes": {"duration_days": "85-120", "harvest_first": "May - July", "harvest_second": "October - December"},
    "Onions": {"duration_days": "120-170", "harvest_first": "July - September", "harvest_second": "December - February"},
    "Cabbage": {"duration_days": "90-130", "harvest_first": "May - July", "harvest_second": "October - December"},
    "Coffee": {"duration_days": "900-1200", "harvest_first": "Main harvest: October - January", "harvest_second": "Main harvest: October - January"},
    "Tea": {"duration_days": "900-1200", "harvest_first": "Plucking starts 18+ months, then year-round", "harvest_second": "Plucking starts 18+ months, then year-round"},
    "Cotton": {"duration_days": "150-210", "harvest_first": "August - October", "harvest_second": "January - March"},
    "Sugarcane": {"duration_days": "360-540", "harvest_first": "From year 1 onward", "harvest_second": "From year 1 onward"},
    "Tobacco": {"duration_days": "120-180", "harvest_first": "July - September", "harvest_second": "December - February"},
    "Sunflower": {"duration_days": "90-130", "harvest_first": "June - July", "harvest_second": "November - December"},
    "Pineapple": {"duration_days": "360-540", "harvest_first": "From year 1 onward", "harvest_second": "From year 1 onward"},
    "Cocoa": {"duration_days": "1000-1500", "harvest_first": "Main harvest after establishment (3+ years)", "harvest_second": "Main harvest after establishment (3+ years)"},
}


def _season_context(season: str) -> dict:
    if season == "First":
        return {
            "season": "First",
            "land_preparation_window": "January - February",
            "planting_window": "March - April",
            "weather_expectation": "Main rains are expected from March to May. Prepare drainage and weed control early.",
        }
    return {
        "season": "Second",
        "land_preparation_window": "July - August",
        "planting_window": "Late August - September",
        "weather_expectation": "Second rains are expected from August to November. Prioritize timely planting and disease scouting.",
    }


def _build_crop_plan(crop: str, season: str) -> dict:
    defaults = {
        "duration_days": "90-140",
        "harvest_first": "June - August",
        "harvest_second": "November - January",
    }
    details = CROP_CALENDAR.get(crop, defaults)
    harvest_window = details["harvest_first"] if season == "First" else details["harvest_second"]
    return {
        "duration_days": details["duration_days"],
        "harvest_window": harvest_window,
        "planning_actions": [
            "Prepare land and inputs 2-4 weeks before planting window.",
            "Plant at first effective rains for stronger establishment.",
            "Scout weekly for pests, diseases, and nutrient stress.",
            "Plan post-harvest handling and market access before harvest starts.",
        ],
    }


def _add_planning_to_recommendations(recommendations: list[dict], season: str) -> list[dict]:
    for item in recommendations:
        crop = item.get("crop")
        if not crop:
            continue
        item["planning"] = _build_crop_plan(str(crop), season)
    return recommendations
