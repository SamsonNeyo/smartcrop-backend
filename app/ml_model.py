from pathlib import Path
from typing import Dict, List

import joblib
import numpy as np
import pandas as pd
from pydantic import BaseModel

MODEL_PATH = Path(__file__).resolve().parent.parent / "model_pipeline.joblib"
artifact = joblib.load(MODEL_PATH)
if isinstance(artifact, dict) and "model" in artifact:
    model = artifact["model"]
    encoders = artifact.get("encoders", {})
else:
    model = artifact
    encoders = {}


class PredictionRequest(BaseModel):
    season: str | int
    soil_type: str
    temperature: float
    rainfall: float


explanations = {
    "Maize": "Thrives in warm temperatures (22-30 C) and moderate-high rainfall.",
    "Beans": "Fast-growing legume. Performs well in loamy soils.",
    "Cassava": "Drought-tolerant. Strong option for lighter soils.",
    "Groundnuts": "Good fit for moderate rainfall and well-drained soils.",
    "Banana": "Perennial crop that benefits from consistent moisture.",
    "Rice": "Can perform well under wetter conditions.",
    "Sweet Potatoes": "Grows well in sandy soils with moderate rain.",
    "Coffee": "Needs moderate temperatures and stable rainfall.",
    "Pineapple": "Performs in warmer conditions and light soils.",
    "Bananas": "Perennial crop that benefits from consistent moisture.",
    "Sorghum": "Performs well in drier conditions and is resilient to heat.",
    "Millet": "Hardy cereal suited for low to moderate rainfall zones.",
    "Soybeans": "Nitrogen-fixing crop that does well in fertile, well-drained soils.",
    "Tomatoes": "High-value horticulture crop for well-managed loamy soils.",
    "Onions": "Prefers well-drained soils and moderate moisture.",
    "Cabbage": "Cooler-season vegetable requiring steady water supply.",
    "Sunflower": "Drought-tolerant oil crop suitable for lighter soils.",
    "Tea": "High-value perennial cash crop for suitable cool-wet zones.",
    "Cotton": "Cash crop that performs best in warm conditions with moderate rainfall.",
    "Sugarcane": "Long-duration cash crop requiring reliable moisture.",
    "Tobacco": "Cash crop suited to well-drained soils and managed fertility.",
    "Cocoa": "Perennial cash crop for humid conditions with careful management.",
}


def _normalize_season(season: str | int) -> str:
    if isinstance(season, int):
        return "First" if season == 1 else "Second"
    raw = str(season).strip().lower()
    if raw in {"1", "first", "first season"}:
        return "First"
    if raw in {"2", "second", "second season"}:
        return "Second"
    return "First"


def _normalize_soil(soil_type: str) -> str:
    raw = soil_type.strip().lower()
    if raw in {"clay", "clay loam", "ferrallitic"}:
        return "Clay"
    if raw in {"sandy", "sandy loam"}:
        return "Sandy"
    return "Loam"


def _build_input_frame(request: PredictionRequest) -> pd.DataFrame:
    season_value = _normalize_season(request.season)
    soil_value = _normalize_soil(request.soil_type)
    if encoders:
        season_encoder = encoders.get("season")
        soil_encoder = encoders.get("soil_type")
        if season_encoder is None or soil_encoder is None:
            encoders.clear()
            return pd.DataFrame(
                [{"season": season_value, "soil_type": soil_value, "temperature": request.temperature, "rainfall": request.rainfall}]
            )

        def _safe_encode(value: str, encoder, fallback: str) -> int:
            classes = {str(c).lower(): c for c in getattr(encoder, "classes_", [])}
            if value.lower() in classes:
                return int(encoder.transform([classes[value.lower()]])[0])
            if fallback.lower() in classes:
                return int(encoder.transform([classes[fallback.lower()]])[0])
            return int(encoder.transform([encoder.classes_[0]])[0])

        season_encoded = _safe_encode(season_value, season_encoder, "First")
        soil_encoded = _safe_encode(soil_value, soil_encoder, "Loam")
        return pd.DataFrame(
            [{"season_encoded": season_encoded, "soil_type_encoded": soil_encoded, "temperature": request.temperature, "rainfall": request.rainfall}]
        )

    return pd.DataFrame(
        [{"season": season_value, "soil_type": soil_value, "temperature": request.temperature, "rainfall": request.rainfall}]
    )


def _build_ranked_results(
    input_data: pd.DataFrame,
    top_k: int,
    preferred_crops: list[str] | None = None,
    include_preferred_missing: bool = False,
) -> List[Dict]:
    preferred = {c.strip().lower() for c in (preferred_crops or []) if c and str(c).strip()}
    proba = model.predict_proba(input_data)[0]
    if encoders and "crop" in encoders:
        classes = encoders["crop"].inverse_transform(np.arange(len(proba)))
    else:
        classes = model.classes_

    ranked = []
    for idx, cls in enumerate(classes):
        crop = str(cls)
        base = float(proba[idx])
        bonus = 0.08 if crop.lower() in preferred else 0.0
        adjusted = base + bonus
        ranked.append((idx, crop, base, adjusted))

    ranked.sort(key=lambda x: x[3], reverse=True)
    top = ranked[:top_k]
    results: List[Dict] = []
    present_crops = set()
    for _, crop, base_score, adjusted_score in top:
        conf = min(adjusted_score, 0.99)
        present_crops.add(crop.lower())
        results.append(
            {
                "crop": crop,
                "confidence": round(conf * 100, 1),
                "confidence_score": round(conf, 4),
                "model_score": round(base_score, 4),
                "explanation": explanations.get(crop, "Suitable based on current conditions."),
            }
        )

    if include_preferred_missing and len(results) < top_k:
        missing_preferred = [c for c in (preferred_crops or []) if c.strip().lower() not in present_crops]
        fallback_conf = 44.0
        for crop in missing_preferred:
            if len(results) >= top_k:
                break
            results.append(
                {
                    "crop": crop,
                    "confidence": round(max(fallback_conf, 28.0), 1),
                    "confidence_score": round(max(fallback_conf, 28.0) / 100, 4),
                    "model_score": None,
                    "local_priority": True,
                    "explanation": explanations.get(crop, "Commonly grown in this sub-county."),
                }
            )
            fallback_conf -= 4.0
    return results


def get_top_crops(
    request: PredictionRequest,
    top_k: int = 3,
    preferred_crops: list[str] | None = None,
    include_preferred_missing: bool = False,
) -> List[Dict]:
    input_data = _build_input_frame(request)
    return _build_ranked_results(
        input_data,
        top_k=top_k,
        preferred_crops=preferred_crops,
        include_preferred_missing=include_preferred_missing,
    )


def get_crop_prediction(request: PredictionRequest, crop_query: str) -> Dict | None:
    input_data = _build_input_frame(request)
    proba = model.predict_proba(input_data)[0]
    if encoders and "crop" in encoders:
        classes = encoders["crop"].inverse_transform(np.arange(len(proba)))
    else:
        classes = model.classes_
    query = crop_query.strip().lower()
    if not query:
        return None
    exact_idx = None
    for i, c in enumerate(classes):
        if str(c).lower() == query:
            exact_idx = i
            break
    match_indices = []
    if exact_idx is not None:
        match_indices = [exact_idx]
    else:
        for i, c in enumerate(classes):
            if query in str(c).lower():
                match_indices.append(i)
    if not match_indices:
        return None
    best_idx = max(match_indices, key=lambda i: proba[i])
    crop = str(classes[best_idx])
    confidence_score = float(proba[best_idx])
    return {
        "crop": crop,
        "confidence": round(confidence_score * 100, 1),
        "confidence_score": round(confidence_score, 4),
        "explanation": explanations.get(crop, "Suitable based on current conditions."),
    }
