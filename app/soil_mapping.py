from __future__ import annotations

from math import asin, cos, radians, sin, sqrt

LUWERO_BOUNDS = {
    "min_lat": 0.45,
    "max_lat": 1.05,
    "min_lon": 32.15,
    "max_lon": 32.75,
}

SUBCOUNTY_SOIL_MAP = [
    {"sub_county": "Bamunanika", "soil_type": "Clay Loam", "first": {"temperature": 24.5, "rainfall": 230.0}, "second": {"temperature": 25.0, "rainfall": 170.0}, "preferred_crops": ["Maize", "Beans", "Groundnuts", "Banana", "Rice", "Cassava", "Coffee", "Soybeans"]},
    {"sub_county": "Bombo Town Council", "soil_type": "Loam", "first": {"temperature": 27.0, "rainfall": 165.0}, "second": {"temperature": 26.5, "rainfall": 120.0}, "preferred_crops": ["Maize", "Beans", "Cassava", "Groundnuts", "Soybeans", "Tomatoes", "Onions", "Sunflower"]},
    {"sub_county": "Busiika Town Council", "soil_type": "Loam", "first": {"temperature": 26.6, "rainfall": 178.0}, "second": {"temperature": 26.1, "rainfall": 128.0}, "preferred_crops": ["Maize", "Beans", "Groundnuts", "Cassava", "Soybeans", "Tomatoes", "Onions", "Sunflower"]},
    {"sub_county": "Butuntumula", "soil_type": "Ferrallitic", "first": {"temperature": 25.0, "rainfall": 210.0}, "second": {"temperature": 25.5, "rainfall": 150.0}, "preferred_crops": ["Maize", "Banana", "Groundnuts", "Beans", "Rice", "Coffee", "Tea", "Cabbage"]},
    {"sub_county": "Kalangala", "soil_type": "Sandy Loam", "first": {"temperature": 27.2, "rainfall": 155.0}, "second": {"temperature": 26.8, "rainfall": 112.0}, "preferred_crops": ["Cassava", "Groundnuts", "Maize", "Sweet Potatoes", "Sorghum", "Sunflower", "Pineapple", "Cotton"]},
    {"sub_county": "Katikamu", "soil_type": "Ferrallitic", "first": {"temperature": 24.8, "rainfall": 220.0}, "second": {"temperature": 25.2, "rainfall": 160.0}, "preferred_crops": ["Maize", "Beans", "Groundnuts", "Banana", "Soybeans", "Coffee", "Rice", "Tomatoes"]},
    {"sub_county": "Katikamu Town Council", "soil_type": "Ferrallitic", "first": {"temperature": 25.3, "rainfall": 198.0}, "second": {"temperature": 25.7, "rainfall": 142.0}, "preferred_crops": ["Maize", "Beans", "Groundnuts", "Banana", "Soybeans", "Coffee", "Tomatoes", "Onions"]},
    {"sub_county": "Kikyusa", "soil_type": "Sandy Loam", "first": {"temperature": 27.8, "rainfall": 145.0}, "second": {"temperature": 27.2, "rainfall": 105.0}, "preferred_crops": ["Cassava", "Sweet Potatoes", "Pineapple", "Groundnuts", "Maize", "Sorghum", "Millet", "Sunflower"]},
    {"sub_county": "Luwero Sub-county", "soil_type": "Loam", "first": {"temperature": 26.2, "rainfall": 185.0}, "second": {"temperature": 26.0, "rainfall": 130.0}, "preferred_crops": ["Maize", "Beans", "Groundnuts", "Banana", "Cassava", "Soybeans", "Coffee", "Tomatoes"]},
    {"sub_county": "Luwero Town Council", "soil_type": "Loam", "first": {"temperature": 26.8, "rainfall": 175.0}, "second": {"temperature": 26.3, "rainfall": 125.0}, "preferred_crops": ["Beans", "Tomatoes", "Onions", "Maize", "Cabbage", "Soybeans", "Sunflower", "Rice"]},
    {"sub_county": "Makulubita", "soil_type": "Sandy Loam", "first": {"temperature": 28.0, "rainfall": 140.0}, "second": {"temperature": 27.4, "rainfall": 98.0}, "preferred_crops": ["Cassava", "Sorghum", "Millet", "Groundnuts", "Sunflower", "Cotton", "Tobacco", "Pineapple"]},
    {"sub_county": "Nyimbwa", "soil_type": "Clay Loam", "first": {"temperature": 24.6, "rainfall": 235.0}, "second": {"temperature": 25.1, "rainfall": 175.0}, "preferred_crops": ["Maize", "Rice", "Beans", "Groundnuts", "Banana", "Coffee", "Sugarcane", "Tea"]},
    {"sub_county": "Wobulenzi Town Council", "soil_type": "Ferrallitic", "first": {"temperature": 25.4, "rainfall": 205.0}, "second": {"temperature": 25.8, "rainfall": 145.0}, "preferred_crops": ["Maize", "Beans", "Groundnuts", "Banana", "Soybeans", "Coffee", "Tomatoes", "Onions"]},
    {"sub_county": "Zirobwe", "soil_type": "Clay Loam", "first": {"temperature": 24.3, "rainfall": 240.0}, "second": {"temperature": 24.9, "rainfall": 180.0}, "preferred_crops": ["Rice", "Maize", "Beans", "Groundnuts", "Banana", "Coffee", "Sugarcane", "Tea"]},
]

FOOD_CROPS = [
    "Maize",
    "Beans",
    "Cassava",
    "Groundnuts",
    "Banana",
    "Rice",
    "Sweet Potatoes",
    "Sorghum",
    "Millet",
    "Soybeans",
    "Tomatoes",
    "Onions",
    "Cabbage",
]

CASH_CROPS = [
    "Coffee",
    "Tea",
    "Cotton",
    "Sugarcane",
    "Tobacco",
    "Sunflower",
    "Pineapple",
    "Cocoa",
]

SOIL_ZONES = [
    {"sub_county": "Bamunanika", "soil_type": "Clay Loam", "lat": 0.78, "lon": 32.55, "radius_km": 18},
    {"sub_county": "Katikamu", "soil_type": "Ferrallitic", "lat": 0.63, "lon": 32.48, "radius_km": 20},
    {"sub_county": "Kikyusa", "soil_type": "Sandy Loam", "lat": 0.92, "lon": 32.36, "radius_km": 20},
    {"sub_county": "Nyimbwa", "soil_type": "Clay Loam", "lat": 0.66, "lon": 32.66, "radius_km": 16},
    {"sub_county": "Makulubita", "soil_type": "Sandy Loam", "lat": 0.9, "lon": 32.49, "radius_km": 16},
    {"sub_county": "Butuntumula", "soil_type": "Ferrallitic", "lat": 0.71, "lon": 32.41, "radius_km": 15},
]


def list_soil_zones() -> list[dict]:
    return [{"sub_county": item["sub_county"], "soil_type": item["soil_type"]} for item in SUBCOUNTY_SOIL_MAP]


def get_soil_by_sub_county(sub_county: str) -> str | None:
    query = sub_county.strip().lower()
    for item in SUBCOUNTY_SOIL_MAP:
        if item["sub_county"].strip().lower() == query:
            return item["soil_type"]
    return None


def get_sub_county_profile(sub_county: str) -> dict | None:
    query = sub_county.strip().lower()
    for item in SUBCOUNTY_SOIL_MAP:
        if item["sub_county"].strip().lower() == query:
            return item
    return None


def get_preferred_crops(sub_county: str) -> list[str]:
    profile = get_sub_county_profile(sub_county)
    if not profile:
        return []
    return profile.get("preferred_crops", [])


def get_crop_catalog() -> dict:
    return {
        "food_crops": FOOD_CROPS,
        "cash_crops": CASH_CROPS,
        "all_crops": list(dict.fromkeys(FOOD_CROPS + CASH_CROPS)),
    }


def _haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    earth_radius_km = 6371.0
    d_lat = radians(lat2 - lat1)
    d_lon = radians(lon2 - lon1)
    a = sin(d_lat / 2) ** 2 + cos(radians(lat1)) * cos(radians(lat2)) * sin(d_lon / 2) ** 2
    c = 2 * asin(sqrt(a))
    return earth_radius_km * c


def detect_soil_zone(latitude: float, longitude: float) -> dict:
    in_luwero = (
        LUWERO_BOUNDS["min_lat"] <= latitude <= LUWERO_BOUNDS["max_lat"]
        and LUWERO_BOUNDS["min_lon"] <= longitude <= LUWERO_BOUNDS["max_lon"]
    )
    if not in_luwero:
        return {
            "in_luwero": False,
            "manual_selection_required": True,
            "suggested_soil_type": None,
            "sub_county": None,
            "message": "Location is outside Luwero boundary. Please select soil manually.",
        }

    best_zone = None
    best_distance = float("inf")
    for zone in SOIL_ZONES:
        dist = _haversine_km(latitude, longitude, zone["lat"], zone["lon"])
        if dist < best_distance:
            best_distance = dist
            best_zone = zone

    if best_zone and best_distance <= best_zone["radius_km"]:
        return {
            "in_luwero": True,
            "manual_selection_required": False,
            "suggested_soil_type": best_zone["soil_type"],
            "sub_county": best_zone["sub_county"],
            "message": f"Mapped to {best_zone['sub_county']} soil zone.",
        }

    return {
        "in_luwero": True,
        "manual_selection_required": True,
        "suggested_soil_type": None,
        "sub_county": None,
        "message": "Within Luwero but no clear soil zone match. Please select soil manually.",
    }
