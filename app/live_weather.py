from __future__ import annotations

from pathlib import Path

import httpx
import pandas as pd


class LiveWeatherError(Exception):
    pass


WEATHER_DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "Luwero_Weather_2015_2025.csv"


def _to_celsius(value: float) -> float:
    # The provided CSV appears to use Fahrenheit temperatures.
    if value > 60:
        return (value - 32.0) * 5.0 / 9.0
    return value


def _to_millimeters(value: float) -> float:
    # The provided CSV appears to use inches for precipitation.
    if 0.0 <= value <= 10.0:
        return value * 25.4
    return value


def _fetch_weather_from_local_csv() -> dict | None:
    if not WEATHER_DATA_PATH.exists():
        return None
    try:
        df = pd.read_csv(WEATHER_DATA_PATH)
        if "datetime" not in df.columns or "temp" not in df.columns or "precip" not in df.columns:
            return None
        df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce")
        df = df.dropna(subset=["datetime", "temp", "precip"]).sort_values("datetime")
        if df.empty:
            return None
        latest = df.iloc[-1]
        temp_c = _to_celsius(float(latest["temp"]))
        rain_mm = _to_millimeters(float(latest["precip"]))
        return {
            "temperature": round(temp_c, 1),
            "rainfall": round(max(rain_mm, 0.0), 1),
            "source": "luwero_weather_2015_2025_csv",
            "as_of": latest["datetime"].strftime("%Y-%m-%d"),
        }
    except Exception:
        return None


async def fetch_live_weather(latitude: float, longitude: float) -> dict:
    local_weather = _fetch_weather_from_local_csv()
    if local_weather is not None:
        return local_weather

    params = {
        "latitude": latitude,
        "longitude": longitude,
        "current": "temperature_2m,precipitation",
        "daily": "precipitation_sum",
        "forecast_days": 1,
        "timezone": "auto",
    }
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.get("https://api.open-meteo.com/v1/forecast", params=params)
            response.raise_for_status()
        payload = response.json()
    except Exception as exc:
        raise LiveWeatherError(f"Failed to fetch live weather data: {exc}") from exc

    current = payload.get("current", {})
    daily = payload.get("daily", {})
    temperature = current.get("temperature_2m")
    daily_rain = None
    if isinstance(daily.get("precipitation_sum"), list) and daily["precipitation_sum"]:
        daily_rain = daily["precipitation_sum"][0]
    current_precip = current.get("precipitation")

    if temperature is None:
        raise LiveWeatherError("Live weather response did not include temperature_2m.")

    if daily_rain is None and current_precip is None:
        raise LiveWeatherError("Live weather response did not include precipitation.")

    rainfall = float(daily_rain) if daily_rain is not None else float(current_precip) * 24.0
    return {
        "temperature": float(temperature),
        "rainfall": max(rainfall, 0.0),
        "source": "open-meteo",
    }
