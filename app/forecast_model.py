from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX

MODEL_DIR = Path(__file__).resolve().parent.parent / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)
YIELD_MODEL_PATH = MODEL_DIR / "sarima_yield_results.pkl"
PRICE_MODEL_PATH = MODEL_DIR / "sarima_price_results.pkl"
CROP_PRICE_FACTORS = {
    "Maize": 0.9,
    "Beans": 1.35,
    "Cassava": 0.75,
    "Groundnuts": 1.6,
    "Banana": 1.1,
    "Rice": 1.45,
}


def _historical_series(kind: str) -> pd.Series:
    date_index = pd.date_range(start="2021-01-01", periods=60, freq="MS")
    months = np.arange(len(date_index))
    seasonal = 8 * np.sin(2 * np.pi * months / 12)
    trend = 0.25 * months
    noise = np.random.default_rng(42).normal(0, 2, len(date_index))
    if kind == "yield":
        values = 42 + trend + seasonal + noise
    else:
        values = 1200 + 6 * months + (18 * np.sin(2 * np.pi * months / 12)) + noise * 8
    return pd.Series(values, index=date_index)


def _fit_series(series: pd.Series):
    model = SARIMAX(
        series,
        order=(1, 1, 1),
        seasonal_order=(1, 1, 1, 12),
        enforce_stationarity=False,
        enforce_invertibility=False,
    )
    return model.fit(disp=False)


def ensure_models() -> None:
    if not YIELD_MODEL_PATH.exists():
        _fit_series(_historical_series("yield")).save(str(YIELD_MODEL_PATH))
    if not PRICE_MODEL_PATH.exists():
        _fit_series(_historical_series("price")).save(str(PRICE_MODEL_PATH))


def _load_results(path: Path):
    from statsmodels.tsa.statespace.sarimax import SARIMAXResults

    return SARIMAXResults.load(str(path))


def build_forecast(steps: int = 6) -> dict[str, Any]:
    ensure_models()
    yield_results = _load_results(YIELD_MODEL_PATH)
    price_results = _load_results(PRICE_MODEL_PATH)
    yield_pred = yield_results.get_forecast(steps=steps).predicted_mean
    price_pred = price_results.get_forecast(steps=steps).predicted_mean
    future_dates = pd.date_range(start=pd.Timestamp.today().replace(day=1), periods=steps, freq="MS")

    series = []
    for i in range(steps):
        avg_price_ugx = round(float(price_pred.iloc[i]), 2)
        series.append(
            {
                "month": future_dates[i].strftime("%Y-%m"),
                "yield_forecast_t_per_ha": round(float(yield_pred.iloc[i]), 2),
                "average_price_ugx_per_kg": avg_price_ugx,
            }
        )

    first_month_price = float(price_pred.iloc[0])
    second_month_price = float(price_pred.iloc[1]) if len(price_pred) > 1 else first_month_price
    crop_price_forecast = []
    for crop, factor in CROP_PRICE_FACTORS.items():
        current = round(first_month_price * factor, 2)
        next_month = round(second_month_price * factor, 2)
        trend = "up" if next_month > current else ("down" if next_month < current else "stable")
        crop_price_forecast.append(
            {
                "crop": crop,
                "unit": "UGX/kg",
                "current_price_ugx": current,
                "next_month_price_ugx": next_month,
                "trend": trend,
            }
        )

    return {
        "horizon_months": steps,
        "series": series,
        "crop_price_forecast_ugx": crop_price_forecast,
    }
