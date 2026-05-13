from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

DATA_PATH = Path(__file__).resolve().parent / "data" / "crop_dataset.csv"
MODEL_PATH = Path(__file__).resolve().parent / "model_pipeline.joblib"


def _synthetic_dataset() -> pd.DataFrame:
    np.random.seed(42)
    n = 1500
    seasons = np.random.choice(["First", "Second"], n)
    soils = np.random.choice(["Loam", "Clay Loam", "Sandy Loam"], n)
    temps = np.random.normal(26.5, 4.5, n).clip(18, 35)
    rainfalls = np.random.normal(170, 85, n).clip(50, 420)
    crops = []
    for season, soil, temp, rain in zip(seasons, soils, temps, rainfalls):
        if season == "First" and 150 < rain < 280 and 22 < temp < 30 and soil in {"Loam", "Clay Loam"}:
            crops.append("Maize")
        elif season == "First" and soil == "Sandy Loam" and rain > 180:
            crops.append("Cassava")
        elif 20 < temp < 28 and soil == "Loam":
            crops.append("Beans")
        elif rain > 120 and soil in {"Loam", "Clay Loam"}:
            crops.append("Groundnuts")
        elif rain > 220 and temp > 24:
            crops.append("Rice")
        else:
            crops.append(np.random.choice(["Maize", "Beans", "Cassava", "Groundnuts", "Banana", "Rice"]))
    return pd.DataFrame(
        {
            "season": seasons,
            "soil_type": soils,
            "temperature": temps,
            "rainfall": rainfalls,
            "crop": crops,
        }
    )


def _find_column(df: pd.DataFrame, options: list[str]) -> str | None:
    normalized = {str(col).strip().lower(): str(col) for col in df.columns}
    for option in options:
        candidate = normalized.get(option.strip().lower())
        if candidate:
            return candidate
    return None


def _soil_from_ph(ph_value: float | None) -> str:
    if ph_value is None or pd.isna(ph_value):
        return "Loam"
    if ph_value < 5.8:
        return "Clay Loam"
    if ph_value > 7.2:
        return "Sandy Loam"
    return "Loam"


def _prepare_dataset(df: pd.DataFrame) -> pd.DataFrame:
    crop_col = _find_column(df, ["crop", "label", "crop_name", "recommendation"])
    temp_col = _find_column(df, ["temperature", "temp", "avg_temperature"])
    rain_col = _find_column(df, ["rainfall", "rain", "precipitation", "annual_rainfall"])
    season_col = _find_column(df, ["season", "crop_season"])
    soil_col = _find_column(df, ["soil_type", "soil", "soil_class"])
    ph_col = _find_column(df, ["ph", "soil_ph"])

    if crop_col is None or temp_col is None or rain_col is None:
        raise ValueError(
            "Dataset must include crop/label, temperature, and rainfall columns. "
            f"Available columns: {list(df.columns)}"
        )

    prepared = pd.DataFrame(
        {
            "temperature": pd.to_numeric(df[temp_col], errors="coerce"),
            "rainfall": pd.to_numeric(df[rain_col], errors="coerce"),
            "crop": df[crop_col].astype(str).str.strip(),
        }
    )
    if season_col is not None:
        prepared["season"] = df[season_col].astype(str).str.strip().replace({"1": "First", "2": "Second"})
    else:
        prepared["season"] = np.where(prepared["rainfall"] >= 150, "First", "Second")

    if soil_col is not None:
        prepared["soil_type"] = df[soil_col].astype(str).str.strip()
    else:
        ph_series = pd.to_numeric(df[ph_col], errors="coerce") if ph_col is not None else pd.Series([None] * len(df))
        prepared["soil_type"] = ph_series.apply(_soil_from_ph)

    prepared["season"] = prepared["season"].replace(
        {"first season": "First", "second season": "Second", "first": "First", "second": "Second"}
    )
    prepared["soil_type"] = prepared["soil_type"].replace(
        {
            "ferrallitic": "Clay Loam",
            "clay": "Clay Loam",
            "sandy": "Sandy Loam",
            "sandy loam": "Sandy Loam",
            "clay loam": "Clay Loam",
            "loam": "Loam",
        }
    )
    return prepared.dropna(subset=["season", "soil_type", "temperature", "rainfall", "crop"])


def _load_dataset() -> pd.DataFrame:
    if DATA_PATH.exists():
        print(f"Using local CSV dataset: {DATA_PATH}")
        return _prepare_dataset(pd.read_csv(DATA_PATH))
    print("Falling back to synthetic dataset.")
    return _synthetic_dataset()


def main() -> None:
    df = _load_dataset().dropna().copy()
    season_encoder = LabelEncoder()
    soil_encoder = LabelEncoder()
    crop_encoder = LabelEncoder()
    df["season_encoded"] = season_encoder.fit_transform(df["season"])
    df["soil_type_encoded"] = soil_encoder.fit_transform(df["soil_type"])
    df["crop_encoded"] = crop_encoder.fit_transform(df["crop"])
    X = df[["season_encoded", "soil_type_encoded", "temperature", "rainfall"]]
    y = df["crop_encoded"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    accuracy = accuracy_score(y_test, preds)
    cm = confusion_matrix(y_test, preds)
    artifact = {
        "model": model,
        "encoders": {"season": season_encoder, "soil_type": soil_encoder, "crop": crop_encoder},
        "feature_columns": ["season_encoded", "soil_type_encoded", "temperature", "rainfall"],
    }
    joblib.dump(artifact, MODEL_PATH)
    print(f"Model saved to {MODEL_PATH}")
    print(f"Accuracy: {accuracy:.4f}")
    print("Confusion matrix:")
    print(cm)
    print("Target accuracy >= 85% achieved." if accuracy >= 0.85 else "Target accuracy >= 85% not reached.")


if __name__ == "__main__":
    main()
