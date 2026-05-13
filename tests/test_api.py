from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


def test_predict_endpoint():
    payload = {"season": 1, "soil_type": "Clay Loam", "rainfall": 120, "temperature": 25}
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    body = response.json()
    assert "recommended_crop" in body
    assert "confidence" in body
    assert isinstance(body.get("recommendations", []), list)


def test_forecast_endpoint():
    response = client.get("/forecast?steps=4")
    assert response.status_code == 200
    body = response.json()
    assert body["horizon_months"] == 4
    assert len(body["series"]) == 4
    assert "crop_price_forecast_ugx" in body
    assert len(body["crop_price_forecast_ugx"]) > 0


def test_detect_soil_endpoint():
    response = client.post("/detect-soil", json={"latitude": 0.78, "longitude": 32.55})
    assert response.status_code == 200
    body = response.json()
    assert "manual_selection_required" in body


def test_predict_sub_county_endpoint():
    response = client.post("/predict/sub-county", json={"sub_county": "Bamunanika"})
    assert response.status_code == 200
    body = response.json()
    assert "recommended_crop" in body
    assert body["inputs"]["sub_county"] == "Bamunanika"
    assert "season_advice" in body
    assert isinstance(body.get("recommendations", []), list)
    if body["recommendations"]:
        assert "planning" in body["recommendations"][0]

