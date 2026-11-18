
import pytest

from fastapi.testclient import TestClient

from src.api.main import app


client = TestClient(app)


# ==========================================
# 1. HEALTH ENDPOINT TEST
# ==========================================

def test_health_endpoint():

    response = client.get("/health")

    assert response.status_code == 200

    data = response.json()

    assert data["status"] == "healthy"


# ==========================================
# 2. TRIP DISCOVERY TEST
# ==========================================

def test_get_trips():

    response = client.get("/trips")

    assert response.status_code == 200

    data = response.json()

    assert "total_trips" in data
    assert "trips" in data

    assert data["total_trips"] == len(data["trips"])


# ==========================================
# 3. INVALID TRIP ID TEST
# ==========================================

def test_invalid_trip_id():

    response = client.post(
        "/predict/",
        json={"trip_id": "123"}
    )

    assert response.status_code == 422


# ==========================================
# 4. PIPELINE STATUS TEST
# ==========================================

def test_pipeline_status():

    response = client.get("/pipeline/status")

    assert response.status_code == 200

    data = response.json()

    assert "pipeline_ready" in data
    assert "components" in data

    assert data["pipeline_ready"] is False

    assert len(data["components"]) == 4


# ==========================================
# 5. PREDICTION PIPELINE NOT READY TEST
# ==========================================

def test_prediction_pipeline_not_ready():

    response = client.post(
        "/predict/",
        json={"trip_id": "00000000"}
    )

    assert response.status_code == 501

    assert "Prediction pipeline components unavailable" in (
        response.json()["detail"]
    )


# ==========================================
# 6. ARTIFACT STATUS TEST
# ==========================================

def test_artifact_status():

    response = client.get("/artifacts/status")

    assert response.status_code == 200

    data = response.json()

    assert "artifacts_ready" in data
    assert "artifacts" in data


# ==========================================
# 7. GPS LOADER TEST
# ==========================================

from src.api.gps_loader import load_gps_track


def test_gps_loader():

    track_df = load_gps_track("00000000")

    assert not track_df.empty

    assert len(track_df) == 2503

    assert "longitude" in track_df.columns
    assert "latitude" in track_df.columns
    assert "timestamp" in track_df.columns

    assert track_df["longitude"].between(-180, 180).all()

    assert track_df["latitude"].between(-90, 90).all()

    assert track_df["timestamp"].is_monotonic_increasing


# ==========================================
# 8. INVALID GPS TRIP ID TEST
# ==========================================

def test_invalid_gps_trip_id():

    with pytest.raises(ValueError):
        load_gps_track("invalid_trip")
