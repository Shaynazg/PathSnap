"""Unit and API tests for FastAPI backend endpoints."""

import pytest
from fastapi.testclient import TestClient
from src.api.main import app

client = TestClient(app)


def test_health_endpoint():
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["model_loaded"] is True
    assert data["available_trips_count"] == 10


def test_trips_endpoint_returns_ten_evaluator_trips():
    response = client.get("/trips")
    assert response.status_code == 200
    data = response.json()
    assert data["total_trips"] == 10
    trip_ids = [t["trip_id"] for t in data["trips"]]
    assert "00000000" in trip_ids
    assert "00000091" in trip_ids


def test_reconstruct_invalid_trip_returns_404():
    response = client.post("/reconstruct", json={"trip_id": "nonexistent_trip_999"})
    assert response.status_code == 404
    assert "not found" in response.json()["detail"].lower()


def test_reconstruct_evaluator_trip():
    # Test on trip 00000000
    response = client.post("/reconstruct", json={"trip_id": "00000000", "split": "student_test"})
    assert response.status_code == 200
    data = response.json()

    assert data["trip_id"] == "00000000"
    assert data["total_observations"] > 0
    assert data["total_candidates"] > 0
    assert data["total_decoded_arcs"] > 0
    assert len(data["route_polyline"]) > 0
    assert len(data["gps_points"]) == data["total_observations"]

    # Verify timing metrics are recorded
    assert "total_ms" in data["timing_ms"]
    assert data["timing_ms"]["total_ms"] > 0

    # Verify evaluation comparison against ground truth
    assert data["evaluation"] is not None
    assert "viterbi_reconstruction" in data["evaluation"]
    assert "classifier_top1_baseline" in data["evaluation"]
    assert data["evaluation"]["viterbi_reconstruction"]["f1_score"] >= 0.0


def test_get_results_cached():
    # First ensure it has been reconstructed
    client.post("/reconstruct", json={"trip_id": "00000000", "split": "student_test"})
    response = client.get("/results/00000000?split=student_test")
    assert response.status_code == 200
    data = response.json()
    assert data["trip_id"] == "00000000"
