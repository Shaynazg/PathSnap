"""End-to-end integration test reconstructing all 10 evaluator test trips."""

import time
import pytest
from src.services.reconstruction import reconstruct_trip

EVALUATOR_TRIPS = [
    "00000000",
    "00000007",
    "00000009",
    "00000040",
    "00000052",
    "00000056",
    "00000065",
    "00000076",
    "00000077",
    "00000091",
]


@pytest.mark.parametrize("trip_id", EVALUATOR_TRIPS)
def test_reconstruct_each_evaluator_trip(trip_id: str):
    t0 = time.perf_counter()
    result = reconstruct_trip(trip_id=trip_id, split="student_test")
    elapsed_ms = (time.perf_counter() - t0) * 1000.0

    assert result["trip_id"] == trip_id
    assert result["total_observations"] > 0
    assert result["total_candidates"] > 0
    assert result["total_decoded_arcs"] > 0
    assert len(result["decoded_arc_sequence"]) > 0
    assert len(result["route_polyline"]) > 0
    assert len(result["gps_points"]) == result["total_observations"]

    # Verify timing
    assert result["timing_ms"]["total_ms"] > 0

    # Print summary for test output inspection
    eval_f1 = result["evaluation"]["viterbi_reconstruction"]["f1_score"] if result.get("evaluation") else None
    print(f"\n[Trip {trip_id}] Obs: {result['total_observations']} | Candidates: {result['total_candidates']} | Decoded Arcs: {result['total_decoded_arcs']} | Viterbi F1: {eval_f1:.4f} | Latency: {elapsed_ms:.1f}ms")
