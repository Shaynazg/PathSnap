"""Unit and integration tests for candidate classifier model loading and inference."""

import numpy as np
import pandas as pd
import pytest

from src.models.classifier import (
    CandidateClassifier,
    FEATURE_COLUMNS,
    IDENTIFIERS,
    get_classifier,
    predict_candidates,
    prepare_candidate_features,
)


def test_classifier_loads_successfully():
    classifier = get_classifier()
    assert classifier.is_loaded
    assert classifier.model is not None
    assert classifier.model_path.is_file()


def test_prepare_candidate_features_renames_and_handles_nans():
    df_raw = pd.DataFrame([
        {
            "trip_id": "00000000",
            "obs_idx": 0,
            "candidate_arc_id": 1234,
            "distance_m": 5.2,
            "heading_difference_deg": None,
            "gps_speed_mps": None,
            "gps_heading_deg": None,
            "projected_offset_m": 12.0,
            "arc_length_m": 100.0,
            "remaining_distance_m": 88.0,
            "offset_ratio": 0.12,
            "time_delta_s": None,
            "gps_displacement_m": None,
            "candidate_heading_deg": 45.0,
        }
    ])
    prepared = prepare_candidate_features(df_raw)
    for col in FEATURE_COLUMNS:
        assert col in prepared.columns
        assert not pd.isna(prepared[col].iloc[0])

    assert prepared["heading_missing"].iloc[0] == 1
    assert prepared["heading_diff"].iloc[0] == 90.0
    assert prepared["speed_mps"].iloc[0] == 0.0


def test_predict_candidates_preserves_identities_and_normalizes():
    classifier = get_classifier()
    data = [
        {"trip_id": "t1", "obs_idx": 0, "candidate_arc_id": 101, "distance_m": 2.0, "projected_offset_m": 5.0, "arc_length_m": 50.0, "remaining_distance_m": 45.0, "offset_ratio": 0.1, "time_delta_s": 1.0, "gps_displacement_m": 0.0, "candidate_heading_deg": 10.0, "speed_mps": 0.0, "bearing": 0.0, "heading_diff": 10.0, "heading_missing": 0},
        {"trip_id": "t1", "obs_idx": 0, "candidate_arc_id": 102, "distance_m": 20.0, "projected_offset_m": 5.0, "arc_length_m": 50.0, "remaining_distance_m": 45.0, "offset_ratio": 0.1, "time_delta_s": 1.0, "gps_displacement_m": 0.0, "candidate_heading_deg": 10.0, "speed_mps": 0.0, "bearing": 0.0, "heading_diff": 10.0, "heading_missing": 0},
        {"trip_id": "t1", "obs_idx": 1, "candidate_arc_id": 201, "distance_m": 1.0, "projected_offset_m": 5.0, "arc_length_m": 50.0, "remaining_distance_m": 45.0, "offset_ratio": 0.1, "time_delta_s": 1.0, "gps_displacement_m": 0.0, "candidate_heading_deg": 10.0, "speed_mps": 0.0, "bearing": 0.0, "heading_diff": 10.0, "heading_missing": 0},
    ]
    df = pd.DataFrame(data)
    preds = classifier.predict_candidates(df)

    assert len(preds) == 3
    assert list(preds.columns) == ["trip_id", "obs_idx", "candidate_arc_id", "raw_probability", "probability"]
    assert list(preds["candidate_arc_id"]) == [101, 102, 201]

    # Verify normalization per observation
    group0_sum = preds[preds["obs_idx"] == 0]["probability"].sum()
    group1_sum = preds[preds["obs_idx"] == 1]["probability"].sum()

    assert group0_sum == pytest.approx(1.0, rel=1e-5)
    assert group1_sum == pytest.approx(1.0, rel=1e-5)


def test_empty_candidates_handled_gracefully():
    classifier = get_classifier()
    empty_df = pd.DataFrame(columns=IDENTIFIERS + FEATURE_COLUMNS)
    preds = classifier.predict_candidates(empty_df)
    assert len(preds) == 0
    assert "probability" in preds.columns
    assert "candidate_arc_id" in preds.columns
