"""Tests for classifier probability validation and HMM log emissions."""

import math

import pandas as pd
import pytest

from src.decoder.emissions import (
    PROBABILITY_SUM_TOLERANCE,
    compute_emissions,
)


def _predictions(rows: list[tuple[object, object, object, object]]) -> pd.DataFrame:
    return pd.DataFrame(
        rows,
        columns=["trip_id", "obs_idx", "candidate_arc_id", "probability"],
    )


def test_valid_probabilities_are_grouped_independently_and_sorted() -> None:
    predictions = _predictions(
        [
            ("trip-b", 1, 8, 1.0),
            ("trip-a", 1, 5, 0.6),
            ("trip-a", 0, 4, 0.25),
            ("trip-a", 0, 2, 0.75),
            ("trip-a", 1, 3, 0.4),
        ]
    )

    result = compute_emissions(predictions)

    assert list(result) == ["trip-a", "trip-b"]
    assert list(result["trip-a"]) == [0, 1]
    assert list(result["trip-a"][0]) == [2, 4]
    assert result["trip-a"][0] == {2: math.log(0.75), 4: math.log(0.25)}
    assert result["trip-a"][1] == {3: math.log(0.4), 5: math.log(0.6)}
    assert result["trip-b"][1] == {8: 0.0}


def test_zero_probability_maps_to_negative_infinity() -> None:
    result = compute_emissions(_predictions([("trip", 0, 12, 0.0), ("trip", 0, 19, 1.0)]))

    assert result["trip"][0][12] == -math.inf
    assert result["trip"][0][19] == 0.0


def test_probability_sum_within_documented_tolerance_is_accepted() -> None:
    result = compute_emissions(
        _predictions([("trip", 0, 0, 0.5), ("trip", 0, 1, 0.5 + PROBABILITY_SUM_TOLERANCE / 2)])
    )

    assert set(result["trip"][0]) == {0, 1}
    assert result["trip"][0][0] == math.log(0.5)


@pytest.mark.parametrize(
    "rows",
    [
        [("trip", 0, 0, 0.2), ("trip", 0, 1, 0.2)],
        [("trip", 0, 0, 0.5), ("trip", 0, 1, 0.5 + 2 * PROBABILITY_SUM_TOLERANCE)],
    ],
)
def test_invalid_probability_sums_are_rejected(
    rows: list[tuple[object, object, object, object]],
) -> None:
    with pytest.raises(ValueError, match="must sum to 1"):
        compute_emissions(_predictions(rows))


def test_duplicate_candidate_entries_are_rejected() -> None:
    with pytest.raises(ValueError, match="duplicate candidate entry"):
        compute_emissions(
            _predictions([("trip", 0, 4, 0.5), ("trip", 0, 4, 0.5)])
        )


def test_missing_required_columns_are_rejected() -> None:
    frame = pd.DataFrame({"trip_id": ["trip"], "obs_idx": [0]})

    with pytest.raises(ValueError, match="missing required columns"):
        compute_emissions(frame)


def test_duplicate_column_names_are_rejected() -> None:
    frame = _predictions([("trip", 0, 0, 1.0)])
    frame.columns = ["trip_id", "obs_idx", "candidate_arc_id", "probability"]
    frame["probability_copy"] = frame["probability"]
    frame.columns = ["trip_id", "obs_idx", "candidate_arc_id", "probability", "probability"]

    with pytest.raises(ValueError, match="duplicate column names"):
        compute_emissions(frame)


@pytest.mark.parametrize(
    "column",
    ["trip_id", "obs_idx", "candidate_arc_id", "probability"],
)
def test_missing_values_are_rejected(column: str) -> None:
    row: dict[str, object] = {
        "trip_id": "trip",
        "obs_idx": 0,
        "candidate_arc_id": 0,
        "probability": 1.0,
    }
    row[column] = float("nan")

    with pytest.raises(ValueError, match="must not be missing"):
        compute_emissions(pd.DataFrame([row]))


@pytest.mark.parametrize(
    ("column", "value"),
    [
        ("obs_idx", 1.0),
        ("obs_idx", -1),
        ("obs_idx", "1"),
        ("candidate_arc_id", 2.0),
        ("candidate_arc_id", -2),
        ("candidate_arc_id", "2"),
    ],
)
def test_invalid_indices_are_rejected_without_float_truncation(
    column: str, value: object
) -> None:
    row: dict[str, object] = {
        "trip_id": "trip",
        "obs_idx": 0,
        "candidate_arc_id": 0,
        "probability": 1.0,
    }
    row[column] = value
    error = ValueError if isinstance(value, int) and value < 0 else TypeError

    with pytest.raises(error):
        compute_emissions(pd.DataFrame([row]))


@pytest.mark.parametrize(
    ("probability", "error"),
    [
        (-0.1, ValueError),
        (1.1, ValueError),
        (float("inf"), ValueError),
        (float("nan"), ValueError),
        ("0.5", TypeError),
        (True, TypeError),
    ],
)
def test_invalid_probabilities_are_rejected(
    probability: object, error: type[Exception]
) -> None:
    with pytest.raises(error):
        compute_emissions(_predictions([("trip", 0, 0, probability)]))


def test_empty_input_returns_empty_mapping() -> None:
    empty = pd.DataFrame(
        columns=["trip_id", "obs_idx", "candidate_arc_id", "probability"]
    )

    assert compute_emissions(empty) == {}


def test_non_dataframe_input_is_rejected() -> None:
    with pytest.raises(TypeError, match="pandas DataFrame"):
        compute_emissions([])  # type: ignore[arg-type]
