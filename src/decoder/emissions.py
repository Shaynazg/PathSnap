"""Validation and log-emission scoring for classifier arc predictions."""

from __future__ import annotations

import math
from collections.abc import Hashable
from numbers import Integral, Real
from typing import TypeAlias

import pandas as pd

TripId: TypeAlias = Hashable
EmissionScores: TypeAlias = dict[TripId, dict[int, dict[int, float]]]

REQUIRED_COLUMNS = frozenset(
    {"trip_id", "obs_idx", "candidate_arc_id", "probability"}
)
PROBABILITY_SUM_TOLERANCE = 1e-6


def _is_missing_scalar(value: object) -> bool:
    """Return whether a scalar is missing; reject container-valued fields."""
    if not pd.api.types.is_scalar(value):
        raise TypeError("prediction identifiers and probabilities must be scalars")
    return bool(pd.isna(value))


def _validated_index(value: object, column: str) -> int:
    """Validate an integer identifier without accepting or truncating floats."""
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise TypeError(f"{column} values must be integers")
    index = int(value)
    if index < 0:
        raise ValueError(f"{column} values must be non-negative")
    return index


def _trip_sort_key(trip_id: TripId) -> tuple[str, str, str]:
    """Provide a stable ordering even when trip IDs have mixed scalar types."""
    trip_type = type(trip_id)
    return (trip_type.__module__, trip_type.__qualname__, repr(trip_id))


def compute_emissions(predictions: pd.DataFrame) -> EmissionScores:
    """Validate classifier predictions and return log emission scores.

    The input must contain ``trip_id``, ``obs_idx``, ``candidate_arc_id`` and
    ``probability`` columns. IDs must be non-missing scalar values; observation
    indices and arc IDs must be non-negative integer values (floats are
    rejected, including integral-valued floats). Probabilities must be finite
    values in ``[0, 1]`` and sum to 1 within an absolute tolerance of
    :data:`PROBABILITY_SUM_TOLERANCE` for every ``(trip_id, obs_idx)`` pair.
    Probabilities are validated as provided and are never renormalized.

    Returns a deterministic nested dictionary of the form
    ``trip_id -> obs_idx -> candidate_arc_id -> log_probability``. Trip keys
    are ordered by identifier type and representation; observation and arc
    keys are in ascending numeric order. Zero probabilities map to
    ``-math.inf``. An empty DataFrame with the required columns returns ``{}``.

    Args:
        predictions: A Pandas DataFrame containing classifier candidates.

    Raises:
        TypeError: If the input is not a DataFrame, values have invalid types,
            or IDs are not scalar and hashable.
        ValueError: If required columns are absent or duplicated, values are
            missing or out of range, entries are duplicated, or probability
            sums are outside the documented tolerance.
    """
    if not isinstance(predictions, pd.DataFrame):
        raise TypeError("predictions must be a pandas DataFrame")
    if not predictions.columns.is_unique:
        raise ValueError("predictions must not contain duplicate column names")
    missing_columns = REQUIRED_COLUMNS.difference(predictions.columns)
    if missing_columns:
        names = ", ".join(sorted(missing_columns))
        raise ValueError(f"predictions are missing required columns: {names}")

    if predictions.empty:
        return {}

    scores: dict[TripId, dict[int, dict[int, float]]] = {}
    group_probabilities: dict[tuple[TripId, int], list[float]] = {}
    seen_entries: set[tuple[TripId, int, int]] = set()

    # Read columns separately to retain their scalar types and avoid iterrows()
    # coercing integer IDs to floats when a row also contains probabilities.
    trip_ids = predictions["trip_id"].array
    obs_indices = predictions["obs_idx"].array
    candidate_ids = predictions["candidate_arc_id"].array
    probabilities = predictions["probability"].array

    for trip_id, obs_value, candidate_value, probability_value in zip(
        trip_ids, obs_indices, candidate_ids, probabilities
    ):
        if _is_missing_scalar(trip_id):
            raise ValueError("trip_id values must not be missing")
        try:
            hash(trip_id)
        except TypeError as exc:
            raise TypeError("trip_id values must be hashable") from exc

        if _is_missing_scalar(obs_value):
            raise ValueError("obs_idx values must not be missing")
        if _is_missing_scalar(candidate_value):
            raise ValueError("candidate_arc_id values must not be missing")
        if _is_missing_scalar(probability_value):
            raise ValueError("probability values must not be missing")

        obs_idx = _validated_index(obs_value, "obs_idx")
        candidate_arc_id = _validated_index(candidate_value, "candidate_arc_id")

        if isinstance(probability_value, bool) or not isinstance(
            probability_value, Real
        ):
            raise TypeError("probability values must be real numbers")
        probability = float(probability_value)
        if not math.isfinite(probability):
            raise ValueError("probability values must be finite")
        if probability < 0.0 or probability > 1.0:
            raise ValueError("probability values must be within [0, 1]")

        entry_key = (trip_id, obs_idx, candidate_arc_id)
        if entry_key in seen_entries:
            raise ValueError(
                "duplicate candidate entry for trip_id, obs_idx, and candidate_arc_id"
            )
        seen_entries.add(entry_key)

        group_key = (trip_id, obs_idx)
        group_probabilities.setdefault(group_key, []).append(probability)
        trip_scores = scores.setdefault(trip_id, {})
        observation_scores = trip_scores.setdefault(obs_idx, {})
        observation_scores[candidate_arc_id] = (
            -math.inf if probability == 0.0 else math.log(probability)
        )

    for (trip_id, obs_idx), values in group_probabilities.items():
        total = math.fsum(values)
        if not math.isclose(
            total,
            1.0,
            rel_tol=0.0,
            abs_tol=PROBABILITY_SUM_TOLERANCE,
        ):
            raise ValueError(
                "probabilities for each (trip_id, obs_idx) must sum to 1 "
                f"within absolute tolerance {PROBABILITY_SUM_TOLERANCE}; "
                f"got {total} for ({trip_id!r}, {obs_idx})"
            )

    return {
        trip_id: {
            obs_idx: dict(sorted(observation_scores.items()))
            for obs_idx, observation_scores in sorted(scores[trip_id].items())
        }
        for trip_id in sorted(scores, key=_trip_sort_key)
    }
