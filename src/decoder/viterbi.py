"""Viterbi decoding over per-observation candidate road arcs."""

from __future__ import annotations

import math
from collections.abc import Hashable, Mapping
from dataclasses import dataclass
from numbers import Integral, Real
from typing import TypeAlias

import pandas as pd

from src.decoder.transitions import TransitionScore

TripId: TypeAlias = Hashable
CandidateId: TypeAlias = int
CandidatePair: TypeAlias = tuple[CandidateId, CandidateId]
EmissionScores: TypeAlias = Mapping[
    TripId, Mapping[int, Mapping[CandidateId, float]]
]
TransitionScores: TypeAlias = Mapping[
    TripId, Mapping[int, Mapping[CandidatePair, TransitionScore]]
]


@dataclass(frozen=True)
class DecodedSegment:
    """Decoded result for one contiguous run of observations.

    ``candidate_arc_ids`` is aligned positionally with ``obs_indices`` and
    preserves the original candidate IDs. It is ``None`` when the segment has
    no complete finite-scoring path. Segment boundaries are inclusive.
    """

    trip_id: TripId
    start_obs_idx: int
    end_obs_idx: int
    obs_indices: tuple[int, ...]
    candidate_arc_ids: tuple[CandidateId, ...] | None
    total_log_score: float
    has_path: bool


ViterbiResults: TypeAlias = dict[TripId, tuple[DecodedSegment, ...]]


def _trip_sort_key(trip_id: TripId) -> tuple[str, str, str]:
    trip_type = type(trip_id)
    return (trip_type.__module__, trip_type.__qualname__, repr(trip_id))


def _validate_identifier(value: object, name: str) -> Hashable:
    if not pd.api.types.is_scalar(value) or bool(pd.isna(value)):
        raise ValueError(f"{name} must be a non-missing scalar")
    try:
        hash(value)
    except TypeError as exc:
        raise TypeError(f"{name} must be hashable") from exc
    return value


def _validate_index(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise TypeError(f"{name} must be an integer")
    index = int(value)
    if index < 0:
        raise ValueError(f"{name} must be non-negative")
    return index


def _validate_log_score(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a real log score")
    score = float(value)
    if math.isnan(score) or score == math.inf:
        raise ValueError(f"{name} must be finite or negative infinity")
    if score > 0.0:
        raise ValueError(f"{name} must not be positive")
    return score


def _read_emissions(emissions: EmissionScores) -> dict[TripId, dict[int, dict[int, float]]]:
    if not isinstance(emissions, Mapping):
        raise TypeError("emissions must be a mapping")

    validated: dict[TripId, dict[int, dict[int, float]]] = {}
    for trip_value, observations in emissions.items():
        trip_id = _validate_identifier(trip_value, "emission trip_id")
        if not isinstance(observations, Mapping):
            raise TypeError(f"emissions for trip {trip_id!r} must be a mapping")
        trip_emissions: dict[int, dict[int, float]] = {}
        for obs_value, candidates in observations.items():
            obs_idx = _validate_index(obs_value, "emission obs_idx")
            if obs_idx in trip_emissions:
                raise ValueError(f"duplicate emission observation {obs_idx}")
            if not isinstance(candidates, Mapping):
                raise TypeError("emission candidates must be a mapping")
            candidate_scores: dict[int, float] = {}
            for candidate_value, score_value in candidates.items():
                candidate_id = _validate_index(candidate_value, "candidate_arc_id")
                if candidate_id in candidate_scores:
                    raise ValueError(
                        f"duplicate candidate arc {candidate_id} at observation {obs_idx}"
                    )
                candidate_scores[candidate_id] = _validate_log_score(
                    score_value, "emission log probability"
                )
            trip_emissions[obs_idx] = candidate_scores
        validated[trip_id] = trip_emissions
    return validated


def _read_transitions(
    transitions: TransitionScores,
) -> dict[TripId, dict[int, dict[CandidatePair, TransitionScore]]]:
    if not isinstance(transitions, Mapping):
        raise TypeError("transitions must be a mapping")

    validated: dict[TripId, dict[int, dict[CandidatePair, TransitionScore]]] = {}
    for trip_value, observations in transitions.items():
        trip_id = _validate_identifier(trip_value, "transition trip_id")
        if not isinstance(observations, Mapping):
            raise TypeError(f"transitions for trip {trip_id!r} must be a mapping")
        trip_transitions: dict[int, dict[CandidatePair, TransitionScore]] = {}
        for obs_value, pairs in observations.items():
            obs_idx = _validate_index(obs_value, "transition obs_idx")
            if obs_idx in trip_transitions:
                raise ValueError(f"duplicate transition observation {obs_idx}")
            if not isinstance(pairs, Mapping):
                raise TypeError("transition pairs must be a mapping")
            pair_scores: dict[CandidatePair, TransitionScore] = {}
            for pair_value, transition_score in pairs.items():
                if not isinstance(pair_value, tuple) or len(pair_value) != 2:
                    raise TypeError("transition pair keys must be (previous_arc_id, current_arc_id)")
                pair = (
                    _validate_index(pair_value[0], "previous candidate_arc_id"),
                    _validate_index(pair_value[1], "current candidate_arc_id"),
                )
                if not isinstance(transition_score, TransitionScore):
                    raise TypeError("transition values must be TransitionScore objects")
                log_score = _validate_log_score(
                    transition_score.log_score, "transition log score"
                )
                network_distance = transition_score.network_distance_m
                if network_distance is None:
                    if log_score != -math.inf:
                        raise ValueError(
                            "unreachable transitions must have negative-infinity score"
                        )
                else:
                    if isinstance(network_distance, bool) or not isinstance(
                        network_distance, Real
                    ):
                        raise TypeError("network_distance_m must be a real number or None")
                    if not math.isfinite(float(network_distance)) or network_distance < 0:
                        raise ValueError("network_distance_m must be finite and non-negative")
                    if log_score == -math.inf:
                        raise ValueError(
                            "reachable transitions must have a finite log score"
                        )
                pair_scores[pair] = TransitionScore(
                    None if network_distance is None else float(network_distance),
                    log_score,
                )
            trip_transitions[obs_idx] = pair_scores
        validated[trip_id] = trip_transitions
    return validated


def _add_log_scores(left: float, right: float) -> float:
    """Add non-positive log scores while keeping reachable paths finite."""
    if left == -math.inf or right == -math.inf:
        return -math.inf
    score = left + right
    # Finite negative values can overflow to -inf when added. Preserve their
    # reachable status while saturating at the lowest finite float.
    return score if math.isfinite(score) else -float.fromhex("0x1.fffffffffffffp+1023")


def _no_path_segment(trip_id: TripId, obs_indices: tuple[int, ...]) -> DecodedSegment:
    return DecodedSegment(
        trip_id=trip_id,
        start_obs_idx=obs_indices[0],
        end_obs_idx=obs_indices[-1],
        obs_indices=obs_indices,
        candidate_arc_ids=None,
        total_log_score=-math.inf,
        has_path=False,
    )


def _decode_run(
    trip_id: TripId,
    obs_indices: tuple[int, ...],
    emissions: Mapping[int, Mapping[int, float]],
    transitions: Mapping[int, Mapping[CandidatePair, TransitionScore]],
) -> DecodedSegment:
    first_obs_idx = obs_indices[0]
    delta: dict[int, float] = dict(emissions[first_obs_idx])
    backpointers: list[dict[int, int]] = [{}]

    for obs_idx in obs_indices[1:]:
        current_delta: dict[int, float] = {}
        current_backpointers: dict[int, int] = {}
        pair_scores = transitions[obs_idx]
        for current_arc_id in sorted(emissions[obs_idx]):
            best_score = -math.inf
            best_previous: int | None = None
            emission_score = emissions[obs_idx][current_arc_id]
            if emission_score != -math.inf:
                for previous_arc_id in sorted(delta):
                    previous_score = delta[previous_arc_id]
                    pair_score = pair_scores[(previous_arc_id, current_arc_id)].log_score
                    candidate_score = _add_log_scores(previous_score, pair_score)
                    if candidate_score > best_score or (
                        candidate_score == best_score
                        and candidate_score != -math.inf
                        and (best_previous is None or previous_arc_id < best_previous)
                    ):
                        best_score = candidate_score
                        best_previous = previous_arc_id
                if best_previous is not None:
                    best_score = _add_log_scores(best_score, emission_score)
                    current_backpointers[current_arc_id] = best_previous
            current_delta[current_arc_id] = best_score
        delta = current_delta
        backpointers.append(current_backpointers)

    terminal_arc_id = min(
        delta,
        key=lambda arc_id: (-delta[arc_id], arc_id),
        default=None,
    )
    if terminal_arc_id is None or delta[terminal_arc_id] == -math.inf:
        return _no_path_segment(trip_id, obs_indices)

    path = [terminal_arc_id]
    for time_index in range(len(obs_indices) - 1, 0, -1):
        previous_arc_id = backpointers[time_index].get(path[-1])
        if previous_arc_id is None:
            # A finite terminal score must have a complete chain; keep this
            # guard to prevent returning a fabricated partial path on bad data.
            return _no_path_segment(trip_id, obs_indices)
        path.append(previous_arc_id)
    path.reverse()

    return DecodedSegment(
        trip_id=trip_id,
        start_obs_idx=obs_indices[0],
        end_obs_idx=obs_indices[-1],
        obs_indices=obs_indices,
        candidate_arc_ids=tuple(path),
        total_log_score=delta[terminal_arc_id],
        has_path=True,
    )


def decode_viterbi(
    emissions: EmissionScores,
    transitions: TransitionScores,
) -> ViterbiResults:
    """Decode the highest-scoring arc sequence for each trip.

    Args:
        emissions: ``trip_id -> obs_idx -> candidate_arc_id -> log_probability``
            from :func:`src.decoder.emissions.compute_emissions`.
        transitions: ``trip_id -> current_obs_idx ->
            (previous_arc_id, current_arc_id) -> TransitionScore`` from
            :func:`src.decoder.transitions.compute_transitions`.

    Returns:
        A deterministic mapping from each trip ID to a tuple of
        :class:`DecodedSegment` objects. Segment arc IDs align with their
        original observation indices; no IDs are renumbered. A segment with
        no complete finite-scoring path has ``has_path=False``,
        ``candidate_arc_ids=None`` and ``total_log_score=-inf``. Each
        observation with no candidates is emitted as its own no-path segment,
        separating the decodable runs on either side.

    Raises:
        TypeError: If mappings, IDs, or score values have incompatible types.
        ValueError: If indices, log scores, transitions, or the correspondence
            between emission and transition mappings is invalid.
    """
    emission_data = _read_emissions(emissions)
    transition_data = _read_transitions(transitions)

    transition_trip_ids = set(transition_data)
    missing_transition_trips = set(emission_data) - transition_trip_ids
    if missing_transition_trips:
        raise ValueError(
            "emission trips are missing from transitions: "
            f"{sorted(missing_transition_trips, key=_trip_sort_key)!r}"
        )

    results: ViterbiResults = {}
    for trip_id in sorted(transition_data, key=_trip_sort_key):
        trip_transitions = transition_data[trip_id]
        trip_emissions = emission_data.get(trip_id, {})
        obs_indices = sorted(trip_transitions)
        if not obs_indices:
            if trip_emissions:
                raise ValueError(
                    f"emissions for trip {trip_id!r} have no transition observations"
                )
            results[trip_id] = ()
            continue

        for left, right in zip(obs_indices, obs_indices[1:]):
            if right != left + 1:
                raise ValueError(
                    f"transition observations for trip {trip_id!r} are not contiguous: "
                    f"{left} followed by {right}"
                )
        extra_emission_indices = set(trip_emissions) - set(obs_indices)
        if extra_emission_indices:
            raise ValueError(
                f"emissions for trip {trip_id!r} contain observations absent from transitions"
            )

        candidates_by_obs = {
            obs_idx: trip_emissions.get(obs_idx, {}) for obs_idx in obs_indices
        }
        for position, obs_idx in enumerate(obs_indices):
            current_pairs = trip_transitions[obs_idx]
            if position == 0:
                if current_pairs:
                    raise ValueError(
                        f"first observation {obs_idx} for trip {trip_id!r} "
                        "must not have incoming transition pairs"
                    )
                continue

            previous_idx = obs_indices[position - 1]
            previous_candidates = candidates_by_obs[previous_idx]
            current_candidates = candidates_by_obs[obs_idx]
            expected_pairs = {
                (previous_arc_id, current_arc_id)
                for previous_arc_id in previous_candidates
                for current_arc_id in current_candidates
            }
            actual_pairs = set(current_pairs)
            if actual_pairs != expected_pairs:
                raise ValueError(
                    f"transition pairs for trip {trip_id!r} observation {obs_idx} "
                    "do not match adjacent emission candidates"
                )

        segments: list[DecodedSegment] = []
        run: list[int] = []

        def finish_run() -> None:
            if run:
                segments.append(
                    _decode_run(
                        trip_id,
                        tuple(run),
                        candidates_by_obs,
                        trip_transitions,
                    )
                )
                run.clear()

        for obs_idx in obs_indices:
            if candidates_by_obs[obs_idx]:
                run.append(obs_idx)
            else:
                finish_run()
                segments.append(_no_path_segment(trip_id, (obs_idx,)))
        finish_run()
        results[trip_id] = tuple(segments)

    return results

