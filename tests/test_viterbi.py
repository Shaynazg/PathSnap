"""Unit tests for Viterbi decoding over synthetic log-score mappings."""

from __future__ import annotations

import math

import pytest

from src.decoder.transitions import TransitionScore
from src.decoder.viterbi import DecodedSegment, decode_viterbi


def _transition(score: float) -> TransitionScore:
    return TransitionScore(None, -math.inf) if score == -math.inf else TransitionScore(0.0, score)


def test_single_observation_selects_best_candidate_and_keeps_original_id() -> None:
    result = decode_viterbi(
        {"trip": {0: {91: -2.0, 7: -0.25}}},
        {"trip": {0: {}}},
    )

    assert result["trip"] == (
        DecodedSegment("trip", 0, 0, (0,), (7,), -0.25, True),
    )


def test_recurrence_finds_global_optimum_and_backtracks_full_path() -> None:
    emissions = {
        "trip": {
            0: {10: -0.1, 20: -0.5},
            1: {30: -0.01, 40: -0.05},
            2: {50: -0.01, 60: -0.02},
        }
    }
    transitions = {
        "trip": {
            0: {},
            1: {
                (10, 30): _transition(-4.0),
                (10, 40): _transition(-0.1),
                (20, 30): _transition(-0.1),
                (20, 40): _transition(-4.0),
            },
            2: {
                (30, 50): _transition(-4.0),
                (30, 60): _transition(-0.1),
                (40, 50): _transition(-0.1),
                (40, 60): _transition(-4.0),
            },
        }
    }

    result = decode_viterbi(emissions, transitions)
    segment = result["trip"][0]

    # The locally best candidates (10, 30, 50) do not form the global optimum.
    assert segment.candidate_arc_ids == (10, 40, 50)
    assert segment.obs_indices == (0, 1, 2)
    assert segment.total_log_score == pytest.approx(-0.36)
    assert segment.has_path


def test_ties_choose_lowest_current_and_previous_arc_ids() -> None:
    result = decode_viterbi(
        {"trip": {0: {90: -1.0, 12: -1.0}, 1: {80: -1.0, 4: -1.0}}},
        {
            "trip": {
                0: {},
                1: {
                    (12, 4): _transition(-1.0),
                    (12, 80): _transition(-1.0),
                    (90, 4): _transition(-1.0),
                    (90, 80): _transition(-1.0),
                },
            }
        },
    )

    assert result["trip"][0].candidate_arc_ids == (12, 4)


def test_zero_emission_is_not_selected_over_finite_candidate() -> None:
    result = decode_viterbi(
        {"trip": {0: {13: -math.inf, 71: -0.3}}},
        {"trip": {0: {}}},
    )

    assert result["trip"][0].candidate_arc_ids == (71,)
    assert result["trip"][0].total_log_score == -0.3


def test_all_zero_emissions_return_explicit_no_path_segment() -> None:
    result = decode_viterbi(
        {"trip": {0: {13: -math.inf, 71: -math.inf}}},
        {"trip": {0: {}}},
    )

    segment = result["trip"][0]
    assert not segment.has_path
    assert segment.candidate_arc_ids is None
    assert segment.total_log_score == -math.inf


def test_unreachable_transitions_return_no_path_without_fabricating_route() -> None:
    result = decode_viterbi(
        {"trip": {0: {10: 0.0}, 1: {20: 0.0}}},
        {"trip": {0: {}, 1: {(10, 20): _transition(-math.inf)}}},
    )

    segment = result["trip"][0]
    assert segment.obs_indices == (0, 1)
    assert segment.candidate_arc_ids is None
    assert segment.total_log_score == -math.inf
    assert not segment.has_path


def test_missing_candidate_observation_splits_trip_into_segments() -> None:
    result = decode_viterbi(
        {"trip": {0: {10: -0.2}, 2: {30: -0.3}}},
        {"trip": {0: {}, 1: {}, 2: {}}},
    )

    segments = result["trip"]
    assert [segment.obs_indices for segment in segments] == [(0,), (1,), (2,)]
    assert [segment.has_path for segment in segments] == [True, False, True]
    assert segments[1].candidate_arc_ids is None
    assert segments[1].total_log_score == -math.inf
    assert segments[0].candidate_arc_ids == (10,)
    assert segments[2].candidate_arc_ids == (30,)


def test_single_candidate_per_observation_decodes_sequence() -> None:
    result = decode_viterbi(
        {"trip": {0: {100: -0.1}, 1: {500: -0.2}, 2: {900: -0.3}}},
        {
            "trip": {
                0: {},
                1: {(100, 500): _transition(-0.4)},
                2: {(500, 900): _transition(-0.5)},
            }
        },
    )

    segment = result["trip"][0]
    assert segment.candidate_arc_ids == (100, 500, 900)
    assert segment.total_log_score == pytest.approx(-1.5)


def test_trips_are_decoded_independently_and_deterministically_ordered() -> None:
    result = decode_viterbi(
        {"z-trip": {0: {41: -1.0}}, "a-trip": {0: {99: -0.1}}},
        {"z-trip": {0: {}}, "a-trip": {0: {}}},
    )

    assert list(result) == ["a-trip", "z-trip"]
    assert result["a-trip"][0].candidate_arc_ids == (99,)
    assert result["z-trip"][0].candidate_arc_ids == (41,)


def test_empty_mappings_return_empty_result() -> None:
    assert decode_viterbi({}, {}) == {}


def test_trip_with_only_empty_observations_has_explicit_no_path_segments() -> None:
    result = decode_viterbi({}, {"trip": {3: {}, 4: {}}})

    assert [segment.obs_indices for segment in result["trip"]] == [(3,), (4,)]
    assert all(not segment.has_path for segment in result["trip"])


def test_inconsistent_missing_transition_pair_is_rejected() -> None:
    with pytest.raises(ValueError, match="do not match"):
        decode_viterbi(
            {"trip": {0: {10: 0.0}, 1: {20: 0.0}}},
            {"trip": {0: {}, 1: {}}},
        )


@pytest.mark.parametrize("bad_score", [float("nan"), math.inf, 0.1])
def test_invalid_emission_log_scores_are_rejected(bad_score: float) -> None:
    with pytest.raises(ValueError):
        decode_viterbi({"trip": {0: {1: bad_score}}}, {"trip": {0: {}}})


def test_transition_objects_must_match_unreachable_score_semantics() -> None:
    with pytest.raises(ValueError, match="unreachable transitions"):
        decode_viterbi(
            {"trip": {0: {1: 0.0}, 1: {2: 0.0}}},
            {"trip": {0: {}, 1: {(1, 2): TransitionScore(None, -1.0)}}},
        )


def test_noncontiguous_transition_observations_are_rejected() -> None:
    with pytest.raises(ValueError, match="not contiguous"):
        decode_viterbi(
            {"trip": {0: {1: 0.0}, 2: {2: 0.0}}},
            {"trip": {0: {}, 2: {}}},
        )

