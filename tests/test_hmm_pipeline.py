"""Integration check for the existing emissions, transitions, and Viterbi APIs."""

from __future__ import annotations

import math
from datetime import datetime, timedelta

import networkx as nx
import pandas as pd
import pytest

from src.decoder.emissions import compute_emissions
from src.decoder.transitions import TransitionScore, compute_transitions
from src.decoder.viterbi import decode_viterbi


def _timestamp(obs_idx: int) -> datetime:
    return datetime(2025, 1, 1) + timedelta(seconds=obs_idx * 60)


def test_emissions_transitions_and_viterbi_outputs_connect_without_adapters() -> None:
    graph = nx.MultiDiGraph()
    for arc_id, source, target, length in [
        (17, "a", "b", 100.0),
        (41, "b", "c", 100.0),
        (29, "c", "d", 100.0),
    ]:
        # NetworkX keys deliberately differ from original arc IDs.
        graph.add_edge(
            source,
            target,
            key=f"edge-key-{arc_id + 1000}",
            arc_id=arc_id,
            from_node_id=source,
            to_node_id=target,
            length_m=length,
        )

    observations = pd.DataFrame(
        [
            ("trip-z", 0, _timestamp(0), 0.0, 0.0),
            ("trip-z", 1, _timestamp(1), 0.0, 0.0),
            ("trip-z", 2, _timestamp(2), 0.0, 0.0),  # no candidates
            ("trip-z", 3, _timestamp(3), 0.0, 0.0),
            ("trip-empty", 0, _timestamp(0), 0.0, 0.0),
            ("trip-empty", 1, _timestamp(1), 0.0, 0.0),
        ],
        columns=["trip_id", "obs_idx", "timestamp", "longitude", "latitude"],
    )

    candidate_specs = [
        ("trip-z", 0, 17, 80.0, 0.6),
        ("trip-z", 0, 29, 10.0, 0.4),
        ("trip-z", 1, 41, 20.0, 0.9),
        ("trip-z", 1, 29, 20.0, 0.1),
        ("trip-z", 3, 29, 30.0, 1.0),
    ]
    observation_lookup = {
        (row.trip_id, row.obs_idx): row
        for row in observations.itertuples(index=False)
    }
    candidates = pd.DataFrame(
        [
            (
                trip_id,
                obs_idx,
                observation_lookup[(trip_id, obs_idx)].timestamp,
                observation_lookup[(trip_id, obs_idx)].longitude,
                observation_lookup[(trip_id, obs_idx)].latitude,
                arc_id,
                0.0,
                offset_m,
            )
            for trip_id, obs_idx, arc_id, offset_m, _probability in candidate_specs
        ],
        columns=[
            "trip_id",
            "obs_idx",
            "timestamp",
            "longitude",
            "latitude",
            "candidate_arc_id",
            "distance_m",
            "projected_offset_m",
        ],
    )
    predictions = pd.DataFrame(
        [
            (trip_id, obs_idx, arc_id, probability)
            for trip_id, obs_idx, arc_id, _offset_m, probability in candidate_specs
        ],
        columns=["trip_id", "obs_idx", "candidate_arc_id", "probability"],
    )

    emissions = compute_emissions(predictions)
    transition_scores = compute_transitions(
        candidates,
        graph,
        observations,
        beta_m=1_000.0,
        max_speed_mps=1_000.0,
        speed_scale_mps=1_000.0,
    )

    assert emissions["trip-z"][0][17] < 0.0
    assert isinstance(transition_scores["trip-z"][1][(17, 41)], TransitionScore)
    assert transition_scores["trip-z"][1][(29, 41)].network_distance_m is None
    assert transition_scores["trip-z"][2] == {}
    assert transition_scores["trip-empty"] == {0: {}, 1: {}}

    result = decode_viterbi(emissions, transition_scores)

    segments = result["trip-z"]
    assert [segment.obs_indices for segment in segments] == [(0, 1), (2,), (3,)]
    assert [segment.candidate_arc_ids for segment in segments] == [
        (17, 41),
        None,
        (29,),
    ]
    assert [segment.has_path for segment in segments] == [True, False, True]
    assert segments[0].total_log_score == pytest.approx(
        math.log(0.6) + math.log(0.9) - 0.04
    )
    assert segments[1].total_log_score == -math.inf
    assert segments[2].total_log_score == 0.0
    assert [segment.has_path for segment in result["trip-empty"]] == [False, False]

