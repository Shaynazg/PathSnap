
"""Synthetic tests for directed transition distances and scores."""

from __future__ import annotations

import math
from datetime import datetime, timedelta
from typing import Any

import pandas as pd
import pytest
import networkx as nx

from src.decoder.transitions import (
    EARTH_RADIUS_M,
    EndpointDistanceCache,
    TransitionScore,
    compute_transitions,
)


class FakeMultiDiGraph:
    """Small NetworkX MultiDiGraph-compatible test double."""

    def __init__(self, *, directed: bool = True, multigraph: bool = True) -> None:
        self.directed = directed
        self.multigraph = multigraph
        self._edges: list[tuple[Any, Any, Any, dict[str, Any]]] = []
        self._nodes: set[Any] = set()
        self.successor_calls = 0

    def add_edge(self, source: Any, target: Any, key: Any, **attributes: Any) -> None:
        self._nodes.update((source, target))
        self._edges.append((source, target, key, attributes))

    def is_directed(self) -> bool:
        return self.directed

    def is_multigraph(self) -> bool:
        return self.multigraph

    def edges(self, keys: bool = False, data: bool = False) -> list[tuple[Any, ...]]:
        assert keys and data
        return list(self._edges)

    def successors(self, node: Any) -> tuple[Any, ...]:
        self.successor_calls += 1
        return tuple(dict.fromkeys(v for u, v, _key, _data in self._edges if u == node))

    def get_edge_data(self, source: Any, target: Any) -> dict[Any, dict[str, Any]]:
        return {
            key: data
            for u, v, key, data in self._edges
            if u == source and v == target
        }


def _graph(
    arcs: list[tuple[int, str, str, float]],
) -> FakeMultiDiGraph:
    graph = FakeMultiDiGraph()
    for arc_id, source, target, length in arcs:
        graph.add_edge(
            source,
            target,
            key=f"networkx-key-{arc_id}",
            arc_id=arc_id,
            from_node_id=source,
            to_node_id=target,
            length_m=length,
        )
    return graph


def _networkx_graph(
    arcs: list[tuple[int, str, str, float, str]],
) -> nx.MultiDiGraph:
    graph = nx.MultiDiGraph()
    for arc_id, source, target, length, edge_key in arcs:
        graph.add_edge(
            source,
            target,
            key=edge_key,
            arc_id=arc_id,
            from_node_id=source,
            to_node_id=target,
            length_m=length,
        )
    return graph


def _frames(
    observations: list[tuple[str, int, datetime, float, float]],
    candidates: list[tuple[str, int, int, float]],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    observation_frame = pd.DataFrame(
        observations,
        columns=["trip_id", "obs_idx", "timestamp", "longitude", "latitude"],
    )
    observation_lookup = {
        (trip_id, obs_idx): (timestamp, longitude, latitude)
        for trip_id, obs_idx, timestamp, longitude, latitude in observations
    }
    candidate_rows = []
    for trip_id, obs_idx, arc_id, offset in candidates:
        timestamp, longitude, latitude = observation_lookup[(trip_id, obs_idx)]
        candidate_rows.append(
            (trip_id, obs_idx, timestamp, longitude, latitude, arc_id, 0.0, offset)
        )
    candidate_frame = pd.DataFrame(
        candidate_rows,
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
    return candidate_frame, observation_frame


def _score(
    graph: FakeMultiDiGraph,
    observations: list[tuple[str, int, datetime, float, float]],
    candidates: list[tuple[str, int, int, float]],
    **kwargs: Any,
) -> dict[str, dict[int, dict[tuple[int, int], TransitionScore]]]:
    candidate_frame, observation_frame = _frames(observations, candidates)
    parameters = {
        "beta_m": 10.0,
        "max_speed_mps": 30.0,
        "speed_scale_mps": 10.0,
    }
    parameters.update(kwargs)
    return compute_transitions(
        candidate_frame, graph, observation_frame, **parameters
    )


def _time(seconds: float = 0.0) -> datetime:
    return datetime(2024, 1, 1) + timedelta(seconds=seconds)


def test_directly_connected_arcs_distance_and_score() -> None:
    graph = _graph([(10, "a", "b", 100.0), (20, "b", "c", 80.0)])
    result = _score(
        graph,
        [("trip", 0, _time(), 0.0, 0.0), ("trip", 1, _time(10), 0.0, 0.0)],
        [("trip", 0, 10, 40.0), ("trip", 1, 20, 20.0)],
    )

    score = result["trip"][1][(10, 20)]
    assert score.network_distance_m == 80.0
    assert score.log_score == -8.0


def test_multihop_directed_path_uses_shortest_route() -> None:
    graph = _graph(
        [
            (1, "a", "b", 100.0),
            (2, "b", "c", 10.0),
            (3, "c", "d", 15.0),
            (4, "b", "e", 40.0),
            (5, "e", "d", 40.0),
            (6, "d", "z", 100.0),
        ]
    )
    result = _score(
        graph,
        [("trip", 0, _time(), 0.0, 0.0), ("trip", 1, _time(10), 0.0, 0.0)],
        [("trip", 0, 1, 70.0), ("trip", 1, 6, 10.0)],
    )

    assert result["trip"][1][(1, 6)].network_distance_m == 65.0


def test_reverse_direction_is_unreachable_without_directed_return_path() -> None:
    graph = _graph([(1, "a", "b", 10.0), (2, "c", "b", 10.0)])
    result = _score(
        graph,
        [("trip", 0, _time(), 0.0, 0.0), ("trip", 1, _time(5), 0.0, 0.0)],
        [("trip", 0, 1, 5.0), ("trip", 1, 2, 5.0)],
    )

    score = result["trip"][1][(1, 2)]
    assert score == TransitionScore(None, -math.inf)


def test_parallel_edges_retain_arc_identity_and_shortest_weight() -> None:
    graph = _graph(
        [
            (1, "a", "b", 10.0),
            (2, "b", "c", 9.0),
            (3, "b", "c", 2.0),
            (4, "c", "d", 10.0),
            (5, "c", "d", 10.0),
        ]
    )
    result = _score(
        graph,
        [("trip", 0, _time(), 0.0, 0.0), ("trip", 1, _time(5), 0.0, 0.0)],
        [("trip", 0, 1, 5.0), ("trip", 1, 4, 3.0), ("trip", 1, 5, 4.0)],
    )

    assert result["trip"][1][(1, 4)].network_distance_m == 10.0
    assert result["trip"][1][(1, 5)].network_distance_m == 11.0
    assert (1, 4) != (1, 5)


def test_same_arc_forward_and_backward_with_directed_loop() -> None:
    graph = _graph(
        [
            (7, "a", "b", 100.0),
            (8, "b", "c", 5.0),
            (9, "c", "a", 7.0),
        ]
    )
    forward = _score(
        graph,
        [("trip", 0, _time(), 0.0, 0.0), ("trip", 1, _time(5), 0.0, 0.0)],
        [("trip", 0, 7, 30.0), ("trip", 1, 7, 60.0)],
    )
    backward = _score(
        graph,
        [("trip", 0, _time(), 0.0, 0.0), ("trip", 1, _time(5), 0.0, 0.0)],
        [("trip", 0, 7, 80.0), ("trip", 1, 7, 20.0)],
    )

    assert forward["trip"][1][(7, 7)].network_distance_m == 30.0
    assert backward["trip"][1][(7, 7)].network_distance_m == 52.0


def test_backward_same_arc_without_loop_is_unreachable() -> None:
    graph = _graph([(7, "a", "b", 100.0)])
    result = _score(
        graph,
        [("trip", 0, _time(), 0.0, 0.0), ("trip", 1, _time(5), 0.0, 0.0)],
        [("trip", 0, 7, 80.0), ("trip", 1, 7, 20.0)],
    )

    assert result["trip"][1][(7, 7)] == TransitionScore(None, -math.inf)


def test_disconnected_components_return_negative_infinity() -> None:
    graph = _graph([(1, "a", "b", 10.0), (2, "c", "d", 10.0)])
    result = _score(
        graph,
        [("trip", 0, _time(), 0.0, 0.0), ("trip", 1, _time(5), 0.0, 0.0)],
        [("trip", 0, 1, 3.0), ("trip", 1, 2, 4.0)],
    )

    assert result["trip"][1][(1, 2)] == TransitionScore(None, -math.inf)


def test_large_gps_gap_does_not_invent_a_route_between_components() -> None:
    graph = _graph([(1, "a", "b", 10.0), (2, "c", "d", 10.0)])
    result = _score(
        graph,
        [
            ("trip", 0, _time(), 0.0, 0.0),
            ("trip", 1, _time(86_400), 10.0, 0.0),
        ],
        [("trip", 0, 1, 5.0), ("trip", 1, 2, 5.0)],
    )

    assert result["trip"][1][(1, 2)] == TransitionScore(None, -math.inf)


def test_multiple_trips_and_observations_are_independent() -> None:
    graph = _graph([(1, "a", "b", 20.0)])
    result = _score(
        graph,
        [
            ("b", 0, _time(), 0.0, 0.0),
            ("b", 1, _time(1), 0.0, 0.0),
            ("a", 0, _time(), 0.0, 0.0),
            ("a", 1, _time(2), 0.0, 0.0),
            ("a", 2, _time(4), 0.0, 0.0),
        ],
        [
            ("b", 0, 1, 0.0),
            ("b", 1, 1, 10.0),
            ("a", 0, 1, 0.0),
            ("a", 1, 1, 5.0),
            ("a", 2, 1, 10.0),
        ],
    )

    assert list(result) == ["a", "b"]
    assert list(result["a"]) == [0, 1, 2]
    assert result["a"][1][(1, 1)].network_distance_m == 5.0
    assert result["a"][2][(1, 1)].network_distance_m == 5.0
    assert result["b"][1][(1, 1)].network_distance_m == 10.0


def test_missing_candidate_observation_is_explicit_and_not_bridged() -> None:
    graph = _graph([(1, "a", "b", 20.0)])
    result = _score(
        graph,
        [
            ("trip", 0, _time(), 0.0, 0.0),
            ("trip", 1, _time(1), 0.0, 0.0),
            ("trip", 2, _time(2), 0.0, 0.0),
        ],
        [("trip", 0, 1, 0.0), ("trip", 2, 1, 10.0)],
    )

    assert result["trip"] == {0: {}, 1: {}, 2: {}}


def test_haversine_distance_and_elapsed_speed_penalty() -> None:
    graph = _graph([(1, "a", "b", 2_000.0)])
    result = _score(
        graph,
        [("trip", 0, _time(), 0.0, 0.0), ("trip", 1, _time(10), 0.001, 0.0)],
        [("trip", 0, 1, 0.0), ("trip", 1, 1, 120.0)],
    )
    gps_distance = EARTH_RADIUS_M * math.radians(0.001)

    assert result["trip"][1][(1, 1)].network_distance_m == 120.0
    assert result["trip"][1][(1, 1)].log_score == pytest.approx(
        -abs(120.0 - gps_distance) / 10.0
    )


def test_reachable_but_implausible_transition_has_finite_score() -> None:
    graph = _graph([(1, "a", "b", 1_000.0)])
    result = _score(
        graph,
        [("trip", 0, _time(), 0.0, 0.0), ("trip", 1, _time(1), 0.0, 0.0)],
        [("trip", 0, 1, 0.0), ("trip", 1, 1, 1_000.0)],
        beta_m=100.0,
        max_speed_mps=10.0,
        speed_scale_mps=10.0,
    )

    score = result["trip"][1][(1, 1)]
    assert score.network_distance_m == 1_000.0
    assert math.isfinite(score.log_score)
    assert score.log_score == pytest.approx(-4_910.5)


@pytest.mark.parametrize("seconds", [0, -1])
def test_nonpositive_elapsed_time_is_rejected(seconds: int) -> None:
    graph = _graph([(1, "a", "b", 10.0)])
    with pytest.raises(ValueError, match="strictly increasing"):
        _score(
            graph,
            [("trip", 0, _time(), 0.0, 0.0), ("trip", 1, _time(seconds), 0.0, 0.0)],
            [("trip", 0, 1, 0.0), ("trip", 1, 1, 5.0)],
        )


@pytest.mark.parametrize("offset", [-0.01, 10.01, float("nan"), float("inf")])
def test_invalid_projected_offsets_are_rejected(offset: float) -> None:
    graph = _graph([(1, "a", "b", 10.0)])
    with pytest.raises(ValueError):
        _score(
            graph,
            [("trip", 0, _time(), 0.0, 0.0)],
            [("trip", 0, 1, offset)],
        )


@pytest.mark.parametrize(
    "attributes",
    [
        {"arc_id": 1, "from_node_id": "a", "to_node_id": "b"},
        {
            "arc_id": 1,
            "from_node_id": "a",
            "to_node_id": "b",
            "length_m": math.inf,
        },
        {
            "arc_id": 1,
            "from_node_id": "a",
            "to_node_id": "b",
            "length_m": -1.0,
        },
    ],
)
def test_invalid_road_arc_attributes_are_rejected(
    attributes: dict[str, Any],
) -> None:
    graph = FakeMultiDiGraph()
    graph.add_edge("a", "b", key="not-arc-id", **attributes)
    empty_candidates, empty_observations = _frames([], [])

    with pytest.raises(ValueError):
        compute_transitions(
            empty_candidates,
            graph,
            empty_observations,
            beta_m=1.0,
            max_speed_mps=1.0,
            speed_scale_mps=1.0,
        )


def test_zero_length_road_arc_is_accepted() -> None:
    graph = _graph(
        [
            (1, "a", "b", 0.0),
            (2, "b", "c", 10.0),
        ]
    )

    result = _score(
        graph,
        [
            ("trip", 0, _time(), 0.0, 0.0),
            ("trip", 1, _time(5), 0.0, 0.0),
        ],
        [
            ("trip", 0, 1, 0.0),
            ("trip", 1, 2, 5.0),
        ],
    )

    score = result["trip"][1][(1, 2)]

    assert score.network_distance_m == 5.0
    assert math.isfinite(score.log_score)


def test_duplicate_arc_ids_and_invalid_graph_type_are_rejected() -> None:
    graph = _graph([(1, "a", "b", 1.0), (1, "b", "c", 1.0)])
    empty_candidates, empty_observations = _frames([], [])

    with pytest.raises(ValueError, match="duplicate original arc_id"):
        compute_transitions(
            empty_candidates,
            graph,
            empty_observations,
            beta_m=1.0,
            max_speed_mps=1.0,
            speed_scale_mps=1.0,
        )

    with pytest.raises(ValueError, match="must be directed"):
        compute_transitions(
            empty_candidates,
            FakeMultiDiGraph(directed=False),
            empty_observations,
            beta_m=1.0,
            max_speed_mps=1.0,
            speed_scale_mps=1.0,
        )


def test_invalid_gps_coordinates_and_arc_references_are_rejected() -> None:
    graph = _graph([(1, "a", "b", 10.0)])
    candidate_frame, observation_frame = _frames(
        [("trip", 0, _time(), 181.0, 0.0)], [("trip", 0, 1, 1.0)]
    )
    with pytest.raises(ValueError, match="longitude"):
        compute_transitions(
            candidate_frame,
            graph,
            observation_frame,
            beta_m=1.0,
            max_speed_mps=1.0,
            speed_scale_mps=1.0,
        )

    candidate_frame, observation_frame = _frames(
        [("trip", 0, _time(), 0.0, 0.0)], [("trip", 0, 99, 1.0)]
    )
    with pytest.raises(ValueError, match="unknown arc_id"):
        compute_transitions(
            candidate_frame,
            graph,
            observation_frame,
            beta_m=1.0,
            max_speed_mps=1.0,
            speed_scale_mps=1.0,
        )


def test_empty_inputs_return_empty_mapping_and_represent_empty_candidate_steps() -> None:
    graph = _graph([(1, "a", "b", 10.0)])
    candidates, observations = _frames([], [])
    assert _score(graph, [], []) == {}

    candidates, observations = _frames(
        [("trip", 0, _time(), 0.0, 0.0), ("trip", 1, _time(1), 0.0, 0.0)], []
    )
    result = compute_transitions(
        candidates,
        graph,
        observations,
        beta_m=1.0,
        max_speed_mps=1.0,
        speed_scale_mps=1.0,
    )
    assert result == {"trip": {0: {}, 1: {}}}


def test_distance_cache_reuses_pairs_is_bounded_and_graph_bound() -> None:
    graph = _graph(
        [
            (1, "a", "b", 10.0),
            (2, "b", "c", 10.0),
            (3, "c", "d", 10.0),
            (4, "d", "e", 10.0),
        ]
    )
    cache = EndpointDistanceCache(max_entries=2)
    observations = [("trip", 0, _time(), 0.0, 0.0), ("trip", 1, _time(1), 0.0, 0.0)]
    candidates = [("trip", 0, 1, 5.0), ("trip", 1, 4, 2.0)]
    first = _score(graph, observations, candidates, distance_cache=cache)
    cache_size = len(cache)
    search_count = graph.successor_calls
    second = _score(graph, observations, candidates, distance_cache=cache)

    assert first == second
    assert first["trip"][1][(1, 4)].network_distance_m == 27.0
    assert cache_size <= 2
    assert len(cache) <= 2
    assert search_count > 0
    assert graph.successor_calls == search_count
    with pytest.raises(ValueError, match="different graph"):
        _score(_graph([(1, "a", "b", 10.0)]), observations, candidates, distance_cache=cache)


@pytest.mark.parametrize("parameter", ["beta_m", "max_speed_mps", "speed_scale_mps"])
def test_nonpositive_or_nonfinite_parameters_are_rejected(parameter: str) -> None:
    graph = _graph([])
    candidates, observations = _frames([], [])
    parameters: dict[str, float] = {
        "beta_m": 1.0,
        "max_speed_mps": 1.0,
        "speed_scale_mps": 1.0,
    }
    parameters[parameter] = 0.0
    with pytest.raises(ValueError, match="positive"):
        compute_transitions(candidates, graph, observations, **parameters)


def test_candidate_observation_mismatch_is_rejected() -> None:
    graph = _graph([(1, "a", "b", 10.0)])
    candidates, observations = _frames(
        [("trip", 0, _time(), 0.0, 0.0)], [("trip", 0, 1, 2.0)]
    )
    candidates.loc[0, "timestamp"] = _time(1)

    with pytest.raises(ValueError, match="timestamp disagrees"):
        compute_transitions(
            candidates,
            graph,
            observations,
            beta_m=1.0,
            max_speed_mps=1.0,
            speed_scale_mps=1.0,
        )


def test_networkx_multidigraph_parallel_edges_use_minimum_and_keep_arc_ids() -> None:
    graph = _networkx_graph(
        [
            (10, "a", "b", 100.0, "edge-p"),
            (11, "b", "c", 9.0, "parallel-long"),
            (12, "b", "c", 2.0, "parallel-short"),
            (13, "c", "d", 4.0, "edge-middle"),
            (20, "d", "e", 100.0, "candidate-one"),
            (21, "d", "e", 100.0, "candidate-two"),
        ]
    )
    result = _score(
        graph,
        [("trip", 0, _time(), 0.0, 0.0), ("trip", 1, _time(10), 0.0, 0.0)],
        [("trip", 0, 10, 90.0), ("trip", 1, 20, 5.0), ("trip", 1, 21, 6.0)],
    )

    assert result["trip"][1][(10, 20)].network_distance_m == 21.0
    assert result["trip"][1][(10, 21)].network_distance_m == 22.0
    assert set(result["trip"][1]) == {(10, 20), (10, 21)}


def test_networkx_multidigraph_directed_multihop_shortest_path() -> None:
    graph = _networkx_graph(
        [
            (1, "a", "b", 30.0, "candidate-previous"),
            (2, "b", "c", 8.0, "route-1"),
            (3, "c", "d", 7.0, "route-2"),
            (4, "b", "e", 30.0, "long-route-1"),
            (5, "e", "d", 30.0, "long-route-2"),
            (6, "d", "z", 40.0, "candidate-current"),
        ]
    )
    result = _score(
        graph,
        [("trip", 0, _time(), 0.0, 0.0), ("trip", 1, _time(10), 0.0, 0.0)],
        [("trip", 0, 1, 20.0), ("trip", 1, 6, 5.0)],
    )

    assert result["trip"][1][(1, 6)].network_distance_m == 30.0


@pytest.mark.parametrize(
    ("add_reverse_edge", "expected_distance"),
    [(False, None), (True, 15.0)],
)
def test_networkx_multidigraph_rejects_reverse_travel_without_edge(
    add_reverse_edge: bool, expected_distance: float | None
) -> None:
    arcs = [
        (1, "a", "b", 10.0, "prev"),
        (2, "c", "b", 5.0, "wrong-way"),
        (3, "c", "d", 10.0, "current"),
    ]
    if add_reverse_edge:
        arcs.append((4, "b", "c", 5.0, "explicit-reverse"))
    graph = _networkx_graph(arcs)
    result = _score(
        graph,
        [("trip", 0, _time(), 0.0, 0.0), ("trip", 1, _time(5), 0.0, 0.0)],
        [("trip", 0, 1, 5.0), ("trip", 1, 3, 5.0)],
    )

    assert result["trip"][1][(1, 3)].network_distance_m == expected_distance


def test_networkx_multidigraph_same_arc_forward_and_backward() -> None:
    graph = _networkx_graph(
        [
            (7, "a", "b", 100.0, "candidate"),
            (8, "b", "c", 5.0, "loop-out"),
            (9, "c", "a", 7.0, "loop-back"),
        ]
    )
    forward = _score(
        graph,
        [("trip", 0, _time(), 0.0, 0.0), ("trip", 1, _time(5), 0.0, 0.0)],
        [("trip", 0, 7, 30.0), ("trip", 1, 7, 60.0)],
    )
    backward = _score(
        graph,
        [("trip", 0, _time(), 0.0, 0.0), ("trip", 1, _time(5), 0.0, 0.0)],
        [("trip", 0, 7, 80.0), ("trip", 1, 7, 20.0)],
    )

    assert forward["trip"][1][(7, 7)].network_distance_m == 30.0
    assert backward["trip"][1][(7, 7)].network_distance_m == 52.0


def test_networkx_multidigraph_disconnected_arcs_are_unreachable() -> None:
    graph = _networkx_graph(
        [(1, "a", "b", 10.0, "left"), (2, "c", "d", 10.0, "right")]
    )
    result = _score(
        graph,
        [("trip", 0, _time(), 0.0, 0.0), ("trip", 1, _time(10), 0.0, 0.0)],
        [("trip", 0, 1, 5.0), ("trip", 1, 2, 5.0)],
    )

    assert result["trip"][1][(1, 2)] == TransitionScore(None, -math.inf)


class CountingMultiDiGraph(nx.MultiDiGraph):
    def __init__(self) -> None:
        super().__init__()
        self.successor_calls = 0

    def successors(self, node: Any) -> Any:
        self.successor_calls += 1
        return super().successors(node)


def test_networkx_multidigraph_distance_cache_reuses_endpoint_search() -> None:
    graph = CountingMultiDiGraph()
    for arc_id, source, target, length in [
        (1, "a", "b", 10.0),
        (2, "b", "c", 10.0),
        (3, "c", "d", 10.0),
        (4, "d", "e", 10.0),
    ]:
        graph.add_edge(
            source,
            target,
            key=f"key-{arc_id + 100}",
            arc_id=arc_id,
            from_node_id=source,
            to_node_id=target,
            length_m=length,
        )

    cache = EndpointDistanceCache(max_entries=4)
    observations = [("trip", 0, _time(), 0.0, 0.0), ("trip", 1, _time(1), 0.0, 0.0)]
    candidates = [("trip", 0, 1, 5.0), ("trip", 1, 4, 2.0)]
    first = _score(graph, observations, candidates, distance_cache=cache)
    calls_after_first = graph.successor_calls
    second = _score(graph, observations, candidates, distance_cache=cache)

    assert first == second
    assert first["trip"][1][(1, 4)].network_distance_m == 27.0
    assert len(cache) == 1
    assert calls_after_first > 0
    assert graph.successor_calls == calls_after_first


def test_invalid_timestamp_is_rejected() -> None:
    graph = _graph([(1, "a", "b", 10.0)])
    candidates, observations = _frames(
        [("trip", 0, _time(), 0.0, 0.0)], [("trip", 0, 1, 2.0)]
    )
    observations["timestamp"] = observations["timestamp"].astype(object)
    observations.at[0, "timestamp"] = "not-a-timestamp"

    with pytest.raises(ValueError, match="valid timestamp"):
        compute_transitions(
            candidates,
            graph,
            observations,
            beta_m=1.0,
            max_speed_mps=1.0,
            speed_scale_mps=1.0,
        )


@pytest.mark.parametrize("arc_id", [1.5, -1])
def test_invalid_candidate_arc_ids_are_rejected(arc_id: object) -> None:
    graph = _graph([(1, "a", "b", 10.0)])
    candidates, observations = _frames(
        [("trip", 0, _time(), 0.0, 0.0)], [("trip", 0, 1, 2.0)]
    )
    candidates["candidate_arc_id"] = [arc_id]
    expected_error = TypeError if isinstance(arc_id, float) else ValueError

    with pytest.raises(expected_error):
        compute_transitions(
            candidates,
            graph,
            observations,
            beta_m=1.0,
            max_speed_mps=1.0,
            speed_scale_mps=1.0,
        )
