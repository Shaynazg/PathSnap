"""Directed, distance-based HMM transition scoring.

The module consumes candidate and observation tables produced upstream. It
does not construct candidates or depend on the decoder's lightweight graph
class. The road graph must be an unchanged directed MultiDiGraph-like object
for the lifetime of any :class:`EndpointDistanceCache` used with it.
"""

from __future__ import annotations

import heapq
import math
from collections import OrderedDict
from collections.abc import Hashable, Iterable, Mapping
from dataclasses import dataclass
from datetime import date, datetime
from numbers import Integral, Real
from typing import Protocol, TypeAlias

import pandas as pd

NodeId: TypeAlias = Hashable
TripId: TypeAlias = Hashable
CandidatePair: TypeAlias = tuple[int, int]

CANDIDATE_COLUMNS = frozenset(
    {
        "trip_id",
        "obs_idx",
        "timestamp",
        "longitude",
        "latitude",
        "candidate_arc_id",
        "distance_m",
        "projected_offset_m",
    }
)
OBSERVATION_COLUMNS = frozenset(
    {"trip_id", "obs_idx", "timestamp", "longitude", "latitude"}
)
EARTH_RADIUS_M = 6_371_008.8
OFFSET_TOLERANCE_M = 1e-6
MAX_FINITE_SCORE_PENALTY = float.fromhex("0x1.fffffffffffffp+1022")


class DirectedMultiDiGraph(Protocol):
    """Structural subset of NetworkX MultiDiGraph used by this module."""

    def is_directed(self) -> bool: ...

    def is_multigraph(self) -> bool: ...

    def edges(self, keys: bool = ..., data: bool = ...) -> Iterable[object]: ...

    def successors(self, node: NodeId) -> Iterable[NodeId]: ...

    def get_edge_data(self, u: NodeId, v: NodeId) -> object: ...


@dataclass(frozen=True)
class ArcInfo:
    """Directed endpoint and length metadata for one original road arc."""

    from_node_id: NodeId
    to_node_id: NodeId
    length_m: float


@dataclass(frozen=True)
class TransitionScore:
    """Score and directed network distance for one consecutive candidate pair.

    ``network_distance_m`` is ``None`` and ``log_score`` is negative infinity
    when no directed route exists. Reachable transitions always receive a
    finite score, including transitions that are highly implausible.
    """

    network_distance_m: float | None
    log_score: float


TransitionScores: TypeAlias = dict[
    TripId,
    dict[int, dict[CandidatePair, TransitionScore]],
]


class EndpointDistanceCache:
    """Bounded LRU cache for directed endpoint-pair distances.

    Cache keys include both endpoint node IDs. The first graph used binds this
    cache to that graph object; using the cache with another graph raises
    ``ValueError``. The graph must not be mutated while the cache is in use.
    Unreachable pairs are cached as well as finite distances.
    """

    def __init__(self, max_entries: int = 100_000) -> None:
        if isinstance(max_entries, bool) or not isinstance(max_entries, Integral):
            raise TypeError("max_entries must be an integer")
        if max_entries <= 0:
            raise ValueError("max_entries must be positive")
        self.max_entries = int(max_entries)
        self._distances: OrderedDict[tuple[NodeId, NodeId], float | None] = (
            OrderedDict()
        )
        self._graph: object | None = None

    def _bind(self, graph: object) -> None:
        if self._graph is None:
            self._graph = graph
        elif self._graph is not graph:
            raise ValueError(
                "distance cache is bound to a different graph; call clear() first"
            )

    def _lookup(self, source: NodeId, target: NodeId) -> tuple[bool, float | None]:
        key = (source, target)
        if key not in self._distances:
            return False, None
        self._distances.move_to_end(key)
        return True, self._distances[key]

    def _store(
        self, source: NodeId, target: NodeId, distance_m: float | None
    ) -> None:
        key = (source, target)
        self._distances[key] = distance_m
        self._distances.move_to_end(key)
        while len(self._distances) > self.max_entries:
            self._distances.popitem(last=False)

    def clear(self) -> None:
        """Drop cached distances and release the graph binding."""
        self._distances.clear()
        self._graph = None

    def __len__(self) -> int:
        """Return the current number of cached endpoint pairs."""
        return len(self._distances)


def _validate_frame(
    frame: object, required_columns: frozenset[str], name: str
) -> pd.DataFrame:
    if not isinstance(frame, pd.DataFrame):
        raise TypeError(f"{name} must be a pandas DataFrame")
    if not frame.columns.is_unique:
        raise ValueError(f"{name} must not contain duplicate column names")
    missing = required_columns.difference(frame.columns)
    if missing:
        raise ValueError(
            f"{name} is missing required columns: {', '.join(sorted(missing))}"
        )
    return frame


def _identifier(value: object, name: str) -> Hashable:
    if not pd.api.types.is_scalar(value) or bool(pd.isna(value)):
        raise ValueError(f"{name} must be a non-missing scalar")
    try:
        hash(value)
    except TypeError as exc:
        raise TypeError(f"{name} must be hashable") from exc
    return value


def _nonnegative_index(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise TypeError(f"{name} must be an integer")
    index = int(value)
    if index < 0:
        raise ValueError(f"{name} must be non-negative")
    return index


def _finite_real(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a real number")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _timestamp(value: object, name: str) -> pd.Timestamp:
    if not pd.api.types.is_scalar(value) or bool(pd.isna(value)):
        raise ValueError(f"{name} must be a valid timestamp")
    if isinstance(value, Real) or isinstance(value, bool):
        raise TypeError(f"{name} must be a datetime-like value, not a number")
    if not isinstance(value, (str, date, datetime, pd.Timestamp)):
        # Let pandas parse other datetime-like scalars such as numpy.datetime64.
        if not hasattr(value, "dtype"):
            raise TypeError(f"{name} must be a datetime-like value")
    try:
        parsed = pd.Timestamp(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} is not a valid timestamp: {value!r}") from exc
    if pd.isna(parsed):
        raise ValueError(f"{name} must be a valid timestamp")
    return parsed


def _trip_sort_key(trip_id: TripId) -> tuple[str, str, str]:
    trip_type = type(trip_id)
    return (trip_type.__module__, trip_type.__qualname__, repr(trip_id))


def _validate_graph(
    road_graph: object,
) -> dict[int, ArcInfo]:
    required_methods = ("is_directed", "is_multigraph", "edges", "successors", "get_edge_data")
    if any(not callable(getattr(road_graph, method, None)) for method in required_methods):
        raise TypeError("road_graph must provide the directed MultiDiGraph interface")
    if not road_graph.is_directed():  # type: ignore[attr-defined]
        raise ValueError("road_graph must be directed")
    if not road_graph.is_multigraph():  # type: ignore[attr-defined]
        raise ValueError("road_graph must preserve parallel edges as a multigraph")

    try:
        edges = road_graph.edges(keys=True, data=True)  # type: ignore[attr-defined]
    except TypeError as exc:
        raise TypeError("road_graph.edges must support keys=True and data=True") from exc

    arc_index: dict[int, ArcInfo] = {}
    for edge in edges:
        try:
            graph_source, graph_target, _edge_key, attributes = edge
        except (TypeError, ValueError) as exc:
            raise ValueError("road_graph edge iteration returned a malformed edge") from exc
        if not isinstance(attributes, Mapping):
            raise TypeError("each road edge must have a mapping of attributes")
        for attribute in ("arc_id", "from_node_id", "to_node_id", "length_m"):
            if attribute not in attributes:
                raise ValueError(f"road edge is missing required attribute {attribute!r}")

        arc_id = _nonnegative_index(attributes["arc_id"], "edge arc_id")
        if arc_id in arc_index:
            raise ValueError(f"duplicate original arc_id in road graph: {arc_id}")
        source = _identifier(attributes["from_node_id"], "edge from_node_id")
        target = _identifier(attributes["to_node_id"], "edge to_node_id")
        if source != graph_source or target != graph_target:
            raise ValueError(
                f"arc {arc_id} endpoint attributes do not match its directed graph edge"
            )
        length_m = _finite_real(attributes["length_m"], f"arc {arc_id} length_m")
        if length_m <= 0.0:
            raise ValueError(f"arc {arc_id} length_m must be positive")
        arc_index[arc_id] = ArcInfo(source, target, length_m)
    return arc_index


def _parallel_edge_weight(road_graph: DirectedMultiDiGraph, source: NodeId, target: NodeId) -> float:
    edge_data = road_graph.get_edge_data(source, target)
    if not isinstance(edge_data, Mapping) or not edge_data:
        raise ValueError("road_graph successor index and edge data are inconsistent")
    weights: list[float] = []
    for attributes in edge_data.values():
        if not isinstance(attributes, Mapping) or "length_m" not in attributes:
            raise ValueError("road edge is missing a valid length_m attribute")
        weight = _finite_real(attributes["length_m"], "road edge length_m")
        if weight <= 0.0:
            raise ValueError("road edge length_m must be positive")
        weights.append(weight)
    return min(weights)


def _fill_source_distances(
    road_graph: DirectedMultiDiGraph,
    source: NodeId,
    targets: set[NodeId],
    cache: EndpointDistanceCache,
) -> dict[NodeId, float | None]:
    resolved: dict[NodeId, float | None] = {}
    remaining: set[NodeId] = set()
    for target in targets:
        found, distance = cache._lookup(source, target)
        if found:
            resolved[target] = distance
        else:
            remaining.add(target)
    if not remaining:
        return resolved

    distances: dict[NodeId, float] = {source: 0.0}
    sequence = 0
    frontier: list[tuple[float, int, NodeId]] = [(0.0, sequence, source)]
    settled: set[NodeId] = set()

    while frontier and remaining:
        current_distance, _order, node = heapq.heappop(frontier)
        if node in settled:
            continue
        settled.add(node)
        if node in remaining:
            cache._store(source, node, current_distance)
            resolved[node] = current_distance
            remaining.remove(node)

        for neighbor in road_graph.successors(node):
            if neighbor in settled:
                continue
            edge_weight = _parallel_edge_weight(road_graph, node, neighbor)
            next_distance = current_distance + edge_weight
            if not math.isfinite(next_distance):
                raise ValueError("directed path distance overflowed to a non-finite value")
            if next_distance < distances.get(neighbor, math.inf):
                distances[neighbor] = next_distance
                sequence += 1
                heapq.heappush(frontier, (next_distance, sequence, neighbor))

    for target in remaining:
        cache._store(source, target, None)
        resolved[target] = None
    return resolved


def _haversine_m(
    longitude_a: float,
    latitude_a: float,
    longitude_b: float,
    latitude_b: float,
) -> float:
    lon_a, lat_a, lon_b, lat_b = map(
        math.radians, (longitude_a, latitude_a, longitude_b, latitude_b)
    )
    delta_lon = lon_b - lon_a
    delta_lat = lat_b - lat_a
    hav = (
        math.sin(delta_lat / 2.0) ** 2
        + math.cos(lat_a) * math.cos(lat_b) * math.sin(delta_lon / 2.0) ** 2
    )
    central_angle = 2.0 * math.asin(math.sqrt(min(1.0, max(0.0, hav))))
    return EARTH_RADIUS_M * central_angle


def _transition_log_score(
    network_distance_m: float,
    gps_distance_m: float,
    elapsed_seconds: float,
    beta_m: float,
    max_speed_mps: float,
    speed_scale_mps: float,
) -> float:
    error = abs(network_distance_m - gps_distance_m)
    distance_penalty = error / beta_m
    if not math.isfinite(distance_penalty):
        distance_penalty = MAX_FINITE_SCORE_PENALTY

    allowed_distance = max_speed_mps * elapsed_seconds
    if network_distance_m <= allowed_distance:
        speed_penalty = 0.0
    else:
        excess_m = network_distance_m - allowed_distance
        denominator = speed_scale_mps * elapsed_seconds
        z = excess_m / denominator if denominator > 0.0 else math.inf
        if not math.isfinite(z) or z >= math.sqrt(2.0 * MAX_FINITE_SCORE_PENALTY):
            speed_penalty = MAX_FINITE_SCORE_PENALTY
        else:
            speed_penalty = 0.5 * z * z

    total_penalty = min(
        MAX_FINITE_SCORE_PENALTY, distance_penalty + speed_penalty
    )
    return -total_penalty


def compute_transitions(
    candidates: pd.DataFrame,
    road_graph: DirectedMultiDiGraph,
    observations: pd.DataFrame,
    *,
    beta_m: float,
    max_speed_mps: float,
    speed_scale_mps: float,
    distance_cache: EndpointDistanceCache | None = None,
) -> TransitionScores:
    """Compute directed distance and time-aware log scores between candidates.

    Results have the form ``trip_id -> current_obs_idx ->
    (previous_arc_id, current_arc_id) -> TransitionScore``. Trips, observation
    indices, and candidate arc pairs are emitted deterministically. Every
    observation from ``observations`` is represented. An observation without
    candidates has an empty pair mapping, preventing the decoder from
    accidentally bridging across that observation. The first observation of
    each trip also has an empty pair mapping because it has no predecessor.

    Candidate arcs are looked up by their original ``arc_id`` edge attribute;
    NetworkX edge keys are ignored. Intermediate shortest paths follow only
    directed edges, using the minimum length among parallel edges between a
    node pair. A same-arc forward movement uses the projected-offset
    difference. A backward same-arc movement uses a directed loop from that
    arc's end node back to its start node, and is unreachable if no such loop
    exists.

    The reachable transition score is
    ``-abs(network_distance_m - gps_distance_m) / beta_m`` minus a soft speed
    penalty ``0.5 * z**2``, where ``z`` is excess distance above
    ``max_speed_mps * dt`` divided by ``speed_scale_mps * dt``. Scores are log
    potentials and are not normalized. Haversine distance uses a mean Earth
    radius of :data:`EARTH_RADIUS_M`. Large GPS gaps are scored using the
    same formula; no route is invented or bridged.

    Args:
        candidates: Candidate table with the agreed candidate schema.
        road_graph: Unchanged directed MultiDiGraph with required edge attrs.
        observations: Full observation table, including observations with no
            candidate arcs.
        beta_m: Positive finite distance-error scale in metres.
        max_speed_mps: Positive finite soft speed limit in metres per second.
        speed_scale_mps: Positive finite speed-penalty scale in metres per
            second.
        distance_cache: Optional bounded cache reusable across calls for this
            same, unchanged graph. Defaults to a 100,000-pair cache.

    Raises:
        TypeError: For malformed input types or nonnumeric values.
        ValueError: For invalid schemas, graph data, IDs, timestamps,
            coordinates, offsets, parameters, or elapsed times.
    """
    candidates = _validate_frame(candidates, CANDIDATE_COLUMNS, "candidates")
    observations = _validate_frame(observations, OBSERVATION_COLUMNS, "observations")
    arc_index = _validate_graph(road_graph)

    beta_m = _finite_real(beta_m, "beta_m")
    max_speed_mps = _finite_real(max_speed_mps, "max_speed_mps")
    speed_scale_mps = _finite_real(speed_scale_mps, "speed_scale_mps")
    if beta_m <= 0.0:
        raise ValueError("beta_m must be positive")
    if max_speed_mps <= 0.0:
        raise ValueError("max_speed_mps must be positive")
    if speed_scale_mps <= 0.0:
        raise ValueError("speed_scale_mps must be positive")

    if distance_cache is None:
        distance_cache = EndpointDistanceCache()
    elif not isinstance(distance_cache, EndpointDistanceCache):
        raise TypeError("distance_cache must be an EndpointDistanceCache or None")
    distance_cache._bind(road_graph)

    observation_groups: dict[TripId, dict[int, tuple[pd.Timestamp, float, float]]] = {}
    obs_trip_ids = observations["trip_id"].array
    obs_indices = observations["obs_idx"].array
    obs_timestamps = observations["timestamp"].array
    obs_longitudes = observations["longitude"].array
    obs_latitudes = observations["latitude"].array

    for trip_value, obs_value, time_value, lon_value, lat_value in zip(
        obs_trip_ids,
        obs_indices,
        obs_timestamps,
        obs_longitudes,
        obs_latitudes,
    ):
        trip_id = _identifier(trip_value, "observations trip_id")
        obs_idx = _nonnegative_index(obs_value, "observations obs_idx")
        timestamp = _timestamp(time_value, f"timestamp for ({trip_id!r}, {obs_idx})")
        longitude = _finite_real(lon_value, "longitude")
        latitude = _finite_real(lat_value, "latitude")
        if not -180.0 <= longitude <= 180.0:
            raise ValueError("longitude must be in [-180, 180]")
        if not -90.0 <= latitude <= 90.0:
            raise ValueError("latitude must be in [-90, 90]")
        trip_observations = observation_groups.setdefault(trip_id, {})
        if obs_idx in trip_observations:
            raise ValueError(f"duplicate observation ({trip_id!r}, {obs_idx})")
        trip_observations[obs_idx] = (timestamp, longitude, latitude)

    ordered_observations: dict[
        TripId, list[tuple[int, tuple[pd.Timestamp, float, float]]]
    ] = {}
    for trip_id, by_index in observation_groups.items():
        ordered = sorted(by_index.items())
        for (previous_idx, previous), (current_idx, current) in zip(
            ordered, ordered[1:]
        ):
            try:
                elapsed = (current[0] - previous[0]).total_seconds()
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"timestamps for trip {trip_id!r} must use comparable time zones"
                ) from exc
            if not math.isfinite(elapsed) or elapsed <= 0.0:
                raise ValueError(
                    "timestamps must be strictly increasing by obs_idx "
                    f"within trip {trip_id!r} ({previous_idx} to {current_idx})"
                )
        ordered_observations[trip_id] = ordered

    candidate_groups: dict[TripId, dict[int, dict[int, float]]] = {}
    candidate_trip_ids = candidates["trip_id"].array
    candidate_obs_indices = candidates["obs_idx"].array
    candidate_timestamps = candidates["timestamp"].array
    candidate_longitudes = candidates["longitude"].array
    candidate_latitudes = candidates["latitude"].array
    candidate_arc_ids = candidates["candidate_arc_id"].array
    candidate_distances = candidates["distance_m"].array
    candidate_offsets = candidates["projected_offset_m"].array

    for trip_value, obs_value, time_value, lon_value, lat_value, arc_value, dist_value, offset_value in zip(
        candidate_trip_ids,
        candidate_obs_indices,
        candidate_timestamps,
        candidate_longitudes,
        candidate_latitudes,
        candidate_arc_ids,
        candidate_distances,
        candidate_offsets,
    ):
        trip_id = _identifier(trip_value, "candidates trip_id")
        obs_idx = _nonnegative_index(obs_value, "candidates obs_idx")
        arc_id = _nonnegative_index(arc_value, "candidate_arc_id")
        if trip_id not in observation_groups or obs_idx not in observation_groups[trip_id]:
            raise ValueError(
                f"candidate references unknown observation ({trip_id!r}, {obs_idx})"
            )
        if arc_id not in arc_index:
            raise ValueError(f"candidate references unknown arc_id {arc_id}")

        obs_time, obs_lon, obs_lat = observation_groups[trip_id][obs_idx]
        candidate_time = _timestamp(time_value, f"candidate timestamp for arc {arc_id}")
        if candidate_time != obs_time:
            raise ValueError(
                f"candidate timestamp disagrees with observation ({trip_id!r}, {obs_idx})"
            )
        longitude = _finite_real(lon_value, "candidate longitude")
        latitude = _finite_real(lat_value, "candidate latitude")
        if not -180.0 <= longitude <= 180.0:
            raise ValueError("candidate longitude must be in [-180, 180]")
        if not -90.0 <= latitude <= 90.0:
            raise ValueError("candidate latitude must be in [-90, 90]")
        if not math.isclose(longitude, obs_lon, rel_tol=0.0, abs_tol=1e-9) or not math.isclose(
            latitude, obs_lat, rel_tol=0.0, abs_tol=1e-9
        ):
            raise ValueError(
                f"candidate coordinates disagree with observation ({trip_id!r}, {obs_idx})"
            )

        distance_m = _finite_real(dist_value, "candidate distance_m")
        if distance_m < 0.0:
            raise ValueError("candidate distance_m must be non-negative")
        offset_m = _finite_real(offset_value, "projected_offset_m")
        arc = arc_index[arc_id]
        if offset_m < -OFFSET_TOLERANCE_M or offset_m > arc.length_m + OFFSET_TOLERANCE_M:
            raise ValueError(
                f"projected_offset_m for arc {arc_id} must be within [0, {arc.length_m}]"
            )
        offset_m = min(arc.length_m, max(0.0, offset_m))

        by_observation = candidate_groups.setdefault(trip_id, {}).setdefault(
            obs_idx, {}
        )
        if arc_id in by_observation:
            raise ValueError(
                f"duplicate candidate arc {arc_id} for ({trip_id!r}, {obs_idx})"
            )
        by_observation[arc_id] = offset_m

    results: TransitionScores = {}
    for trip_id in sorted(ordered_observations, key=_trip_sort_key):
        ordered = ordered_observations[trip_id]
        trip_candidates = candidate_groups.get(trip_id, {})
        trip_results: dict[int, dict[CandidatePair, TransitionScore]] = {
            obs_idx: {} for obs_idx, _data in ordered
        }

        for (previous_idx, previous_obs), (current_idx, current_obs) in zip(
            ordered, ordered[1:]
        ):
            previous_candidates = trip_candidates.get(previous_idx, {})
            current_candidates = trip_candidates.get(current_idx, {})
            if not previous_candidates or not current_candidates:
                # Keep this observation's empty mapping: never bridge a gap.
                continue

            elapsed_seconds = (current_obs[0] - previous_obs[0]).total_seconds()
            gps_distance_m = _haversine_m(
                previous_obs[1], previous_obs[2], current_obs[1], current_obs[2]
            )
            source_nodes = {
                arc_index[arc_id].to_node_id for arc_id in previous_candidates
            }
            target_nodes = {
                arc_index[arc_id].from_node_id for arc_id in current_candidates
            }
            # One directed single-source search serves all current candidate
            # starts reachable from a previous candidate's endpoint.
            middle_distances: dict[tuple[NodeId, NodeId], float | None] = {}
            for source in source_nodes:
                source_distances = _fill_source_distances(
                    road_graph, source, target_nodes, distance_cache
                )
                middle_distances.update(
                    ((source, target), distance)
                    for target, distance in source_distances.items()
                )

            for previous_arc_id in sorted(previous_candidates):
                previous_arc = arc_index[previous_arc_id]
                previous_offset = previous_candidates[previous_arc_id]
                for current_arc_id in sorted(current_candidates):
                    current_arc = arc_index[current_arc_id]
                    current_offset = current_candidates[current_arc_id]

                    if previous_arc_id == current_arc_id:
                        if current_offset >= previous_offset:
                            network_distance = current_offset - previous_offset
                        else:
                            loop_distance = middle_distances[
                                previous_arc.to_node_id,
                                previous_arc.from_node_id,
                            ]
                            network_distance = (
                                None
                                if loop_distance is None
                                else previous_arc.length_m
                                - previous_offset
                                + loop_distance
                                + current_offset
                            )
                    else:
                        middle_distance = middle_distances[
                            previous_arc.to_node_id,
                            current_arc.from_node_id,
                        ]
                        network_distance = (
                            None
                            if middle_distance is None
                            else previous_arc.length_m
                            - previous_offset
                            + middle_distance
                            + current_offset
                        )

                    if network_distance is None:
                        score = TransitionScore(None, -math.inf)
                    else:
                        if not math.isfinite(network_distance) or network_distance < 0.0:
                            raise ValueError("computed network distance is invalid")
                        score = TransitionScore(
                            network_distance,
                            _transition_log_score(
                                network_distance,
                                gps_distance_m,
                                elapsed_seconds,
                                beta_m,
                                max_speed_mps,
                                speed_scale_mps,
                            ),
                        )
                    trip_results[current_idx][
                        (previous_arc_id, current_arc_id)
                    ] = score

        results[trip_id] = trip_results
    return results

