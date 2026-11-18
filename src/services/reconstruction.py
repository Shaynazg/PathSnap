"""End-to-end route reconstruction service.

Orchestrates candidate retrieval, ML probability estimation, HMM emissions,
directed transition scoring, Viterbi decoding, and route geometry reconstruction.
"""

from __future__ import annotations

import logging
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import networkx as nx
import numpy as np
import pandas as pd

from src.data.loader import load_trip
from src.data.preprocessing import build_road_network, validate_trip
from src.decoder.emissions import compute_emissions
from src.decoder.transitions import EndpointDistanceCache, compute_transitions
from src.decoder.viterbi import DecodedSegment, decode_viterbi
from src.features.candidates import generate_candidates
from src.features.extraction import extract_candidate_features
from src.models.classifier import CandidateClassifier, get_classifier, prepare_candidate_features

logger = logging.getLogger(__name__)

# In-memory caches to avoid expensive repeated disk/graph processing
_GRAPH_CACHE: dict[str, dict[str, Any]] = {}
_DISTANCE_CACHE: dict[str, EndpointDistanceCache] = {}
_CANDIDATE_CACHE: dict[str, pd.DataFrame] = {}


def get_cached_or_build_road_network(
    trip: dict[str, pd.DataFrame], trip_id: str
) -> tuple[nx.MultiDiGraph, pd.DataFrame, dict[int, tuple[float, float]]]:
    """Retrieve or construct the road network, GeoDataFrame, and WGS84 node coordinates."""
    if trip_id in _GRAPH_CACHE:
        cached = _GRAPH_CACHE[trip_id]
        return cached["graph"], cached["gdf"], cached["node_coords"]

    road_graph, road_gdf, _ = build_road_network(trip)

    # Build fast lookup for node (lat, lon)
    node_coords: dict[int, tuple[float, float]] = {}
    for node_id, data in road_graph.nodes(data=True):
        node_coords[node_id] = (float(data["latitude"]), float(data["longitude"]))

    _GRAPH_CACHE[trip_id] = {
        "graph": road_graph,
        "gdf": road_gdf,
        "node_coords": node_coords,
    }
    return road_graph, road_gdf, node_coords


def get_or_generate_candidates(
    trip: dict[str, pd.DataFrame],
    trip_id: str,
    split: str,
    road_gdf: pd.DataFrame,
    data_dir: str = "data",
) -> pd.DataFrame:
    """Retrieve pre-extracted candidates or dynamically generate them."""
    cache_key = f"{split}_{trip_id}"
    if cache_key in _CANDIDATE_CACHE:
        return _CANDIDATE_CACHE[cache_key].copy()

    # 1. Check if processed candidate CSV exists
    processed_candidates_path = Path(data_dir) / "processed" / f"{split}_candidates.csv"
    if processed_candidates_path.is_file():
        try:
            # Read chunk or filter by trip_id
            df_all = pd.read_csv(processed_candidates_path, dtype={"trip_id": str})
            trip_str = str(trip_id).zfill(8)
            trip_cands = df_all[df_all["trip_id"] == trip_str].copy()
            if not trip_cands.empty:
                # Merge track coordinates and timestamps if missing
                track = trip["track"]
                if "longitude" not in trip_cands.columns:
                    trip_cands = trip_cands.merge(
                        track[["obs_idx", "longitude", "latitude", "timestamp"]],
                        on="obs_idx",
                        how="left",
                        suffixes=("", "_track"),
                    )
                _CANDIDATE_CACHE[cache_key] = trip_cands
                return trip_cands.copy()
        except Exception as e:
            logger.warning("Could not read from preprocessed candidate file: %s", e)

    # 2. Dynamic generation fallback
    logger.info("Generating candidates dynamically for trip %s...", trip_id)
    candidates, _ = generate_candidates(
        trip=trip,
        road_gdf=road_gdf,
        radius_m=50,
        max_candidates=8,
    )
    features = extract_candidate_features(
        candidates=candidates,
        road_gdf=road_gdf,
        track=trip["track"],
    )
    prepared = prepare_candidate_features(features)
    _CANDIDATE_CACHE[cache_key] = prepared
    return prepared.copy()


def reconstruct_trip(
    trip_id: str,
    split: str = "student_test",
    data_dir: str = "data",
    classifier: CandidateClassifier | None = None,
) -> dict[str, Any]:
    """Execute complete end-to-end map-matching and route reconstruction."""
    t_start = time.perf_counter()
    trip_id = str(trip_id).zfill(8)

    timing: dict[str, float] = {}

    # 1. Load trip data
    t0 = time.perf_counter()
    trip = load_trip(trip_id=trip_id, split=split, data_dir=data_dir)
    validation = validate_trip(trip)
    if not validation["valid"]:
        raise ValueError(f"Trip {trip_id} failed validation: {validation['errors']}")
    timing["data_loading_ms"] = (time.perf_counter() - t0) * 1000.0

    # 2. Road network
    t0 = time.perf_counter()
    road_graph, road_gdf, node_coords = get_cached_or_build_road_network(trip, trip_id)
    timing["road_network_ms"] = (time.perf_counter() - t0) * 1000.0

    # 3. Candidates and features
    t0 = time.perf_counter()
    candidates = get_or_generate_candidates(
        trip=trip,
        trip_id=trip_id,
        split=split,
        road_gdf=road_gdf,
        data_dir=data_dir,
    )
    timing["candidate_retrieval_ms"] = (time.perf_counter() - t0) * 1000.0

    # 4. Model inference
    t0 = time.perf_counter()
    if classifier is None:
        classifier = get_classifier()

    predictions = classifier.predict_candidates(candidates)
    timing["model_inference_ms"] = (time.perf_counter() - t0) * 1000.0

    # 5. HMM Emissions & Transitions
    t0 = time.perf_counter()
    emissions = compute_emissions(predictions)

    # Prepare observations DataFrame
    observations = trip["track"][
        ["trip_id", "obs_idx", "timestamp", "longitude", "latitude"]
    ].copy() if "trip_id" in trip["track"].columns else trip["track"][
        ["obs_idx", "timestamp", "longitude", "latitude"]
    ].assign(trip_id=trip_id)

    # Convert timestamps if numeric to datetime
    if np.issubdtype(observations["timestamp"].dtype, np.number):
        observations["timestamp"] = pd.to_datetime(
            observations["timestamp"], unit="s", origin="unix"
        )

    # Ensure candidates has timestamp, longitude, latitude
    cands_for_transitions = candidates.copy()
    if np.issubdtype(cands_for_transitions["timestamp"].dtype, np.number):
        cands_for_transitions["timestamp"] = pd.to_datetime(
            cands_for_transitions["timestamp"], unit="s", origin="unix"
        )
    if "longitude" not in cands_for_transitions.columns:
        cands_for_transitions = cands_for_transitions.merge(
            observations[["obs_idx", "longitude", "latitude"]],
            on="obs_idx",
            how="left",
        )

    if trip_id not in _DISTANCE_CACHE:
        _DISTANCE_CACHE[trip_id] = EndpointDistanceCache()
    dist_cache = _DISTANCE_CACHE[trip_id]

    transitions = compute_transitions(
        cands_for_transitions,
        road_graph,
        observations,
        distance_cache=dist_cache,
        beta_m=100.0,
        max_speed_mps=50.0,
        speed_scale_mps=15.0,
    )
    timing["hmm_scoring_ms"] = (time.perf_counter() - t0) * 1000.0

    # 6. Viterbi decoding
    t0 = time.perf_counter()
    viterbi_results = decode_viterbi(emissions, transitions)
    segments = viterbi_results.get(trip_id, ())
    timing["viterbi_decoding_ms"] = (time.perf_counter() - t0) * 1000.0

    # 7. Route and Geometry Reconstruction
    t0 = time.perf_counter()

    # Pre-index arc geometries and endpoints
    arc_lookup: dict[int, dict[str, Any]] = {}
    for _, _, _, data in road_graph.edges(keys=True, data=True):
        arc_id = int(data["arc_id"])
        u = int(data["from_node_id"])
        v = int(data["to_node_id"])
        p1 = node_coords.get(u, (0.0, 0.0))
        p2 = node_coords.get(v, (0.0, 0.0))
        arc_lookup[arc_id] = {
            "arc_id": arc_id,
            "from_node_id": u,
            "to_node_id": v,
            "length_m": float(data.get("length_m", 0.0)),
            "coords": [p1, p2],
        }

    decoded_arcs: list[int] = []
    decoded_segments_info: list[dict[str, Any]] = []
    route_polyline_coords: list[list[float]] = []

    for seg in segments:
        seg_dict = {
            "start_obs_idx": seg.start_obs_idx,
            "end_obs_idx": seg.end_obs_idx,
            "has_path": seg.has_path,
            "total_log_score": None if np.isneginf(seg.total_log_score) else float(seg.total_log_score),
            "candidate_arc_ids": list(seg.candidate_arc_ids) if seg.candidate_arc_ids else [],
        }
        decoded_segments_info.append(seg_dict)

        if seg.has_path and seg.candidate_arc_ids:
            for arc_id in seg.candidate_arc_ids:
                decoded_arcs.append(int(arc_id))
                if arc_id in arc_lookup:
                    c = arc_lookup[arc_id]["coords"]
                    route_polyline_coords.append(c[0])
                    route_polyline_coords.append(c[1])

    # Remove consecutive duplicate points in route polyline
    cleaned_polyline: list[list[float]] = []
    for pt in route_polyline_coords:
        if not cleaned_polyline or cleaned_polyline[-1] != pt:
            cleaned_polyline.append(pt)

    # Observations for UI map
    obs_points: list[dict[str, Any]] = []
    for r in trip["track"].itertuples(index=False):
        obs_points.append({
            "obs_idx": int(r.obs_idx),
            "lat": float(r.latitude),
            "lon": float(r.longitude),
            "timestamp": float(r.timestamp) if np.issubdtype(type(r.timestamp), np.number) else str(r.timestamp),
        })

    # Top-1 Classifier baseline (without HMM)
    top1_classifier_arcs: list[int] = []
    if not predictions.empty:
        top1_preds = predictions.sort_values(
            ["obs_idx", "probability"], ascending=[True, False]
        ).groupby("obs_idx").first().reset_index()
        top1_classifier_arcs = [int(a) for a in top1_preds["candidate_arc_id"]]

    timing["geometry_ms"] = (time.perf_counter() - t0) * 1000.0
    timing["total_ms"] = (time.perf_counter() - t_start) * 1000.0

    # Evaluation comparison if ground truth route exists
    evaluation: dict[str, Any] | None = None
    if "route" in trip and not trip["route"].empty:
        gt_arcs = [int(a) for a in trip["route"]["arc_id"].tolist()]
        gt_arc_set = set(gt_arcs)

        # Arc-level overlap metrics
        viterbi_arc_set = set(decoded_arcs)
        top1_arc_set = set(top1_classifier_arcs)

        def _calc_set_metrics(pred_set: set[int], true_set: set[int]) -> dict[str, float]:
            tp = len(pred_set & true_set)
            prec = tp / len(pred_set) if pred_set else 0.0
            rec = tp / len(true_set) if true_set else 0.0
            f1 = (2 * prec * rec) / (prec + rec) if (prec + rec) > 0 else 0.0
            return {
                "precision": float(prec),
                "recall": float(rec),
                "f1_score": float(f1),
                "exact_unique_overlap_count": int(tp),
                "total_predicted_unique": int(len(pred_set)),
                "total_ground_truth_unique": int(len(true_set)),
            }

        viterbi_eval = _calc_set_metrics(viterbi_arc_set, gt_arc_set)
        top1_eval = _calc_set_metrics(top1_arc_set, gt_arc_set)

        evaluation = {
            "ground_truth_route_length": len(gt_arcs),
            "viterbi_reconstruction": viterbi_eval,
            "classifier_top1_baseline": top1_eval,
            "hmm_f1_improvement": float(viterbi_eval["f1_score"] - top1_eval["f1_score"]),
        }

    return {
        "trip_id": trip_id,
        "split": split,
        "model_name": classifier.metadata.get("model_name", "Random Forest Classifier"),
        "total_observations": len(trip["track"]),
        "total_candidates": len(candidates),
        "total_decoded_arcs": len(decoded_arcs),
        "total_segments": len(segments),
        "no_path_segments": sum(1 for s in segments if not s.has_path),
        "has_disconnected_segments": any(not s.has_path for s in segments),
        "decoded_arc_sequence": decoded_arcs,
        "decoded_segments": decoded_segments_info,
        "route_polyline": cleaned_polyline,
        "gps_points": obs_points,
        "top1_classifier_arc_sequence": top1_classifier_arcs,
        "evaluation": evaluation,
        "timing_ms": timing,
    }
