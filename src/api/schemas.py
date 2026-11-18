"""Pydantic schemas for FastAPI backend."""

from __future__ import annotations

from typing import Any
from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    status: str = Field(..., example="healthy")
    model_loaded: bool = Field(..., example=True)
    model_name: str = Field(..., example="Random Forest (Balanced, Tuned)")
    road_graph_cached: bool = Field(..., example=True)
    available_trips_count: int = Field(..., example=10)
    version: str = Field(default="1.0.0")


class TripSummary(BaseModel):
    trip_id: str = Field(..., example="00000000")
    split: str = Field(default="student_test")
    total_observations: int = Field(..., example=500)
    has_ground_truth: bool = Field(default=True)
    label: str = Field(..., example="Evaluator Trip 00000000")


class TripsListResponse(BaseModel):
    trips: list[TripSummary]
    total_trips: int = Field(..., example=10)


class ReconstructRequest(BaseModel):
    trip_id: str = Field(..., example="00000000", description="Evaluator trip ID")
    split: str = Field(default="student_test", description="Dataset split (student_test or train)")


class DecodedSegmentInfo(BaseModel):
    start_obs_idx: int
    end_obs_idx: int
    has_path: bool
    total_log_score: float | None = None
    candidate_arc_ids: list[int] = Field(default_factory=list)


class GpsPoint(BaseModel):
    obs_idx: int
    lat: float
    lon: float
    timestamp: float | str


class MetricSet(BaseModel):
    precision: float
    recall: float
    f1_score: float
    exact_unique_overlap_count: int
    total_predicted_unique: int
    total_ground_truth_unique: int


class EvaluationResult(BaseModel):
    ground_truth_route_length: int
    viterbi_reconstruction: MetricSet
    classifier_top1_baseline: MetricSet
    hmm_f1_improvement: float


class TimingInfo(BaseModel):
    data_loading_ms: float
    road_network_ms: float
    candidate_retrieval_ms: float
    model_inference_ms: float
    hmm_scoring_ms: float
    viterbi_decoding_ms: float
    geometry_ms: float
    total_ms: float


class ReconstructResponse(BaseModel):
    trip_id: str
    split: str
    model_name: str
    total_observations: int
    total_candidates: int
    total_decoded_arcs: int
    total_segments: int
    no_path_segments: int
    has_disconnected_segments: bool
    decoded_arc_sequence: list[int]
    decoded_segments: list[DecodedSegmentInfo]
    route_polyline: list[list[float]] = Field(description="List of [latitude, longitude] pairs for map")
    gps_points: list[GpsPoint]
    top1_classifier_arc_sequence: list[int]
    evaluation: EvaluationResult | None = None
    timing_ms: TimingInfo
