"""FastAPI backend for PathSnap map-matching and route reconstruction."""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from src.api.schemas import (
    HealthResponse,
    ReconstructRequest,
    ReconstructResponse,
    TripSummary,
    TripsListResponse,
)
from src.data.loader import load_trip
from src.models.classifier import get_classifier
from src.services.reconstruction import reconstruct_trip, _GRAPH_CACHE

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("pathsnap.api")

# Stored reconstruction results for GET /results/{trip_id}
_STORED_RESULTS: dict[str, dict[str, Any]] = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup and shutdown lifecycle manager."""
    logger.info("Initializing PathSnap backend service...")

    # Preload classifier model
    classifier = get_classifier()
    if classifier.is_loaded:
        logger.info("Candidate classifier loaded: %s", classifier.model_path)
    else:
        logger.warning("No classifier loaded at startup; will attempt lazy loading.")

    # Pre-warm road network cache for student test trips in background if possible
    yield

    logger.info("Shutting down PathSnap backend service.")


app = FastAPI(
    title="PathSnap API",
    description="Distributed ML & HMM/Viterbi Map-Matching and Route Reconstruction Engine",
    version="1.0.0",
    lifespan=lifespan,
)

# Mandatory CORS configuration for cross-origin LAN two-laptop deployment
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health", response_model=HealthResponse, tags=["System"])
def get_health() -> HealthResponse:
    """Check health, model status, and road graph status."""
    classifier = get_classifier()
    data_dir = Path("data") / "student_test"
    trip_count = len([d for d in data_dir.iterdir() if d.is_dir()]) if data_dir.is_dir() else 0

    return HealthResponse(
        status="healthy",
        model_loaded=classifier.is_loaded,
        model_name=classifier.metadata.get("model_name", "Random Forest Classifier" if classifier.is_loaded else "Heuristic Fallback"),
        road_graph_cached=len(_GRAPH_CACHE) > 0,
        available_trips_count=trip_count,
        version="1.0.0",
    )


@app.get("/trips", response_model=TripsListResponse, tags=["Trips"])
def list_trips(split: str = "student_test") -> TripsListResponse:
    """Return the available evaluator-facing trip options from actual data."""
    data_path = Path("data") / split
    if not data_path.is_dir():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Dataset split folder not found: {data_path}",
        )

    folders = sorted([d.name for d in data_path.iterdir() if d.is_dir()])
    summaries: list[TripSummary] = []

    for trip_id in folders:
        trip_dir = data_path / trip_id
        track_file = trip_dir / f"{trip_id}.track"
        route_file = trip_dir / f"{trip_id}.route"

        total_obs = 0
        if track_file.is_file():
            try:
                # Fast line count for GPS track observations
                with open(track_file, "r", encoding="utf-8") as f:
                    total_obs = sum(1 for _ in f)
            except Exception:
                total_obs = 0

        has_gt = route_file.is_file()
        summaries.append(
            TripSummary(
                trip_id=trip_id,
                split=split,
                total_observations=total_obs,
                has_ground_truth=has_gt,
                label=f"Evaluator Trip {trip_id} ({total_obs} GPS pts)",
            )
        )

    return TripsListResponse(trips=summaries, total_trips=len(summaries))


@app.post("/reconstruct", response_model=ReconstructResponse, tags=["Reconstruction"])
def reconstruct(req: ReconstructRequest) -> ReconstructResponse:
    """Run full HMM + ML Viterbi route reconstruction on a selected trip."""
    trip_id = req.trip_id.strip()
    split = req.split.strip()

    try:
        classifier = get_classifier()
        result = reconstruct_trip(
            trip_id=trip_id,
            split=split,
            data_dir="data",
            classifier=classifier,
        )
        _STORED_RESULTS[f"{split}_{trip_id}"] = result
        return ReconstructResponse(**result)
    except FileNotFoundError as exc:
        logger.error("Trip data missing for %s: %s", trip_id, exc)
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Trip files not found: {exc}",
        ) from exc
    except ValueError as exc:
        logger.error("Reconstruction validation error on trip %s: %s", trip_id, exc)
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Validation failed: {exc}",
        ) from exc
    except Exception as exc:
        logger.exception("Unexpected error reconstructing trip %s", trip_id)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Internal reconstruction error: {str(exc)}",
        ) from exc


@app.get("/results/{trip_id}", response_model=ReconstructResponse, tags=["Reconstruction"])
def get_results(trip_id: str, split: str = "student_test") -> ReconstructResponse:
    """Retrieve cached reconstruction results or run reconstruction if missing."""
    cache_key = f"{split}_{trip_id.zfill(8)}"
    if cache_key in _STORED_RESULTS:
        return ReconstructResponse(**_STORED_RESULTS[cache_key])

    # Run on demand
    return reconstruct(ReconstructRequest(trip_id=trip_id, split=split))


# Mount static frontend directory if exists
frontend_dir = Path("frontend")
if frontend_dir.is_dir() and (frontend_dir / "index.html").is_file():
    app.mount("/", StaticFiles(directory=str(frontend_dir), html=True), name="frontend")
