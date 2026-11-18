<<<<<<< HEAD
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
=======

import logging

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from src.api.config import (
    API_TITLE,
    API_DESCRIPTION,
    API_VERSION,
    STUDENT_TEST_DIR,
)

from src.api.model_loader import get_artifact_status
from src.api.prediction_service import PredictionService


# ==========================================
# 1. LOGGING CONFIGURATION
# ==========================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s"
)

logger = logging.getLogger("pathsnap.api")


# ==========================================
# 2. FASTAPI APPLICATION
# ==========================================

app = FastAPI(
    title=API_TITLE,
    description=API_DESCRIPTION,
    version=API_VERSION
)

# Initialize prediction service
prediction_service = PredictionService()


# ==========================================
# 3. PREDICTION REQUEST VALIDATION
# ==========================================

class PredictionRequest(BaseModel):
    trip_id: str = Field(
        pattern=r"^\d{8}$",
        description="Eight-digit student-testing trip ID"
    )


# ==========================================
# 4. HEALTH ENDPOINT
# ==========================================

@app.get("/health")
def health_check():

    logger.info("Health check requested")

    return {
        "status": "healthy",
        "message": "PathSnap API is running"
    }


# ==========================================
# 5. TRIP DISCOVERY ENDPOINTS
# ==========================================

@app.get("/trips")
@app.get("/trips/")
def get_trips():

    logger.info("Fetching available student-testing trips")

    if not STUDENT_TEST_DIR.exists():

        logger.error("Student testing directory not found")

        raise HTTPException(
            status_code=500,
            detail="Student testing directory not found"
        )

    trips = sorted(
        folder.name
        for folder in STUDENT_TEST_DIR.iterdir()
        if folder.is_dir()
        and (folder / f"{folder.name}.track").exists()
    )

    logger.info(
        "Discovered %d student-testing trips",
        len(trips)
    )

    return {
        "total_trips": len(trips),
        "trips": trips
    }


# ==========================================
# 6. PREDICTION ENDPOINT
# ==========================================

@app.post("/predict/")
def predict_route(request: PredictionRequest):

    trip_id = request.trip_id

    logger.info(
        "Prediction requested for trip %s",
        trip_id
    )

    try:

        # Load GPS and execute prediction service
        result = prediction_service.predict_trip(trip_id)

        logger.info(
            "Prediction successfully completed for trip %s",
            trip_id
        )

        return {
            "status": "success",
            **result
        }

    except FileNotFoundError as error:

        logger.warning(
            "GPS track not found: %s",
            error
        )

        raise HTTPException(
            status_code=404,
            detail=str(error)
        )

    except ValueError as error:

        logger.warning(
            "Invalid prediction input: %s",
            error
        )

        raise HTTPException(
            status_code=400,
            detail=str(error)
        )

    except RuntimeError as error:

        logger.warning(
            "Prediction pipeline unavailable: %s",
            error
        )

        raise HTTPException(
            status_code=501,
            detail=str(error)
        )

    except Exception:

        logger.exception(
            "Unexpected prediction failure for trip %s",
            trip_id
        )

        raise HTTPException(
            status_code=500,
            detail="Internal prediction error"
        )


# ==========================================
# 7. ARTIFACT STATUS ENDPOINT
# ==========================================

@app.get("/artifacts/status")
def artifact_status():

    status = get_artifact_status()

    all_available = all(
        value == "available"
        for value in status.values()
    )

    logger.info(
        "Artifact availability checked: %s",
        status
    )

    return {
        "artifacts_ready": all_available,
        "artifacts": status
    }


# ==========================================
# 8. PREDICTION PIPELINE STATUS
# ==========================================

@app.get("/pipeline/status")
def pipeline_status():

    components = prediction_service.check_readiness()

    ready = all(components.values())

    logger.info(
        "Prediction pipeline readiness checked: %s",
        components
    )

    return {
        "pipeline_ready": ready,
        "components": components
    }
>>>>>>> feature/backend-integration
