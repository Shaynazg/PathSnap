
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
