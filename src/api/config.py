
import os
from pathlib import Path

# ==========================================
# PROJECT CONFIGURATION
# ==========================================

# Locate the project root automatically
PROJECT_ROOT = Path(__file__).resolve().parents[2]

# ==========================================
# DATASET PATHS
# ==========================================

DATA_DIR = Path(
    os.getenv(
        "PATHSNAP_DATA_DIR",
        str(PROJECT_ROOT / "data")
    )
)

STUDENT_TEST_DIR = DATA_DIR / "student_test"
TRAIN_DIR = DATA_DIR / "train"

# ==========================================
# ARTIFACT DIRECTORIES
# ==========================================

ARTIFACTS_DIR = Path(
    os.getenv(
        "PATHSNAP_ARTIFACTS_DIR",
        str(PROJECT_ROOT / "artifacts")
    )
)

MODEL_DIR = ARTIFACTS_DIR / "model"
CONFIG_DIR = ARTIFACTS_DIR / "config"

# ==========================================
# MODEL ARTIFACT PATHS
# ==========================================

# These remain optional until teammates provide
# their actual artifact filenames.

MODEL_PATH = os.getenv("PATHSNAP_MODEL_PATH")

PREPROCESSOR_PATH = os.getenv(
    "PATHSNAP_PREPROCESSOR_PATH"
)

DECODER_PATH = os.getenv(
    "PATHSNAP_DECODER_PATH"
)

# ==========================================
# API CONFIGURATION
# ==========================================

API_TITLE = "PathSnap API"

API_DESCRIPTION = (
    "GPS Map Matching and Route Reconstruction"
)

API_VERSION = "1.0.0"
