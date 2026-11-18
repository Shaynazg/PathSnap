
import logging
import pickle
from pathlib import Path

import joblib

from src.api.config import (
    MODEL_PATH,
    PREPROCESSOR_PATH,
    DECODER_PATH,
)

logger = logging.getLogger("pathsnap.model_loader")


def load_artifact(path, artifact_name):
    """
    Loads a trusted ML artifact from its configured path.

    Supported formats:
    - .joblib
    - .pkl
    - .pickle
    """

    if not path:
        logger.warning(
            "%s path has not been configured",
            artifact_name
        )
        return None

    artifact_path = Path(path)

    if not artifact_path.is_file():
        logger.error(
            "%s artifact not found: %s",
            artifact_name,
            artifact_path
        )
        raise FileNotFoundError(
            f"{artifact_name} artifact not found: {artifact_path}"
        )

    suffix = artifact_path.suffix.lower()

    logger.info("Loading %s from %s", artifact_name, artifact_path)

    if suffix == ".joblib":
        artifact = joblib.load(artifact_path)

    elif suffix in (".pkl", ".pickle"):
        with open(artifact_path, "rb") as file:
            artifact = pickle.load(file)

    else:
        raise ValueError(
            f"Unsupported artifact format: {suffix}"
        )

    logger.info("%s loaded successfully", artifact_name)

    return artifact


def load_configured_artifacts():
    """
    Loads the configured model, preprocessor
    and HMM decoder.

    Missing configurations remain None until
    teammates provide the actual artifacts.
    """

    model = load_artifact(
        MODEL_PATH,
        "ML model"
    )

    preprocessor = load_artifact(
        PREPROCESSOR_PATH,
        "Preprocessing pipeline"
    )

    decoder = load_artifact(
        DECODER_PATH,
        "HMM decoder"
    )

    return {
        "model": model,
        "preprocessor": preprocessor,
        "decoder": decoder
    }


def get_artifact_status():
    """
    Checks artifact availability without
    loading the actual files.
    """

    artifacts = {
        "model": MODEL_PATH,
        "preprocessor": PREPROCESSOR_PATH,
        "decoder": DECODER_PATH
    }

    status = {}

    for name, path in artifacts.items():

        if not path:
            status[name] = "not_configured"

        elif Path(path).is_file():
            status[name] = "available"

        else:
            status[name] = "file_not_found"

    return status
