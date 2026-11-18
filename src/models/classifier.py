"""Candidate classification and probability estimation interface.

Connects the trained machine-learning model to downstream HMM emission scoring.
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

FEATURE_COLUMNS = [
    "distance_m",
    "heading_diff",
    "speed_mps",
    "bearing",
    "projected_offset_m",
    "arc_length_m",
    "remaining_distance_m",
    "offset_ratio",
    "time_delta_s",
    "gps_displacement_m",
    "candidate_heading_deg",
    "heading_missing",
]

IDENTIFIERS = ["trip_id", "obs_idx", "candidate_arc_id"]

COLUMN_RENAME_MAP = {
    "gps_speed_mps": "speed_mps",
    "gps_heading_deg": "bearing",
    "heading_difference_deg": "heading_diff",
}


def prepare_candidate_features(candidate_df: pd.DataFrame) -> pd.DataFrame:
    """Ensure candidate DataFrame has all required model feature columns.

    Applies renaming if upstream raw names are present and ensures
    appropriate default fallbacks for missing/NaN feature fields.
    """
    df = candidate_df.copy()

    # Check identifiers
    for col in IDENTIFIERS:
        if col not in df.columns:
            raise ValueError(f"Candidate DataFrame missing required identifier: '{col}'")

    # Rename if necessary
    for old_col, new_col in COLUMN_RENAME_MAP.items():
        if old_col in df.columns and new_col not in df.columns:
            df[new_col] = df[old_col]

    # Compute heading_missing if absent
    if "heading_missing" not in df.columns:
        if "heading_diff" in df.columns:
            df["heading_missing"] = df["heading_diff"].isna().astype(int)
        else:
            df["heading_missing"] = 1

    # Fill default values matching feature pipeline
    if "heading_diff" in df.columns:
        df["heading_diff"] = df["heading_diff"].fillna(90.0)
    else:
        df["heading_diff"] = 90.0

    if "speed_mps" in df.columns:
        df["speed_mps"] = df["speed_mps"].fillna(0.0)
    else:
        df["speed_mps"] = 0.0

    if "bearing" in df.columns:
        df["bearing"] = df["bearing"].fillna(0.0)
    else:
        df["bearing"] = 0.0

    if "time_delta_s" in df.columns:
        df["time_delta_s"] = df["time_delta_s"].fillna(1.0)
    else:
        df["time_delta_s"] = 1.0

    if "gps_displacement_m" in df.columns:
        df["gps_displacement_m"] = df["gps_displacement_m"].fillna(0.0)
    else:
        df["gps_displacement_m"] = 0.0

    for col in FEATURE_COLUMNS:
        if col not in df.columns:
            df[col] = 0.0

    return df


class CandidateClassifier:
    """Manages model loading, feature validation, and candidate inference."""

    def __init__(self, model_path: str | Path | None = None) -> None:
        self.model_path: Path | None = None
        self.model: Any = None
        self.metadata: dict[str, Any] = {}

        if model_path is not None:
            self.load(model_path)
        else:
            # Check default path or env var
            env_path = os.environ.get("PATHSNAP_MODEL_PATH")
            default_path = Path(os.environ.get("PATHSNAP_MODEL_PATH", "artifacts/pathsnap_gradient_boosting.joblib"))
            if env_path and Path(env_path).exists():
                self.load(env_path)
            elif default_path.exists():
                self.load(default_path)
            elif Path("artifacts/pathsnap_best_model.joblib").exists():
                self.load(Path("artifacts/pathsnap_best_model.joblib"))

    @property
    def is_loaded(self) -> bool:
        return self.model is not None

    def load(self, model_path: str | Path) -> None:
        """Load fitted sklearn model / pipeline and optional metadata."""
        path = Path(model_path)
        if not path.is_file():
            raise FileNotFoundError(f"Model file not found: {path}")

        logger.info("Loading candidate classifier from %s", path)
        self.model = joblib.load(path)
        self.model_path = path

        meta_path = path.parent / "pathsnap_model_metadata.json"
        if meta_path.is_file():
            try:
                with open(meta_path, "r", encoding="utf-8") as f:
                    self.metadata = json.load(f)
            except Exception as e:
                logger.warning("Could not read metadata from %s: %s", meta_path, e)

    def predict_candidates(self, candidate_df: pd.DataFrame) -> pd.DataFrame:
        """Predict positive-class probabilities for candidate road arcs.

        Returns a DataFrame with columns:
            ['trip_id', 'obs_idx', 'candidate_arc_id', 'probability', 'raw_probability']

        Probabilities are normalized to sum to 1.0 within each (trip_id, obs_idx) group.
        Candidate identities and input row order are strictly preserved.
        """
        if candidate_df.empty:
            return pd.DataFrame(
                columns=IDENTIFIERS + ["probability", "raw_probability"]
            )

        prepared = prepare_candidate_features(candidate_df)

        if not self.is_loaded:
            # Fallback when no model loaded: distance-based heuristic probability
            logger.warning("No model loaded; using normalized inverse distance fallback.")
            raw = 1.0 / (prepared["distance_m"].clip(lower=1.0))
        else:
            X = prepared[FEATURE_COLUMNS]
            try:
                if hasattr(self.model, "predict_proba"):
                    proba = self.model.predict_proba(X)
                    if hasattr(self.model, "named_steps") and "model" in self.model.named_steps:
                        classes = list(self.model.named_steps["model"].classes_)
                    elif hasattr(self.model, "classes_"):
                        classes = list(self.model.classes_)
                    else:
                        classes = [0, 1]

                    if 1 in classes:
                        pos_idx = classes.index(1)
                        raw = proba[:, pos_idx]
                    else:
                        raw = proba[:, -1]
                elif hasattr(self.model, "decision_function"):
                    scores = self.model.decision_function(X)
                    raw = 1.0 / (1.0 + np.exp(-scores))
                else:
                    preds = self.model.predict(X)
                    raw = preds.astype(float)
            except Exception as exc:
                logger.error("Model prediction failed: %s; falling back to inverse distance", exc)
                raw = 1.0 / (prepared["distance_m"].clip(lower=1.0))

        result = candidate_df[IDENTIFIERS].copy()
        result["raw_probability"] = np.asarray(raw, dtype=float)
        # Ensure finite and non-negative
        result["raw_probability"] = np.nan_to_num(result["raw_probability"], nan=1e-6)
        result["raw_probability"] = np.clip(result["raw_probability"], 1e-6, 1.0)

        # Normalize within (trip_id, obs_idx)
        def _normalize_group(series: pd.Series) -> pd.Series:
            total = series.sum()
            if total > 0:
                return series / total
            n = len(series)
            return pd.Series(1.0 / n if n > 0 else 0.0, index=series.index)

        result["probability"] = (
            result.groupby(["trip_id", "obs_idx"])["raw_probability"]
            .transform(_normalize_group)
        )

        return result


# Module-level convenience singleton
_default_classifier = CandidateClassifier()


def get_classifier(model_path: str | Path | None = None) -> CandidateClassifier:
    """Retrieve or initialize the candidate classifier."""
    global _default_classifier
    if model_path is not None:
        _default_classifier = CandidateClassifier(model_path)
    return _default_classifier


def predict_candidates(
    candidate_df: pd.DataFrame,
    model_path: str | Path | None = None,
) -> pd.DataFrame:
    """Direct inference function matching expected specification."""
    classifier = get_classifier(model_path)
    return classifier.predict_candidates(candidate_df)
