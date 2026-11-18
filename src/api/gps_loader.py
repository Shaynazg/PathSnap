
import logging
from pathlib import Path

import pandas as pd

from src.api.config import STUDENT_TEST_DIR

logger = logging.getLogger("pathsnap.gps_loader")


def load_gps_track(trip_id: str) -> pd.DataFrame:
    """
    Load GPS observations for a student-testing trip.

    Expected .track format:
    longitude latitude timestamp
    """

    if not trip_id.isdigit() or len(trip_id) != 8:
        raise ValueError("Invalid trip ID")

    track_path = (
        STUDENT_TEST_DIR
        / trip_id
        / f"{trip_id}.track"
    )

    if not track_path.is_file():
        raise FileNotFoundError(
            f"GPS track not found for trip {trip_id}"
        )

    logger.info("Loading GPS track: %s", trip_id)

    track_df = pd.read_csv(
        track_path,
        sep=r"\s+",
        header=None,
        names=[
            "longitude",
            "latitude",
            "timestamp"
        ]
    )

    track_df = track_df.apply(
        pd.to_numeric,
        errors="coerce"
    ).dropna()

    valid_coordinates = (
        track_df["longitude"].between(-180, 180)
        & track_df["latitude"].between(-90, 90)
    )

    track_df = (
        track_df[valid_coordinates]
        .reset_index(drop=True)
    )

    if track_df.empty:
        raise ValueError(
            "GPS track contains no valid observations"
        )

    if not track_df["timestamp"].is_monotonic_increasing:
        raise ValueError(
            "GPS timestamps are not in chronological order"
        )

    logger.info(
        "Loaded %d GPS observations for trip %s",
        len(track_df),
        trip_id
    )

    return track_df
