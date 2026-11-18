
import numpy as np
import pandas as pd
from pyproj import Transformer


def extract_candidate_features(candidates, road_gdf, track):
    """
    Extract geometric and movement-related features for road candidates.

    Candidate identity remains:
    (trip_id, obs_idx, candidate_arc_id).
    """

    required_candidate_columns = {
        "trip_id",
        "obs_idx",
        "timestamp",
        "longitude",
        "latitude",
        "candidate_arc_id",
        "distance_m",
        "projected_offset_m",
    }

    required_road_columns = {
        "arc_id",
        "length_m",
        "geometry",
    }

    required_track_columns = {
        "obs_idx",
        "longitude",
        "latitude",
        "timestamp",
    }

    if not required_candidate_columns.issubset(candidates.columns):
        raise ValueError("Candidate DataFrame is missing required columns.")

    if not required_road_columns.issubset(road_gdf.columns):
        raise ValueError("Road GeoDataFrame is missing required columns.")

    if not required_track_columns.issubset(track.columns):
        raise ValueError("GPS track is missing required columns.")

    if road_gdf["arc_id"].duplicated().any():
        raise ValueError("Road GeoDataFrame contains duplicate arc IDs.")

    if track["obs_idx"].duplicated().any():
        raise ValueError("GPS track contains duplicate observation indices.")

    # ---------------------------------------------------------
    # 1. Attach road-arc length and direction
    # ---------------------------------------------------------

    arc_information = road_gdf[
        ["arc_id", "length_m", "geometry"]
    ].copy()

    arc_information = arc_information.rename(
        columns={
            "arc_id": "candidate_arc_id",
            "length_m": "arc_length_m",
        }
    )

    def calculate_arc_heading(geometry):
        """
        Calculate bearing clockwise from north,
        following the directed arc geometry.
        """

        if geometry is None or geometry.is_empty:
            return np.nan

        coordinates = list(geometry.coords)

        if len(coordinates) < 2:
            return np.nan

        start_x, start_y = coordinates[0]
        end_x, end_y = coordinates[-1]

        dx = end_x - start_x
        dy = end_y - start_y

        if np.hypot(dx, dy) <= 1e-9:
            return np.nan

        return float(np.degrees(np.arctan2(dx, dy)) % 360)

    arc_information["candidate_heading_deg"] = (
        arc_information["geometry"].apply(calculate_arc_heading)
    )

    arc_information = arc_information.drop(columns="geometry")

    features = candidates.merge(
        arc_information,
        on="candidate_arc_id",
        how="left",
        validate="many_to_one",
        sort=False,
        indicator=True,
    )

    if (features["_merge"] != "both").any():
        raise ValueError("Some candidate arc IDs are missing from road network.")

    features = features.drop(columns="_merge")

    # ---------------------------------------------------------
    # 2. Calculate movement features for each GPS observation
    # ---------------------------------------------------------

    observations = track[
        ["obs_idx", "longitude", "latitude", "timestamp"]
    ].copy()

    observations = observations.sort_values("obs_idx").reset_index(drop=True)

    transformer = Transformer.from_crs(
        "EPSG:4326",
        road_gdf.crs,
        always_xy=True,
    )

    x, y = transformer.transform(
        observations["longitude"].to_numpy(),
        observations["latitude"].to_numpy(),
    )

    observations["gps_x"] = x
    observations["gps_y"] = y

    # Previous GPS observation.
    previous_x = observations["gps_x"].shift(1)
    previous_y = observations["gps_y"].shift(1)
    previous_timestamp = observations["timestamp"].shift(1)

    dx = observations["gps_x"] - previous_x
    dy = observations["gps_y"] - previous_y

    observations["time_delta_s"] = (
        observations["timestamp"] - previous_timestamp
    )

    observations["gps_displacement_m"] = np.hypot(dx, dy)

    # Calculate speed only when elapsed time is positive.
    valid_time = observations["time_delta_s"] > 0

    observations["gps_speed_mps"] = np.where(
        valid_time,
        observations["gps_displacement_m"] / observations["time_delta_s"],
        np.nan,
    )

    # Heading is undefined when movement is effectively zero.
    valid_heading = (
        valid_time
        & (observations["gps_displacement_m"] > 1e-6)
    )

    observations["gps_heading_deg"] = np.where(
        valid_heading,
        np.degrees(np.arctan2(dx, dy)) % 360,
        np.nan,
    )

    movement_columns = [
        "obs_idx",
        "time_delta_s",
        "gps_displacement_m",
        "gps_speed_mps",
        "gps_heading_deg",
    ]

    features = features.merge(
        observations[movement_columns],
        on="obs_idx",
        how="left",
        validate="many_to_one",
        sort=False,
    )

    #if features["time_delta_s"].isna().any():
        #raise ValueError("Some candidate observations have no matching GPS movement data.")
    # Verify that every candidate observation exists in the GPS track.
# Do not check time_delta_s for NaN here:
# the first GPS observation legitimately has no previous observation.

    if not features["obs_idx"].isin(observations["obs_idx"]).all():
        raise ValueError(
        "Some candidate observations have no matching GPS movement data."
    )
    # ---------------------------------------------------------
    # 3. Calculate angular difference between GPS and road
    # ---------------------------------------------------------

    raw_difference = (
        features["gps_heading_deg"]
        - features["candidate_heading_deg"]
    ).abs()

    features["heading_difference_deg"] = np.minimum(
        raw_difference,
        360 - raw_difference,
    )

    # ---------------------------------------------------------
    # 4. Calculate remaining geometric features
    # ---------------------------------------------------------

    numeric_columns = [
        "distance_m",
        "projected_offset_m",
        "arc_length_m",
    ]

    for column in numeric_columns:
        features[column] = pd.to_numeric(
            features[column],
            errors="raise",
        )

    if not np.isfinite(features[numeric_columns].to_numpy()).all():
        raise ValueError("Candidate geometric features contain non-finite values.")

    if (features[numeric_columns] < 0).any().any():
        raise ValueError("Candidate geometric measurements cannot be negative.")

    tolerance = 1e-6

    if (
        features["projected_offset_m"]
        > features["arc_length_m"] + tolerance
    ).any():
        raise ValueError("A projected offset exceeds its corresponding arc length.")

    features["remaining_distance_m"] = (
        features["arc_length_m"] - features["projected_offset_m"]
    ).clip(lower=0)

    features["is_zero_length_arc"] = (
        features["arc_length_m"] == 0
    )

    features["offset_ratio"] = np.where(
        features["arc_length_m"] > 0,
        features["projected_offset_m"] / features["arc_length_m"],
        0.0,
    )

    # ---------------------------------------------------------
    # 5. Return features in a predictable order
    # ---------------------------------------------------------

    feature_columns = [
        "trip_id",
        "obs_idx",
        "timestamp",
        "longitude",
        "latitude",
        "candidate_arc_id",
        "distance_m",
        "projected_offset_m",
        "arc_length_m",
        "remaining_distance_m",
        "offset_ratio",
        "is_zero_length_arc",
        "time_delta_s",
        "gps_displacement_m",
        "gps_speed_mps",
        "gps_heading_deg",
        "candidate_heading_deg",
        "heading_difference_deg",
    ]

    return features[feature_columns]
