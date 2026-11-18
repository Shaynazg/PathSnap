
from pathlib import Path

import pandas as pd
import numpy as np

from src.data.loader import load_trip


def inspect_trip(trip_id):
    # Load ground-truth route
    trip = load_trip(trip_id, split="train")

    route_ids = set(
        trip["route"]["arc_id"].astype(int).tolist()
    )

    # Load previously generated candidate features
    feature_path = (
        Path("data/processed/train")
        / f"{trip_id}_features.csv"
    )

    if not feature_path.exists():
        raise FileNotFoundError(
            f"Feature file not found: {feature_path}"
        )

    features = pd.read_csv(
        feature_path,
        dtype={"trip_id": str},
    )

    features["candidate_arc_id"] = (
        features["candidate_arc_id"].astype(int)
    )

    # Mark candidates whose arc IDs occur in the ground-truth route.
    features["is_route_arc"] = (
        features["candidate_arc_id"].isin(route_ids)
    )

    # Group candidates by GPS observation.
    observation_stats = (
        features.groupby("obs_idx")
        .agg(
            candidate_count=("candidate_arc_id", "size"),
            route_candidate_count=("is_route_arc", "sum"),
            nearest_candidate_distance=("distance_m", "min"),
        )
    )

    observation_stats["has_route_candidate"] = (
        observation_stats["route_candidate_count"] > 0
    )

    # Distance to the closest candidate belonging to the route.
    route_candidates = features[
        features["is_route_arc"]
    ]

    route_candidate_distances = (
        route_candidates.groupby("obs_idx")["distance_m"]
        .min()
    )

    observation_stats["nearest_route_candidate_distance"] = (
        route_candidate_distances
    )

    total_observations = len(trip["track"])

    observations_with_route_candidate = int(
        observation_stats["has_route_candidate"].sum()
    )

    observations_without_route_candidate = (
        total_observations - observations_with_route_candidate
    )

    route_candidate_count = int(
        features["is_route_arc"].sum()
    )

    return {
        "trip_id": trip_id,
        "gps_observations": total_observations,
        "route_arc_count": len(trip["route"]),
        "unique_route_arcs": len(route_ids),
        "observations_with_route_candidate": (
            observations_with_route_candidate
        ),
        "observations_without_route_candidate": (
            observations_without_route_candidate
        ),
        "route_candidate_coverage": (
            observations_with_route_candidate / total_observations
            if total_observations else np.nan
        ),
        "route_candidate_rows": route_candidate_count,
        "average_route_candidates_per_observation": (
            route_candidate_count / total_observations
            if total_observations else 0
        ),
        "median_nearest_route_candidate_distance_m": (
            route_candidate_distances.median()
            if not route_candidate_distances.empty else np.nan
        ),
        "max_nearest_route_candidate_distance_m": (
            route_candidate_distances.max()
            if not route_candidate_distances.empty else np.nan
        ),
    }


def main():
    train_dir = Path("data/train")

    trip_folders = sorted(
        folder
        for folder in train_dir.iterdir()
        if folder.is_dir()
        and (folder / f"{folder.name}.track").is_file()
    )

    results = []
    failures = []

    for index, folder in enumerate(trip_folders, start=1):
        trip_id = folder.name

        print(
            f"[{index}/{len(trip_folders)}] "
            f"Checking route candidates for {trip_id}..."
        )

        try:
            results.append(inspect_trip(trip_id))

        except Exception as error:
            failures.append({
                "trip_id": trip_id,
                "error": str(error),
            })
            print(f"FAILED: {error}")

    result_df = pd.DataFrame(results)

    output_path = (
        Path("data/processed/train")
        / "route_candidate_overlap.csv"
    )

    result_df.to_csv(output_path, index=False)

    print("\n" + "=" * 65)
    print("ROUTE-CANDIDATE OVERLAP SUMMARY")
    print("=" * 65)

    if not result_df.empty:
        print("\nSUMMARY STATISTICS")
        print(
            result_df[[
                "route_candidate_coverage",
                "average_route_candidates_per_observation",
                "median_nearest_route_candidate_distance_m",
            ]].describe().to_string()
        )

        print("\nFIRST 20 TRIPS")
        print(
            result_df[[
                "trip_id",
                "route_candidate_coverage",
                "observations_without_route_candidate",
                "median_nearest_route_candidate_distance_m",
            ]].head(20).to_string(index=False)
        )

        print(
            "\nTrips with zero route-candidate coverage:",
            (result_df["route_candidate_coverage"] == 0).sum(),
        )

        print(
            "Trips with incomplete route-candidate coverage:",
            (result_df["route_candidate_coverage"] < 1).sum(),
        )

    print(f"\nSaved to: {output_path}")

    if failures:
        print("\nFAILED TRIPS")
        for failure in failures:
            print(failure)

    print(
        f"\nSuccessfully inspected: "
        f"{len(results)}/{len(trip_folders)}"
    )


if __name__ == "__main__":
    main()
