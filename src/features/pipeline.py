
import json
from pathlib import Path

from src.data.loader import load_trip
from src.data.preprocessing import (
    validate_trip,
    build_road_network,
)
from src.features.candidates import generate_candidates
from src.features.extraction import extract_candidate_features


def process_trip(
    trip_id,
    split="train",
    data_dir="data",
    radius_m=50,
    max_candidates=8,
    save=True,
):
    """
    Run the complete data and feature pipeline for one trip.

    Returns the extracted features, road network and diagnostics.
    """

    trip_id = str(trip_id).zfill(8)

    print(f"\nProcessing trip {trip_id} ({split})...")

    # 1. Load trip files
    trip = load_trip(
        trip_id=trip_id,
        split=split,
        data_dir=data_dir,
    )

    # 2. Validate input data
    print("Validating trip...")
    validation_report = validate_trip(trip)

    if not validation_report["valid"]:
        raise ValueError(
            f"Trip {trip_id} failed validation: "
            f"{validation_report['errors']}"
        )

    # 3. Build directed road network
    print("Building road network...")
    road_graph, road_gdf, projected_crs = build_road_network(trip)

    # 4. Generate GPS-to-road candidates
    print("Generating candidates...")
    candidates, candidate_stats = generate_candidates(
        trip=trip,
        road_gdf=road_gdf,
        radius_m=radius_m,
        max_candidates=max_candidates,
    )

    # 5. Extract candidate features
    print("Extracting features...")
    features = extract_candidate_features(
        candidates=candidates,
        road_gdf=road_gdf,
        track=trip["track"],
    )

    # 6. Verify candidate identity preservation
    identity_columns = [
        "trip_id",
        "obs_idx",
        "candidate_arc_id",
    ]

    if len(features) != len(candidates):
        raise ValueError("Feature extraction changed the candidate row count.")

    if not features[identity_columns].reset_index(drop=True).equals(
        candidates[identity_columns].reset_index(drop=True)
    ):
        raise ValueError("Candidate identities or their order changed.")

    # 7. Prepare diagnostics
    total_observations = len(trip["track"])
    observations_with_candidates = features["obs_idx"].nunique()

    diagnostics = {
        "trip_id": trip_id,
        "split": split,
        "projected_crs": str(projected_crs),
        "total_observations": total_observations,
        "observations_with_candidates": observations_with_candidates,
        "observations_without_candidates": (
            total_observations - observations_with_candidates
        ),
        "total_candidate_rows": len(candidates),
        "average_candidates_per_observation": (
            len(candidates) / total_observations
            if total_observations else 0
        ),
        "road_graph_nodes": road_graph.number_of_nodes(),
        "road_graph_edges": road_graph.number_of_edges(),
        "candidate_generation": candidate_stats,
        "feature_columns": features.columns.tolist(),
        "validation_warnings": validation_report["warnings"],
    }

    # 8. Save processed features and diagnostics
    feature_path = None
    diagnostics_path = None

    if save:
        output_dir = Path(data_dir) / "processed" / split
        output_dir.mkdir(parents=True, exist_ok=True)

        feature_path = output_dir / f"{trip_id}_features.csv"
        diagnostics_path = output_dir / f"{trip_id}_diagnostics.json"

        features.to_csv(feature_path, index=False)

        with open(diagnostics_path, "w", encoding="utf-8") as file:
            json.dump(diagnostics, file, indent=4, default=str)

        print(f"\nFeatures saved to: {feature_path}")
        print(f"Diagnostics saved to: {diagnostics_path}")

    print("\nPIPELINE SUMMARY")
    print("-" * 45)
    for key, value in diagnostics.items():
        if key != "candidate_generation" and key != "feature_columns":
            print(f"{key}: {value}")

    return {
        "features": features,
        "candidates": candidates,
        "road_graph": road_graph,
        "road_gdf": road_gdf,
        "projected_crs": projected_crs,
        "diagnostics": diagnostics,
        "validation_report": validation_report,
        "feature_path": feature_path,
        "diagnostics_path": diagnostics_path,
    }
