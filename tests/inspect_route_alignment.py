
from pathlib import Path

import pandas as pd
import numpy as np
from pyproj import Transformer

from src.data.loader import load_trip
from src.data.preprocessing import (
    validate_trip,
    build_road_network,
)


def inspect_trip(trip_id):
    trip = load_trip(trip_id, split="train")

    # 1. Validate trip
    report = validate_trip(trip)

    if not report["valid"]:
        raise ValueError(f"Invalid trip: {report['errors']}")

    # 2. Build road network
    _, road_gdf, projected_crs = build_road_network(trip)

    # 3. Calculate total GPS trajectory distance in metres
    transformer = Transformer.from_crs(
        "EPSG:4326",
        projected_crs,
        always_xy=True,
    )

    gps_x, gps_y = transformer.transform(
        trip["track"]["longitude"].to_numpy(),
        trip["track"]["latitude"].to_numpy(),
    )

    gps_coordinates = np.column_stack([gps_x, gps_y])

    gps_step_distances = np.linalg.norm(
        np.diff(gps_coordinates, axis=0),
        axis=1,
    )

    gps_total_distance = float(gps_step_distances.sum())

    # 4. Calculate route distance, preserving repeated arc occurrences
    arc_lengths = road_gdf.set_index("arc_id")["length_m"]

    route_ids = trip["route"]["arc_id"].astype(int)

    route_lengths = arc_lengths.reindex(route_ids.to_numpy())

    missing_route_arcs = int(route_lengths.isna().sum())
    route_total_distance = float(route_lengths.sum())

    # 5. Check route continuity using original arc IDs
    arcs_by_id = trip["arcs"].set_index("arc_id")

    route_arc_rows = arcs_by_id.reindex(route_ids.to_numpy())

    from_nodes = route_arc_rows["from_node_id"].to_numpy()
    to_nodes = route_arc_rows["to_node_id"].to_numpy()

    if len(route_ids) > 1:
        connected = to_nodes[:-1] == from_nodes[1:]

        valid_transitions = (
            pd.notna(to_nodes[:-1])
            & pd.notna(from_nodes[1:])
        )

        discontinuity_count = int(
            (valid_transitions & ~connected).sum()
            + (~valid_transitions).sum()
        )
    else:
        discontinuity_count = 0

    route_continuous = discontinuity_count == 0

    # 6. Return diagnostics
    return {
        "trip_id": trip_id,
        "gps_observations": len(trip["track"]),
        "route_arc_count": len(route_ids),
        "unique_route_arcs": route_ids.nunique(),
        "missing_route_arcs": missing_route_arcs,
        "gps_total_distance_m": gps_total_distance,
        "route_total_distance_m": route_total_distance,
        "route_to_gps_distance_ratio": (
            route_total_distance / gps_total_distance
            if gps_total_distance > 0 else np.nan
        ),
        "route_discontinuities": discontinuity_count,
        "route_continuous": route_continuous,
        "gps_duration_s": (
            float(
                trip["track"]["timestamp"].iloc[-1]
                - trip["track"]["timestamp"].iloc[0]
            )
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

        print(f"[{index}/{len(trip_folders)}] Inspecting {trip_id}...")

        try:
            results.append(inspect_trip(trip_id))

        except Exception as error:
            failures.append({
                "trip_id": trip_id,
                "error": str(error),
            })

            print(f"FAILED: {error}")

    # 7. Save results
    output_dir = Path("data/processed/train")
    output_dir.mkdir(parents=True, exist_ok=True)

    result_df = pd.DataFrame(results)

    output_path = output_dir / "route_alignment_diagnostics.csv"
    result_df.to_csv(output_path, index=False)

    # 8. Print summary
    print("\n" + "=" * 60)
    print("ROUTE ALIGNMENT DIAGNOSTICS")
    print("=" * 60)

    if not result_df.empty:
        print("\nSUMMARY STATISTICS")

        print(
            result_df[[
                "gps_observations",
                "route_arc_count",
                "gps_total_distance_m",
                "route_total_distance_m",
                "route_to_gps_distance_ratio",
                "route_discontinuities",
            ]].describe().to_string()
        )

        print("\nROUTE CONTINUITY")
        print(
            result_df["route_continuous"].value_counts(
                dropna=False
            )
        )

        print("\nROUTE DISTANCE RATIOS — FIRST 15 TRIPS")

        print(
            result_df[[
                "trip_id",
                "gps_total_distance_m",
                "route_total_distance_m",
                "route_to_gps_distance_ratio",
                "route_discontinuities",
            ]].head(15).to_string(index=False)
        )

        print(
            "\nTrips with route discontinuities:",
            (result_df["route_discontinuities"] > 0).sum(),
        )

        print(
            "Trips with missing route arcs:",
            (result_df["missing_route_arcs"] > 0).sum(),
        )

    print(f"\nDiagnostics saved to: {output_path}")

    if failures:
        print("\nFAILED TRIPS:")

        for failure in failures:
            print(failure)

    print(f"\nSuccessfully inspected: {len(results)}/{len(trip_folders)}")


if __name__ == "__main__":
    main()
