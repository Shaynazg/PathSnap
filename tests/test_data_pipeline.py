
from src.data.loader import load_trip

from src.data.preprocessing import (
    validate_trip,
    print_validation_report,
    build_road_network,
    analyze_road_graph,
    check_route_continuity,
    shortest_directed_arc_path,
)

from src.features.candidates import generate_candidates
from src.features.extraction import extract_candidate_features


def main():
    trip_id = "00000001"

    # 1. Load trip
    print(f"\nLoading trip {trip_id}...")
    trip = load_trip(trip_id)

    # 2. Validate trip
    print("\nValidating trip...")
    report = validate_trip(trip)
    print_validation_report(report)

    if not report["valid"]:
        raise SystemExit("\nTrip validation failed.")

    # 3. Construct road network
    print("\nBuilding road network...")
    road_graph, road_gdf, projected_crs = build_road_network(trip)

    # 4. Display graph statistics
    print("\nROAD NETWORK SUMMARY")
    print("-" * 45)

    print("Projected CRS:", projected_crs)
    print("Graph type:", type(road_graph).__name__)
    print("Number of graph nodes:", road_graph.number_of_nodes())
    print("Number of graph edges:", road_graph.number_of_edges())

    print("\nROAD GEOMETRY SUMMARY")
    print("-" * 45)

    print("GeoDataFrame rows:", len(road_gdf))
    print("GeoDataFrame CRS:", road_gdf.crs)
    print("Columns:", list(road_gdf.columns))

    print("\nFirst five road arcs:")
    print(road_gdf.head())

    print("\nArc length statistics (metres):")
    print(road_gdf["length_m"].describe())

    # 5. Verify original arc IDs are preserved
    original_arc_ids = set(trip["arcs"]["arc_id"])

    graph_arc_ids = {
        data["arc_id"]
        for _, _, _, data in road_graph.edges(
            keys=True, data=True
        )
    }

    print("\nARC ID PRESERVATION")
    print("-" * 45)
    print("Original arc IDs:", len(original_arc_ids))
    print("Graph arc IDs:", len(graph_arc_ids))
    print("All arc IDs preserved:", original_arc_ids == graph_arc_ids)

    # 6. Check geometry and length
    print("\nROAD GEOMETRY VALIDATION")

    print(
        "All geometries present:",
        road_gdf.geometry.notna().all()
    )

    print(
        "All lengths finite:",
        road_gdf["length_m"].map(
            lambda x: x >= 0 and x < float("inf")
        ).all()
    )

    print("\nRoad-network construction completed.")

    # 7. Analyze graph connectivity
    print("\nGRAPH CONNECTIVITY ANALYSIS")
    print("-" * 45)

    graph_stats = analyze_road_graph(road_graph)

    for key, value in graph_stats.items():
        print(f"{key}: {value}")

    # 8. Check ground-truth route continuity
    print("\nROUTE CONTINUITY CHECK")
    print("-" * 45)

    route_stats = check_route_continuity(trip)

    for key, value in route_stats.items():
        print(f"{key}: {value}")

    # 9. Test directed shortest-path calculation
    print("\nSHORTEST DIRECTED PATH TEST")
    print("-" * 45)

    first_route_arc_id = int(trip["route"].iloc[0]["arc_id"])

    first_arc = trip["arcs"].loc[
        trip["arcs"]["arc_id"] == first_route_arc_id
    ].iloc[0]

    source = int(first_arc["from_node_id"])
    target = int(first_arc["to_node_id"])

    path_result = shortest_directed_arc_path(
        road_graph, source, target
    )

    print("Source node:", source)
    print("Target node:", target)
    print("Reachable:", path_result["reachable"])
    print("Selected arc IDs:", path_result["arc_ids"])
    print("Shortest distance (m):", path_result["distance_m"])

    # 10. Generate GPS-to-road candidates
    print("\nGENERATING ROAD CANDIDATES")
    print("-" * 45)

    candidates, candidate_stats = generate_candidates(
        trip=trip,
        road_gdf=road_gdf,
        radius_m=50,
        max_candidates=8,
    )

    print("\nCANDIDATE GENERATION SUMMARY")

    for key, value in candidate_stats.items():
        print(f"{key}: {value}")

    print("\nFIRST 10 CANDIDATES")
    print(candidates.head(10).to_string(index=False))

    # Verify candidate arc IDs belong to the original road network.
    candidate_arc_ids = set(candidates["candidate_arc_id"])

    print("\nCANDIDATE ARC ID VALIDATION")
    print("-" * 45)

    print(
        "All candidate arc IDs exist:",
        candidate_arc_ids.issubset(original_arc_ids)
    )

    candidates_per_observation = candidates.groupby("obs_idx").size()

    print(
        "Maximum candidates for one observation:",
        candidates_per_observation.max()
        if not candidates_per_observation.empty else 0
    )

    print(
        "All distances non-negative:",
        (candidates["distance_m"] >= 0).all()
    )

    print(
        "All projected offsets non-negative:",
        (candidates["projected_offset_m"] >= 0).all()
    )

    # 11. Extract candidate features
    print("\nEXTRACTING CANDIDATE FEATURES")
    print("-" * 45)

    features = extract_candidate_features(
        candidates=candidates,
        road_gdf=road_gdf,
        track=trip["track"],
    )

    print("Total feature rows:", len(features))
    print("Feature columns:", features.columns.tolist())

    print("\nFIRST 10 FEATURE ROWS")
    print(features.head(10).to_string(index=False))

    # 12. Validate geometric features
    print("\nFEATURE VALIDATION")
    print("-" * 45)

    print(
        "Candidate row count preserved:",
        len(features) == len(candidates)
    )

    print(
        "All candidate identities preserved:",
        features[
            ["trip_id", "obs_idx", "candidate_arc_id"]
        ].reset_index(drop=True).equals(
            candidates[
                ["trip_id", "obs_idx", "candidate_arc_id"]
            ].reset_index(drop=True)
        )
    )

    print(
        "All arc lengths non-negative:",
        (features["arc_length_m"] >= 0).all()
    )

    print(
        "All remaining distances non-negative:",
        (features["remaining_distance_m"] >= 0).all()
    )

    print(
        "All offset ratios between 0 and 1:",
        features["offset_ratio"].between(0, 1).all()
    )

    print(
        "Zero-length candidate rows:",
        features["is_zero_length_arc"].sum()
    )

    # 13. Validate movement features
    print("\nMOVEMENT FEATURE VALIDATION")
    print("-" * 45)

    first_obs_idx = trip["track"]["obs_idx"].min()

    first_observation_rows = features[
        features["obs_idx"] == first_obs_idx
    ]

    later_observation_rows = features[
        features["obs_idx"] > first_obs_idx
    ]

    print(
        "First observation has no previous movement:",
        first_observation_rows["time_delta_s"].isna().all()
    )

    print(
        "Positive time intervals after first observation:",
        (later_observation_rows["time_delta_s"] > 0).all()
    )

    print(
        "All calculated speeds non-negative:",
        (features["gps_speed_mps"].dropna() >= 0).all()
    )

    print(
        "Heading differences within 0-180 degrees:",
        features["heading_difference_deg"]
        .dropna()
        .between(0, 180)
        .all()
    )

    print("\nMOVEMENT FEATURE SAMPLE")

    print(
        features[
            [
                "obs_idx",
                "candidate_arc_id",
                "gps_displacement_m",
                "gps_speed_mps",
                "gps_heading_deg",
                "candidate_heading_deg",
                "heading_difference_deg",
            ]
        ].head(10).to_string(index=False)
    )

    # 14. Feature statistics
    print("\nFEATURE STATISTICS")

    print(
        features[
            [
                "distance_m",
                "projected_offset_m",
                "arc_length_m",
                "remaining_distance_m",
                "offset_ratio",
                "gps_displacement_m",
                "gps_speed_mps",
                "heading_difference_deg",
            ]
        ].describe()
    )


if __name__ == "__main__":
    main()
