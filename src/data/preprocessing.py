
import networkx as nx
import numpy as np
import geopandas as gpd

from shapely.geometry import LineString
from pyproj import Transformer


def validate_trip(trip):
    """
    Validate the four loaded files belonging to one trip.

    Returns a report containing errors, warnings, and statistics.
    """

    track = trip["track"]
    nodes = trip["nodes"]
    arcs = trip["arcs"]
    route = trip["route"]

    errors = []
    warnings = []

    # --------------------------------------------------
    # 1. Check missing values
    # --------------------------------------------------

    for name, df in [
        ("track", track),
        ("nodes", nodes),
        ("arcs", arcs),
        ("route", route),
    ]:
        if df.isnull().any().any():
            errors.append(f"{name} contains missing values.")

    # --------------------------------------------------
    # 2. Validate GPS observations
    # --------------------------------------------------

    if not track.empty:
        if not track["longitude"].between(-180, 180).all():
            errors.append("Invalid GPS longitude values.")

        if not track["latitude"].between(-90, 90).all():
            errors.append("Invalid GPS latitude values.")

        if not np.isfinite(
            track[["longitude", "latitude", "timestamp"]].to_numpy()
        ).all():
            errors.append("GPS data contains non-finite values.")

        timestamp_diffs = track["timestamp"].diff().dropna()

        if (timestamp_diffs < 0).any():
            errors.append("GPS timestamps are not in chronological order.")

        if (timestamp_diffs == 0).any():
            warnings.append(
                "Some GPS observations have duplicate timestamps."
            )

        if (timestamp_diffs > 1).any():
            warnings.append(
                "GPS observations contain gaps greater than one second."
            )

    # --------------------------------------------------
    # 3. Validate nodes
    # --------------------------------------------------

    if nodes["node_id"].duplicated().any():
        errors.append("Duplicate node IDs found.")

    if not np.isfinite(
        nodes[["longitude", "latitude"]].to_numpy()
    ).all():
        errors.append("Node coordinates contain non-finite values.")

    # --------------------------------------------------
    # 4. Validate arcs
    # --------------------------------------------------

    if arcs["arc_id"].duplicated().any():
        errors.append("Duplicate arc IDs found.")

    valid_node_ids = set(nodes["node_id"])

    referenced_nodes = (
        set(arcs["from_node_id"]) |
        set(arcs["to_node_id"])
    )

    invalid_nodes = referenced_nodes - valid_node_ids

    if invalid_nodes:
        errors.append(
            f"Arcs reference {len(invalid_nodes)} nonexistent nodes."
        )

    # Detect parallel directed arcs
    parallel_mask = arcs.duplicated(
        subset=["from_node_id", "to_node_id"],
        keep=False
    )

    parallel_arc_count = int(parallel_mask.sum())

    if parallel_arc_count:
        warnings.append(
            f"{parallel_arc_count} arcs belong to parallel "
            "directed node pairs. Preserve them using MultiDiGraph."
        )

    # --------------------------------------------------
    # 5. Validate route
    # --------------------------------------------------

    valid_arc_ids = set(arcs["arc_id"])
    route_arc_ids = set(route["arc_id"])

    invalid_route_arcs = route_arc_ids - valid_arc_ids

    if invalid_route_arcs:
        errors.append(
            f"Route references {len(invalid_route_arcs)} "
            "nonexistent arc IDs."
        )

    # --------------------------------------------------
    # 6. Build validation report
    # --------------------------------------------------

    report = {
        "trip_id": trip["trip_id"],
        "num_observations": len(track),
        "num_nodes": len(nodes),
        "num_arcs": len(arcs),
        "num_route_arcs": len(route),
        "num_parallel_arcs": parallel_arc_count,
        "errors": errors,
        "warnings": warnings,
        "valid": len(errors) == 0,
    }

    return report


def print_validation_report(report):
    """Print a readable validation summary."""

    print(f"\nVALIDATION REPORT: {report['trip_id']}")
    print("-" * 45)

    print("GPS observations:", report["num_observations"])
    print("Road nodes:", report["num_nodes"])
    print("Road arcs:", report["num_arcs"])
    print("Route arcs:", report["num_route_arcs"])
    print("Arcs in parallel directed pairs:", report["num_parallel_arcs"])

    print("\nErrors:")

    if report["errors"]:
        for error in report["errors"]:
            print("  ERROR:", error)
    else:
        print("  None")

    print("\nWarnings:")

    if report["warnings"]:
        for warning in report["warnings"]:
            print("  WARNING:", warning)
    else:
        print("  None")

    print(
        "\nOverall status:",
        "VALID" if report["valid"] else "INVALID"
    )


def build_road_network(trip):
    """
    Construct a directed MultiDiGraph and a GeoDataFrame
    from the nodes and arcs of one trip.

    Returns
    -------
    road_graph : networkx.MultiDiGraph
        Directed graph preserving original arc IDs and endpoints.

    road_gdf : geopandas.GeoDataFrame
        Projected road geometries with arc lengths in metres.

    projected_crs : pyproj.CRS
        Projected coordinate reference system used.
    """

    nodes = trip["nodes"]
    arcs = trip["arcs"]

    # --------------------------------------------------
    # 1. Determine a suitable projected CRS
    # --------------------------------------------------

    if nodes.empty:
        raise ValueError("Cannot build road network: nodes DataFrame is empty.")

    node_points = gpd.GeoSeries(
        gpd.points_from_xy(
            nodes["longitude"],
            nodes["latitude"]
        ),
        crs="EPSG:4326"
    )

    projected_crs = node_points.estimate_utm_crs()

    if projected_crs is None:
        raise ValueError("Could not determine a suitable UTM CRS.")

    # Convert longitude/latitude into projected metres
    transformer = Transformer.from_crs(
        "EPSG:4326",
        projected_crs,
        always_xy=True
    )

    x_coords, y_coords = transformer.transform(
        nodes["longitude"].to_numpy(),
        nodes["latitude"].to_numpy()
    )

    # Map original node IDs to projected coordinates
    node_coordinates = {
        int(row.node_id): (
            float(x_coords[i]),
            float(y_coords[i])
        )
        for i, row in enumerate(nodes.itertuples(index=False))
    }

    # --------------------------------------------------
    # 2. Create the directed MultiDiGraph
    # --------------------------------------------------

    road_graph = nx.MultiDiGraph()

    for i, row in enumerate(nodes.itertuples(index=False)):

        node_id = int(row.node_id)

        road_graph.add_node(
            node_id,
            longitude=float(row.longitude),
            latitude=float(row.latitude),
            x=float(x_coords[i]),
            y=float(y_coords[i])
        )

    # --------------------------------------------------
    # 3. Create geometry and length for every arc
    # --------------------------------------------------

    road_records = []

    for row in arcs.itertuples(index=False):

        arc_id = int(row.arc_id)
        from_node = int(row.from_node_id)
        to_node = int(row.to_node_id)

        # Retrieve projected coordinates
        start = node_coordinates[from_node]
        end = node_coordinates[to_node]

        # Construct the straight-line geometry
        geometry = LineString([start, end])

        # Because coordinates are projected, length is in metres
        # Ensure positive length for routing and HMM transition validation
        length_m = max(float(geometry.length), 1e-3)

        # Preserve original arc ID and endpoint attributes
        road_graph.add_edge(
            from_node,
            to_node,
            key=arc_id,
            arc_id=arc_id,
            from_node_id=from_node,
            to_node_id=to_node,
            length_m=length_m,
            geometry=geometry
        )

        road_records.append({
            "arc_id": arc_id,
            "from_node_id": from_node,
            "to_node_id": to_node,
            "length_m": length_m,
            "geometry": geometry
        })

    # --------------------------------------------------
    # 4. Create the GeoDataFrame
    # --------------------------------------------------

    road_gdf = gpd.GeoDataFrame(
        road_records,
        geometry="geometry",
        crs=projected_crs
    )

    return road_graph, road_gdf, projected_crs


def analyze_road_graph(road_graph):
    """
    Analyze basic connectivity and structural properties
    of the directed road graph.
    """

    # Weak connectivity ignores edge direction
    weak_components = list(
        nx.weakly_connected_components(road_graph)
    )

    largest_component_size = (
        max(map(len, weak_components))
        if weak_components else 0
    )

    # Identify isolated nodes and self-loop edges
    isolated_nodes = list(nx.isolates(road_graph))

    self_loop_count = nx.number_of_selfloops(road_graph)

    # Identify zero-length edges
    zero_length_edges = [
        (u, v, key)
        for u, v, key, data in road_graph.edges(
            keys=True,
            data=True
        )
        if data["length_m"] == 0
    ]

    return {
        "num_nodes": road_graph.number_of_nodes(),
        "num_edges": road_graph.number_of_edges(),
        "num_weak_components": len(weak_components),
        "largest_component_size": largest_component_size,
        "largest_component_fraction": (
            largest_component_size / road_graph.number_of_nodes()
            if road_graph.number_of_nodes() else 0
        ),
        "num_isolated_nodes": len(isolated_nodes),
        "num_self_loops": self_loop_count,
        "num_zero_length_edges": len(zero_length_edges),
        "zero_length_edge_examples": zero_length_edges[:10],
    }


def check_route_continuity(trip):
    """
    Check whether consecutive ground-truth route arcs
    connect end-to-start.

    This does not compare route positions with GPS indices.
    """

    arcs = trip["arcs"].set_index("arc_id")
    route_arc_ids = trip["route"]["arc_id"].tolist()

    discontinuities = []

    for i in range(len(route_arc_ids) - 1):

        current_arc_id = int(route_arc_ids[i])
        next_arc_id = int(route_arc_ids[i + 1])

        current_to = int(
            arcs.loc[current_arc_id, "to_node_id"]
        )

        next_from = int(
            arcs.loc[next_arc_id, "from_node_id"]
        )

        if current_to != next_from:
            discontinuities.append({
                "route_idx": i,
                "current_arc_id": current_arc_id,
                "next_arc_id": next_arc_id,
                "current_to_node": current_to,
                "next_from_node": next_from,
            })

    return {
        "num_route_arcs": len(route_arc_ids),
        "num_transitions_checked": max(0, len(route_arc_ids) - 1),
        "num_discontinuities": len(discontinuities),
        "discontinuity_examples": discontinuities[:10],
        "continuous": len(discontinuities) == 0,
    }


def shortest_directed_arc_path(road_graph, source_node, target_node):
    """
    Find the shortest directed path between two nodes.

    Returns both the node sequence and the original arc IDs
    selected along the path.

    For parallel arcs, the minimum-length edge is selected.
    """

    source_node = int(source_node)
    target_node = int(target_node)

    if source_node not in road_graph:
        raise ValueError(f"Unknown source node: {source_node}")

    if target_node not in road_graph:
        raise ValueError(f"Unknown target node: {target_node}")

    try:
        node_path = nx.shortest_path(
            road_graph,
            source=source_node,
            target=target_node,
            weight="length_m",
            method="dijkstra"
        )

    except nx.NetworkXNoPath:
        return {
            "reachable": False,
            "node_path": [],
            "arc_ids": [],
            "distance_m": float("inf"),
        }

    selected_arc_ids = []
    total_distance_m = 0.0

    for u, v in zip(node_path[:-1], node_path[1:]):

        parallel_edges = road_graph.get_edge_data(u, v)

        # Select the shortest parallel directed edge
        selected_key, selected_data = min(
            parallel_edges.items(),
            key=lambda item: item[1]["length_m"]
        )

        selected_arc_ids.append(
            int(selected_data["arc_id"])
        )

        total_distance_m += float(
            selected_data["length_m"]
        )

    return {
        "reachable": True,
        "node_path": node_path,
        "arc_ids": selected_arc_ids,
        "distance_m": total_distance_m,
    }
