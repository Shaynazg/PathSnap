
from pathlib import Path

import pandas as pd


def load_trip(trip_id, split="train", data_dir="data"):
    """
    Load all four files belonging to one trip.

    Parameters
    ----------
    trip_id : str
        Trip folder name, e.g. "00000001".
    split : str
        Dataset partition: "train" or "student_test".
    data_dir : str
        Root data directory.

    Returns
    -------
    dict
        Dictionary containing track, nodes, arcs, and route DataFrames.
    """

    trip_id = str(trip_id).zfill(8)

    trip_folder = Path(data_dir) / split / trip_id

    if not trip_folder.is_dir():
        raise FileNotFoundError(
            f"Trip folder not found: {trip_folder}"
        )

    # Expected file paths
    track_file = trip_folder / f"{trip_id}.track"
    nodes_file = trip_folder / f"{trip_id}.nodes"
    arcs_file = trip_folder / f"{trip_id}.arcs"
    route_file = trip_folder / f"{trip_id}.route"

    files = {
        "track": track_file,
        "nodes": nodes_file,
        "arcs": arcs_file,
        "route": route_file,
    }

    # Check that all four files exist
    for file_type, file_path in files.items():
        if not file_path.is_file():
            raise FileNotFoundError(
                f"Missing {file_type} file: {file_path}"
            )

    # Load GPS observations
    track = pd.read_csv(
        track_file,
        sep=r"\s+",
        header=None,
        names=["longitude", "latitude", "timestamp"]
    )

    # Original GPS observation indices
    track.insert(0, "obs_idx", range(len(track)))

    # Load road network nodes
    nodes = pd.read_csv(
        nodes_file,
        sep=r"\s+",
        header=None,
        names=["longitude", "latitude"]
    )

    # Original node IDs are their zero-based row indices
    nodes.insert(0, "node_id", range(len(nodes)))

    # Load directed road arcs
    arcs = pd.read_csv(
        arcs_file,
        sep=r"\s+",
        header=None,
        names=["from_node_id", "to_node_id"]
    )

    # Original arc IDs are their zero-based row indices
    arcs.insert(0, "arc_id", range(len(arcs)))

    # Load ground-truth route
    route = pd.read_csv(
        route_file,
        sep=r"\s+",
        header=None,
        names=["arc_id"]
    )

    # Route position is independent of GPS observation index
    route.insert(0, "route_idx", range(len(route)))

    # Basic validation
    if track.empty or nodes.empty or arcs.empty or route.empty:
        raise ValueError(f"Trip {trip_id} contains an empty file.")

    return {
        "trip_id": trip_id,
        "track": track,
        "nodes": nodes,
        "arcs": arcs,
        "route": route,
    }
