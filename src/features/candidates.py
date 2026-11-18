
import pandas as pd
from pyproj import Transformer
from shapely.geometry import Point


def generate_candidates(
    trip,
    road_gdf,
    radius_m=50,
    max_candidates=8,
):
    """
    Generate nearby road-arc candidates for every GPS observation.

    Parameters
    ----------
    trip : dict
        Trip dictionary returned by load_trip().
    road_gdf : GeoDataFrame
        Projected road network returned by build_road_network().
    radius_m : float
        Maximum perpendicular distance from a GPS point to a candidate arc.
    max_candidates : int
        Maximum number of candidates retained per GPS observation.

    Returns
    -------
    pandas.DataFrame
        Candidate arcs with GPS identity and projection information.
    """

    if radius_m <= 0:
        raise ValueError("radius_m must be positive.")

    if max_candidates < 1:
        raise ValueError("max_candidates must be at least 1.")

    if road_gdf.crs is None:
        raise ValueError("Road GeoDataFrame must have a CRS.")

    required_track_columns = {
        "obs_idx", "longitude", "latitude", "timestamp"
    }

    required_road_columns = {
        "arc_id", "geometry", "length_m"
    }

    if not required_track_columns.issubset(trip["track"].columns):
        raise ValueError("Track is missing required columns.")

    if not required_road_columns.issubset(road_gdf.columns):
        raise ValueError("Road GeoDataFrame is missing required columns.")

    trip_id = trip["trip_id"]
    track = trip["track"]

    # Convert GPS coordinates from WGS84 into the road network CRS.
    transformer = Transformer.from_crs(
        "EPSG:4326",
        road_gdf.crs,
        always_xy=True,
    )

    # Build the spatial index once, not once per GPS observation.
    spatial_index = road_gdf.sindex

    candidate_rows = []
    observations_without_candidates = []

    for _, observation in track.iterrows():

        obs_idx = int(observation["obs_idx"])

        longitude = float(observation["longitude"])
        latitude = float(observation["latitude"])

        # Project the GPS observation into metres.
        x, y = transformer.transform(longitude, latitude)
        gps_point = Point(x, y)

        # Search for road geometries intersecting the search buffer.
        search_area = gps_point.buffer(radius_m)

        possible_indices = spatial_index.query(search_area)

        observation_candidates = []

        for road_position in possible_indices:

            road_arc = road_gdf.iloc[int(road_position)]
            road_geometry = road_arc.geometry

            if road_geometry is None or road_geometry.is_empty:
                continue

            # Exact perpendicular distance from GPS point to road arc.
            distance_m = gps_point.distance(road_geometry)

            if distance_m > radius_m:
                continue

            # Position of the nearest point along the directed arc geometry.
            projected_offset_m = road_geometry.project(gps_point)

            observation_candidates.append({
                "trip_id": trip_id,
                "obs_idx": obs_idx,
                "timestamp": observation["timestamp"],
                "longitude": longitude,
                "latitude": latitude,
                "candidate_arc_id": int(road_arc["arc_id"]),
                "distance_m": float(distance_m),
                "projected_offset_m": float(projected_offset_m),
            })

        # Keep the nearest candidates, with arc ID as deterministic tie-breaker.
        observation_candidates.sort(
            key=lambda candidate: (
                candidate["distance_m"],
                candidate["candidate_arc_id"],
            )
        )

        observation_candidates = observation_candidates[:max_candidates]

        if not observation_candidates:
            observations_without_candidates.append(obs_idx)

        candidate_rows.extend(observation_candidates)

    columns = [
        "trip_id",
        "obs_idx",
        "timestamp",
        "longitude",
        "latitude",
        "candidate_arc_id",
        "distance_m",
        "projected_offset_m",
    ]

    candidates = pd.DataFrame(candidate_rows, columns=columns)

    # Diagnostics for checking candidate-generation coverage.
    total_observations = len(track)
    observations_with_candidates = (
        total_observations - len(observations_without_candidates)
    )

    diagnostics = {
        "trip_id": trip_id,
        "total_observations": total_observations,
        "observations_with_candidates": observations_with_candidates,
        "observations_without_candidates": len(
            observations_without_candidates
        ),
        "total_candidates": len(candidates),
        "average_candidates_per_observation": (
            len(candidates) / total_observations
            if total_observations else 0
        ),
        "radius_m": radius_m,
        "max_candidates": max_candidates,
        "missing_observation_indices": observations_without_candidates,
    }

    return candidates, diagnostics
