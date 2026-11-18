
from pathlib import Path
from collections import defaultdict
from bisect import bisect_left

import pandas as pd

from src.data.loader import load_trip


DATA_DIR = Path("data")
TRIPS = ["00000061", "00000033", "00000089"]


def inspect_trip(trip_id):
    trip = load_trip(trip_id, split="train", data_dir=str(DATA_DIR))

    route_ids = trip["route"]["arc_id"].astype(int).tolist()

    occurrences = defaultdict(list)
    for idx, arc_id in enumerate(route_ids):
        occurrences[arc_id].append(idx)

    features = pd.read_csv(
        DATA_DIR / "processed" / "train" / f"{trip_id}_features.csv",
        dtype={"trip_id": str},
    )

    rows = []

    for obs_idx, group in features.groupby("obs_idx", sort=True):
        candidate_ids = set(group["candidate_arc_id"].astype(int))

        positions = sorted(
            pos
            for arc_id in candidate_ids
            for pos in occurrences.get(arc_id, [])
        )

        rows.append({
            "obs_idx": int(obs_idx),
            "candidate_count": len(candidate_ids),
            "route_candidate_count": sum(
                arc_id in occurrences for arc_id in candidate_ids
            ),
            "nearest_candidate_distance_m": group["distance_m"].min(),
            "earliest_route_position": min(positions) if positions else None,
            "latest_route_position": max(positions) if positions else None,
            "possible_positions": positions,
        })

    last_position = 0

    for row in rows:
        positions = row["possible_positions"]

        if not positions:
            row["status"] = "NO_ROUTE_CANDIDATE"
            row["chosen_position"] = None
            continue

        idx = bisect_left(positions, last_position)

        if idx == len(positions):
            row["status"] = "ORDER_FAILURE"
            row["chosen_position"] = None
        else:
            chosen = positions[idx]
            row["status"] = "ASSIGNED"
            row["chosen_position"] = chosen
            last_position = chosen

    df = pd.DataFrame(rows)

    print("\n" + "=" * 70)
    print(f"TRIP {trip_id}")
    print("=" * 70)

    print(df["status"].value_counts())

    print("\nFIRST 20 ORDER FAILURES")
    print(
        df[df["status"] == "ORDER_FAILURE"][
            [
                "obs_idx",
                "nearest_candidate_distance_m",
                "earliest_route_position",
                "latest_route_position",
            ]
        ].head(20).to_string(index=False)
    )

    print("\nFIRST 20 OBSERVATIONS WITHOUT ROUTE CANDIDATES")
    print(
        df[df["status"] == "NO_ROUTE_CANDIDATE"][
            [
                "obs_idx",
                "candidate_count",
                "nearest_candidate_distance_m",
            ]
        ].head(20).to_string(index=False)
    )

    output = (
        DATA_DIR / "processed" / "train"
        / f"{trip_id}_route_order_details.csv"
    )
    df.drop(columns=["possible_positions"]).to_csv(output, index=False)
    print(f"\nSaved: {output}")


def main():
    for trip_id in TRIPS:
        inspect_trip(trip_id)


if __name__ == "__main__":
    main()
