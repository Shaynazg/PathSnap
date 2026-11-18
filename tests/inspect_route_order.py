
from pathlib import Path
from collections import defaultdict
from bisect import bisect_left

import pandas as pd

from src.data.loader import load_trip


DATA_DIR = Path("data")
PROCESSED_DIR = DATA_DIR / "processed" / "train"
TRAIN_DIR = DATA_DIR / "train"


def inspect_trip(trip_id):
    trip = load_trip(trip_id, split="train", data_dir=str(DATA_DIR))

    route_ids = trip["route"]["arc_id"].astype(int).tolist()

    # Preserve every occurrence of an arc in the ordered route.
    occurrences = defaultdict(list)
    for route_idx, arc_id in enumerate(route_ids):
        occurrences[arc_id].append(route_idx)

    features = pd.read_csv(
        PROCESSED_DIR / f"{trip_id}_features.csv",
        dtype={"trip_id": str},
    )

    results = []

    # Process observations in chronological order.
    for obs_idx, group in features.groupby("obs_idx", sort=True):
        candidate_ids = set(
            group["candidate_arc_id"].astype(int)
        )

        route_candidate_ids = candidate_ids.intersection(occurrences)

        possible_positions = sorted(
            position
            for arc_id in route_candidate_ids
            for position in occurrences[arc_id]
        )

        results.append({
            "obs_idx": int(obs_idx),
            "possible_positions": possible_positions,
            "has_route_candidate": bool(possible_positions),
        })

    # Greedily choose the earliest route occurrence that does
    # not move backwards relative to the previous observation.
    last_position = 0
    assigned = 0
    order_failures = 0
    no_route_candidate = 0
    first_position = None
    final_position = None

    for result in results:
        positions = result["possible_positions"]

        if not positions:
            no_route_candidate += 1
            continue

        choice_idx = bisect_left(positions, last_position)

        if choice_idx == len(positions):
            order_failures += 1
            continue

        chosen_position = positions[choice_idx]

        if first_position is None:
            first_position = chosen_position

        last_position = chosen_position
        final_position = chosen_position
        assigned += 1

    total = len(results)

    return {
        "trip_id": trip_id,
        "observations": total,
        "no_route_candidate": no_route_candidate,
        "order_failures": order_failures,
        "monotonic_assignments": assigned,
        "monotonic_coverage": assigned / total if total else 0,
        "first_route_position": first_position,
        "last_route_position": final_position,
        "route_arc_count": len(route_ids),
    }


def main():
    rows = []

    for folder in sorted(TRAIN_DIR.iterdir()):
        if not folder.is_dir():
            continue

        trip_id = folder.name

        try:
            rows.append(inspect_trip(trip_id))
            print(f"Inspected {trip_id}")
        except Exception as exc:
            print(f"FAILED {trip_id}: {exc}")

    if not rows:
        print("No trips inspected.")
        return

    summary = pd.DataFrame(rows)
    output = PROCESSED_DIR / "route_order_diagnostics.csv"
    summary.to_csv(output, index=False)

    print("\n" + "=" * 65)
    print("ROUTE ORDER FEASIBILITY SUMMARY")
    print("=" * 65)

    print(
        summary[
            [
                "monotonic_coverage",
                "no_route_candidate",
                "order_failures",
            ]
        ].describe()
    )

    print("\nLOWEST MONOTONIC COVERAGE")
    print(
        summary.sort_values("monotonic_coverage").head(15).to_string(
            index=False
        )
    )

    print(f"\nSaved to: {output}")


if __name__ == "__main__":
    main()
