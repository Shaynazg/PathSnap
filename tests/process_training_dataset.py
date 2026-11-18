
from pathlib import Path
import pandas as pd

from src.features.pipeline import process_trip


def main():
    data_dir = Path("data")
    split = "train"

    input_dir = data_dir / split
    output_dir = data_dir / "processed" / split
    output_dir.mkdir(parents=True, exist_ok=True)

    # Discover actual trip folders, not placeholder files.
    trip_folders = sorted(
        folder
        for folder in input_dir.iterdir()
        if folder.is_dir()
        and (folder / f"{folder.name}.track").is_file()
    )

    if not trip_folders:
        raise FileNotFoundError(
            f"No trip folders containing .track files found in {input_dir}"
        )

    print(f"Found {len(trip_folders)} trips in {input_dir}")

    successful = []
    failed = []

    for index, folder in enumerate(trip_folders, start=1):
        trip_id = folder.name

        print("\n" + "=" * 65)
        print(f"TRIP {index}/{len(trip_folders)}: {trip_id}")
        print("=" * 65)

        try:
            result = process_trip(
                trip_id=trip_id,
                split=split,
                data_dir=str(data_dir),
                radius_m=50,
                max_candidates=8,
                save=True,
            )

            diagnostics = result["diagnostics"]

            successful.append({
                "trip_id": trip_id,
                "status": "success",
                "total_observations": diagnostics["total_observations"],
                "observations_with_candidates": (
                    diagnostics["observations_with_candidates"]
                ),
                "observations_without_candidates": (
                    diagnostics["observations_without_candidates"]
                ),
                "candidate_coverage": (
                    diagnostics["observations_with_candidates"]
                    / diagnostics["total_observations"]
                    if diagnostics["total_observations"] else 0
                ),
                "total_candidate_rows": diagnostics["total_candidate_rows"],
                "average_candidates_per_observation": (
                    diagnostics["average_candidates_per_observation"]
                ),
                "error": "",
            })

        except Exception as error:
            print(f"\nFAILED: {trip_id}")
            print(f"Reason: {error}")

            failed.append({
                "trip_id": trip_id,
                "status": "failed",
                "total_observations": None,
                "observations_with_candidates": None,
                "observations_without_candidates": None,
                "candidate_coverage": None,
                "total_candidate_rows": None,
                "average_candidates_per_observation": None,
                "error": str(error),
            })

    # Save a combined dataset-level report.
    all_results = successful + failed
    summary = pd.DataFrame(all_results)

    summary_path = output_dir / "dataset_summary.csv"
    summary.to_csv(summary_path, index=False)

    print("\n" + "=" * 65)
    print("TRAINING DATASET PROCESSING SUMMARY")
    print("=" * 65)
    print(f"Trips discovered: {len(trip_folders)}")
    print(f"Successful: {len(successful)}")
    print(f"Failed: {len(failed)}")
    print(f"Summary saved to: {summary_path}")

    if successful:
        successful_df = pd.DataFrame(successful)

        print(
            "Total GPS observations:",
            successful_df["total_observations"].sum()
        )
        print(
            "Total candidate rows:",
            successful_df["total_candidate_rows"].sum()
        )
        print(
            "Mean candidate coverage:",
            f"{successful_df['candidate_coverage'].mean():.2%}"
        )
        print(
            "Trips with incomplete candidate coverage:",
            (
                successful_df["observations_without_candidates"] > 0
            ).sum()
        )

    if failed:
        print("\nFAILED TRIPS")
        for item in failed:
            print(f"{item['trip_id']}: {item['error']}")

    print("\nProcessing finished.")


if __name__ == "__main__":
    main()
