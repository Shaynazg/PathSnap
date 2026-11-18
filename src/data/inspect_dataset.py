
from pathlib import Path
import pandas as pd
import numpy as np

DATA_ROOT = Path("data/train")

OUTPUT_DIR = Path("data/processed/eda")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

FILE_SCHEMAS = {
    ".track": ["longitude", "latitude", "timestamp"],
    ".nodes": ["longitude", "latitude"],
    ".arcs": ["from_node_id", "to_node_id"],
    ".route": ["arc_id"]
}


def inspect_file(path):

    extension = path.suffix.lower()

    if extension not in FILE_SCHEMAS:
        return None

    expected_columns = FILE_SCHEMAS[extension]

    result = {
        "file": str(path),
        "extension": extension,
        "size_bytes": path.stat().st_size,
        "rows": None,
        "columns": len(expected_columns),
        "missing_values": None,
        "invalid_values": None,
        "status": "OK"
    }

    try:
        df = pd.read_csv(
            path,
            sep=r"\s+",
            header=None,
            names=expected_columns
        )

        result["rows"] = len(df)

        # Convert values to numeric.
        for col in expected_columns:
            df[col] = pd.to_numeric(
                df[col], errors="coerce"
            )

        result["missing_values"] = int(
            df.isna().sum().sum()
        )

        invalid = 0

        if extension in [".track", ".nodes"]:
            invalid += int(
                ((df["longitude"] < -180) |
                 (df["longitude"] > 180) |
                 (df["latitude"] < -90) |
                 (df["latitude"] > 90)).sum()
            )

        if extension == ".track":
            timestamps = df["timestamp"].dropna()

            if len(timestamps) > 1:
                result["timestamp_monotonic"] = bool(
                    timestamps.is_monotonic_increasing
                )

                result["duplicate_timestamps"] = int(
                    timestamps.duplicated().sum()
                )

                result["median_time_gap"] = float(
                    timestamps.diff().median()
                )

        if extension == ".arcs":
            invalid += int(
                ((df < 0).any(axis=1)).sum()
            )

        if extension == ".route":
            invalid += int(
                (df["arc_id"] < 0).sum()
            )

        result["invalid_values"] = invalid

        if result["missing_values"] > 0 or invalid > 0:
            result["status"] = "CHECK REQUIRED"

    except Exception as e:
        result["status"] = f"ERROR: {e}"

    return result


def main():

    if not DATA_ROOT.exists():
        raise FileNotFoundError(
            f"Dataset directory not found: {DATA_ROOT}"
        )

    files = [
        p for p in DATA_ROOT.rglob("*")
        if p.is_file()
        and p.suffix.lower() in FILE_SCHEMAS
    ]

    print(f"\nTotal recognized files: {len(files)}")

    results = []

    for path in files:
        result = inspect_file(path)

        if result:
            results.append(result)

    inventory = pd.DataFrame(results)

    if inventory.empty:
        print("No recognized dataset files found.")
        return

    inventory.to_csv(
        OUTPUT_DIR / "file_inventory.csv",
        index=False
    )

    print("\nFILE COUNTS BY EXTENSION")
    print(inventory["extension"].value_counts())

    print("\nFILE INVENTORY")
    print(inventory.to_string(index=False))

    print("\nTOTAL ROWS BY FILE TYPE")

    print(
        inventory.groupby("extension")["rows"]
        .sum()
    )

    print("\nFILES REQUIRING ATTENTION")

    print(
        inventory[
            inventory["status"] != "OK"
        ][["file", "status"]].to_string(index=False)
    )

    print(
        f"\nInventory saved to: "
        f"{OUTPUT_DIR / 'file_inventory.csv'}"
    )


if __name__ == "__main__":
    main()
