"""
Builds the agreed candidate table for model training.

    python -m src.features.build_training_table            # train + student_test

Outputs (data/processed/):
    train_candidates.csv, student_test_candidates.csv   - one row per (obs, candidate arc)
    <split>_label_quality.csv                           - per-trip alignment coverage
Teammates: ALWAYS train/evaluate on rows with aligned == 1, and use
GroupKFold on trip_id.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from src.data.loader import load_trip
from src.features.pipeline import process_trip
from src.features.route_alignment import add_is_match

DATA_DIR = Path("data")

# agreed schema names  <-  names produced by extraction.py
RENAME = {
    "gps_speed_mps": "speed_mps",
    "gps_heading_deg": "bearing",
    "heading_difference_deg": "heading_diff",
}
AGREED = ["trip_id", "obs_idx", "timestamp", "candidate_arc_id", "distance_m",
          "heading_diff", "speed_mps", "bearing", "is_match"]
# extra columns that are safe to use as model inputs
EXTRA = ["projected_offset_m", "arc_length_m", "remaining_distance_m",
         "offset_ratio", "time_delta_s", "gps_displacement_m",
         "candidate_heading_deg", "heading_missing"]
META = ["aligned"]          # filter on this; never a model input
# longitude/latitude/timestamp/obs_idx/trip_id are identifiers - NOT model inputs.


def build_trip(trip_id, split):
    feat_path = DATA_DIR / "processed" / split / f"{trip_id}_features.csv"
    if feat_path.exists():
        feats = pd.read_csv(feat_path, dtype={"trip_id": str})
    else:
        feats = process_trip(trip_id, split=split, data_dir=str(DATA_DIR))["features"]
    route = load_trip(trip_id, split, str(DATA_DIR))["route"]["arc_id"].astype(int).tolist()
    n_obs = len(load_trip(trip_id, split, str(DATA_DIR))["track"])

    df = add_is_match(feats, route).rename(columns=RENAME)
    # NaN handling so sklearn models work out of the box (flag keeps the information)
    df["heading_missing"] = df["heading_diff"].isna().astype(int)
    df["heading_diff"] = df["heading_diff"].fillna(90.0)   # uninformative
    df["speed_mps"] = df["speed_mps"].fillna(0.0)
    df["bearing"] = df["bearing"].fillna(0.0)
    df["time_delta_s"] = df["time_delta_s"].fillna(1.0)
    df["gps_displacement_m"] = df["gps_displacement_m"].fillna(0.0)

    aligned_obs = df.loc[df["aligned"] == 1, "obs_idx"].nunique()
    hit = df.loc[df["is_match"] == 1, "obs_idx"].nunique()
    quality = {"trip_id": trip_id, "split": split, "n_obs": n_obs,
               "obs_with_candidates": df["obs_idx"].nunique(),
               "aligned_obs": aligned_obs, "aligned_frac": aligned_obs / n_obs,
               "labels_in_candidates": hit}
    return df[AGREED + EXTRA + META], quality


def build_split(split):
    folders = sorted(p.name for p in (DATA_DIR / split).iterdir()
                     if p.is_dir() and (p / f"{p.name}.track").is_file())
    tables, quals = [], []
    for i, t in enumerate(folders, 1):
        print(f"[{split} {i}/{len(folders)}] {t}", flush=True)
        df, q = build_trip(t, split)
        tables.append(df); quals.append(q)
    out = DATA_DIR / "processed"
    full = pd.concat(tables, ignore_index=True)
    full.to_csv(out / f"{split}_candidates.csv", index=False)
    pd.DataFrame(quals).to_csv(out / f"{split}_label_quality.csv", index=False)
    return full, pd.DataFrame(quals)


if __name__ == "__main__":
    for split in (sys.argv[1:] or ["train", "student_test"]):
        full, q = build_split(split)
        lab = full[full["aligned"] == 1]
        print(f"\n{split}: {len(full)} rows | trainable rows {len(lab)} | "
              f"positive rate {lab['is_match'].mean():.3f} | "
              f"mean aligned frac {q['aligned_frac'].mean():.2%}")
