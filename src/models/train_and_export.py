"""Train and export the production candidate ranking model.

Fits the RandomForest pipeline on all aligned training candidates,
evaluates on the student test candidates, and exports model artifacts.
"""

import json
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.pipeline import Pipeline

from src.models.classifier import FEATURE_COLUMNS, IDENTIFIERS, prepare_candidate_features

TARGET = "is_match"
ARTIFACT_DIR = Path("artifacts")


def ranking_metrics(frame: pd.DataFrame) -> dict[str, float]:
    top1_correct = 0
    top3_correct = 0
    reciprocal_ranks = []

    for _, group in frame.groupby(["trip_id", "obs_idx"]):
        if int(group["is_match"].sum()) != 1:
            continue
        ranked = group.sort_values("raw_probability", ascending=False)
        match_ranks = np.where(ranked["is_match"].to_numpy() == 1)[0]
        if len(match_ranks) == 0:
            continue
        rank = int(match_ranks[0] + 1)
        top1_correct += int(rank == 1)
        top3_correct += int(rank <= 3)
        reciprocal_ranks.append(1.0 / rank)

    if not reciprocal_ranks:
        return {"Top-1 Accuracy": float("nan"), "Top-3 Recall": float("nan"), "MRR": float("nan")}
    n = len(reciprocal_ranks)
    return {
        "Top-1 Accuracy": top1_correct / n,
        "Top-3 Recall": top3_correct / n,
        "MRR": float(np.mean(reciprocal_ranks)),
    }


def main():
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    model_path = ARTIFACT_DIR / "pathsnap_gradient_boosting.joblib"
    metadata_path = ARTIFACT_DIR / "pathsnap_model_metadata.json"
    predictions_path = ARTIFACT_DIR / "student_test_candidate_predictions.csv"
    comparison_path = ARTIFACT_DIR / "pathsnap_tuned_model_comparison.csv"

    print("Loading datasets...")
    train_df = pd.read_csv("data/processed/train_candidates.csv")
    train_aligned = train_df[train_df["aligned"] == 1].copy()

    test_df = pd.read_csv("data/processed/student_test_candidates.csv")
    test_aligned = test_df[test_df["aligned"] == 1].copy()

    print(f"Train rows: {len(train_aligned)}, Test rows: {len(test_aligned)}")

    # Prepare features
    X_train = train_aligned[FEATURE_COLUMNS]
    y_train = train_aligned[TARGET]

    X_test = test_aligned[FEATURE_COLUMNS]
    y_test = test_aligned[TARGET]

    print("Building Gradient Boosting baseline pipeline...")
    pipeline = Pipeline([
        ("preprocess", SimpleImputer(strategy="median", keep_empty_features=True)),
        ("model", GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42)),
    ])

    print("Fitting model...")
    t0 = time.time()
    pipeline.fit(X_train, y_train)
    fit_time = time.time() - t0
    print(f"Fitted in {fit_time:.2f}s")

    # Evaluate on student test set
    print("Evaluating on held-out student test candidates...")
    test_proba = pipeline.predict_proba(X_test)[:, 1]
    test_pred = (test_proba >= 0.5).astype(int)

    eval_frame = test_aligned[["trip_id", "obs_idx", "candidate_arc_id", "is_match"]].copy()
    eval_frame["raw_probability"] = test_proba
    eval_frame["probability"] = eval_frame.groupby(["trip_id", "obs_idx"])["raw_probability"].transform(
        lambda s: s / s.sum() if s.sum() > 0 else np.ones(len(s)) / len(s)
    )

    test_metrics = ranking_metrics(eval_frame)
    test_acc = float(accuracy_score(y_test, test_pred))
    test_prec = float(precision_score(y_test, test_pred, zero_division=0))
    test_rec = float(recall_score(y_test, test_pred, zero_division=0))
    test_f1 = float(f1_score(y_test, test_pred, average="macro", zero_division=0))

    print("Student Test Metrics:")
    print(f"  Top-1 Accuracy: {test_metrics['Top-1 Accuracy']:.4f}")
    print(f"  Top-3 Recall:   {test_metrics['Top-3 Recall']:.4f}")
    print(f"  MRR:            {test_metrics['MRR']:.4f}")
    print(f"  Macro F1:       {test_f1:.4f}")
    print(f"  Accuracy:       {test_acc:.4f}")

    # Export model artifact
    print(f"Exporting model to {model_path}...")
    joblib.dump(pipeline, model_path)

    # Export metadata
    metadata = {
        "model_name": "Gradient Boosting (Baseline)",
        "selected_model": "Gradient Boosting",
        "configuration": {"n_estimators": 100, "learning_rate": 0.1, "max_depth": 3, "random_state": 42},
        "hyperparameter_optimization": False,
        "feature_columns": FEATURE_COLUMNS,
        "target": TARGET,
        "identifier_columns": IDENTIFIERS,
        "training_rows": int(len(train_aligned)),
        "training_trips": int(train_aligned["trip_id"].nunique()),
        "student_test_rows": int(len(test_aligned)),
        "student_test_trips": int(test_aligned["trip_id"].nunique()),
        "fit_duration_seconds": fit_time,
        "metrics": {
            "top1_accuracy": test_metrics["Top-1 Accuracy"],
            "top3_recall": test_metrics["Top-3 Recall"],
            "mrr": test_metrics["MRR"],
            "macro_f1": test_f1,
            "accuracy": test_acc,
            "precision": test_prec,
            "recall": test_rec,
        },
    }
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    # Export student test predictions
    eval_frame.to_csv(predictions_path, index=False)

    # Copy comparison CSV if available or generate from comparison
    comp_src = Path("synthetic_model_comparison (5).csv")
    if comp_src.is_file():
        df_comp = pd.read_csv(comp_src)
        df_comp.to_csv(comparison_path, index=False)

    print("Export complete!")


if __name__ == "__main__":
    main()
