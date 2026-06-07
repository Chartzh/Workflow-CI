"""
modelling.py  (versi MLProject)
──────────────────────────────────────────────────────────────────────────────
Model Training Pipeline — Diabetes Prediction Dataset
Dicoding Submission: Membangun Sistem Machine Learning (Tier Advance — Kriteria 3)
──────────────────────────────────────────────────────────────────────────────
"""

import argparse
import logging
import os
import sys

import dagshub
import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split

# ─── Logging Setup ────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  [%(levelname)s]  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger(__name__)

# ─── Konstanta ────────────────────────────────────────────────────────────────
TARGET_COL   = "diabetes"
RANDOM_STATE = 42
TEST_SIZE    = 0.2
EXPERIMENT   = "diabetes-prediction-baseline"

DEFAULT_PARAMS = {
    "n_estimators"     : 100,
    "max_depth"        : None,
    "min_samples_split": 2,
    "min_samples_leaf" : 1,
    "max_features"     : "sqrt",
    "class_weight"     : "balanced",
    "random_state"     : RANDOM_STATE,
    "n_jobs"           : -1,
}


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────
def plot_confusion_matrix(cm: np.ndarray, output_path: str) -> None:
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=["Non-Diabetes", "Diabetes"],
        yticklabels=["Non-Diabetes", "Diabetes"],
        linewidths=0.5, ax=ax,
    )
    ax.set_title("Confusion Matrix — Random Forest", fontsize=12, pad=12)
    ax.set_ylabel("Actual Label", fontsize=10)
    ax.set_xlabel("Predicted Label", fontsize=10)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_feature_importance(
    importances: np.ndarray, feature_names: list, output_path: str
) -> None:
    idx           = np.argsort(importances)[::-1]
    sorted_names  = [feature_names[i] for i in idx]
    sorted_values = importances[idx]
    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.barh(sorted_names[::-1], sorted_values[::-1], color="#4C72B0", edgecolor="white")
    ax.bar_label(bars, fmt="%.4f", padding=3, fontsize=8)
    ax.set_title("Feature Importance — Random Forest", fontsize=12, pad=12)
    ax.set_xlabel("Importance Score", fontsize=9)
    ax.set_xlim(0, sorted_values.max() * 1.15)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


# ──────────────────────────────────────────────────────────────────────────────
# Main Training Pipeline
# ──────────────────────────────────────────────────────────────────────────────
def run_training(data_path: str, repo_owner: str, repo_name: str, dagshub_token: str = "") -> None:
    # ── 0. DagsHub Init ───────────────────────────────────────────────────────
    log.info("=" * 60)
    log.info("  DIABETES PREDICTION — TRAINING (Workflow-CI)")
    log.info("=" * 60)
    log.info(f"[INIT] Connecting to DagsHub: {repo_owner}/{repo_name}")
    
    # 🎯 FIX MUTLAK: Gunakan token dari parameter terminal, jangan biarkan ditimpa env lain!
    if dagshub_token:
        os.environ["DAGSHUB_TOKEN"] = dagshub_token
        os.environ["MLFLOW_TRACKING_PASSWORD"] = dagshub_token
        
    os.environ["MLFLOW_TRACKING_USERNAME"] = repo_owner
    os.environ["MLFLOW_TRACKING_URI"] = f"https://dagshub.com/{repo_owner}/{repo_name}.mlflow"

    dagshub.init(repo_owner=repo_owner, repo_name=repo_name, mlflow=True)
    mlflow.set_experiment(EXPERIMENT)

    # ── 1. Load Data ──────────────────────────────────────────────────────────
    log.info(f"[STEP 1] Loading data: {data_path}")
    df = pd.read_csv(data_path)
    X  = df.drop(columns=[TARGET_COL])
    y  = df[TARGET_COL]
    feature_names = X.columns.tolist()
    log.info(f"         Shape: {df.shape[0]:,} × {df.shape[1]}")

    # ── 2. Split ──────────────────────────────────────────────────────────────
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )
    log.info(f"         Train: {len(X_train):,}  |  Test: {len(X_test):,}")

    # ── 3. Train ──────────────────────────────────────────────────────────────
    log.info("[STEP 3] Training Random Forest ...")
    model = RandomForestClassifier(**DEFAULT_PARAMS)
    model.fit(X_train, y_train)

    # ── 4. Evaluate ───────────────────────────────────────────────────────────
    y_pred  = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    metrics = {
        "accuracy" : round(accuracy_score(y_test, y_pred), 6),
        "precision": round(precision_score(y_test, y_pred, zero_division=0), 6),
        "recall"   : round(recall_score(y_test, y_pred, zero_division=0), 6),
        "f1_score" : round(f1_score(y_test, y_pred, zero_division=0), 6),
        "roc_auc"  : round(roc_auc_score(y_test, y_proba), 6),
    }
    for k, v in metrics.items():
        log.info(f"         {k:<12}: {v}")

    cm     = confusion_matrix(y_test, y_pred)
    report = classification_report(
        y_test, y_pred, target_names=["Non-Diabetes", "Diabetes"]
    )

    # ── 5. Artifacts ──────────────────────────────────────────────────────────
    os.makedirs("artifacts", exist_ok=True)
    cm_path     = "artifacts/confusion_matrix.png"
    fi_path     = "artifacts/feature_importance.png"
    report_path = "artifacts/classification_report.txt"

    plot_confusion_matrix(cm, cm_path)
    plot_feature_importance(model.feature_importances_, feature_names, fi_path)
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("=" * 55 + "\n")
        f.write("  CLASSIFICATION REPORT — Random Forest\n")
        f.write("=" * 55 + "\n\n")
        f.write(report)
        f.write("\n\nConfusion Matrix:\n")
        f.write(str(cm))

    # ── 6. MLflow Manual Logging ──────────────────────────────────────────────
    log.info("[STEP 6] Logging to MLflow / DagsHub ...")
    with mlflow.start_run(run_name="RandomForest-CI-Retrain") as run:
        run_id = run.info.run_id

        # Params
        mlflow.log_param("model_type",          "RandomForestClassifier")
        mlflow.log_param("n_estimators",        DEFAULT_PARAMS["n_estimators"])
        mlflow.log_param("max_depth",           str(DEFAULT_PARAMS["max_depth"]))
        mlflow.log_param("min_samples_split",   DEFAULT_PARAMS["min_samples_split"])
        mlflow.log_param("min_samples_leaf",    DEFAULT_PARAMS["min_samples_leaf"])
        mlflow.log_param("max_features",        DEFAULT_PARAMS["max_features"])
        mlflow.log_param("class_weight",        DEFAULT_PARAMS["class_weight"])
        mlflow.log_param("random_state",        RANDOM_STATE)
        mlflow.log_param("test_size",           TEST_SIZE)
        mlflow.log_param("train_samples",       len(X_train))
        mlflow.log_param("test_samples",        len(X_test))
        mlflow.log_param("n_features",          len(feature_names))
        mlflow.log_param("feature_names",       str(feature_names))

        # Metrics
        mlflow.log_metric("accuracy",  metrics["accuracy"])
        mlflow.log_metric("precision", metrics["precision"])
        mlflow.log_metric("recall",    metrics["recall"])
        mlflow.log_metric("f1_score",  metrics["f1_score"])
        mlflow.log_metric("roc_auc",   metrics["roc_auc"])

        # Artifacts
        mlflow.log_artifact(cm_path,     artifact_path="evaluation")
        mlflow.log_artifact(fi_path,     artifact_path="evaluation")
        mlflow.log_artifact(report_path, artifact_path="evaluation")

        # Model
        model_info = mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path="model",
            registered_model_name="DiabetesPrediction-RandomForest",
        )

        model_uri = model_info.model_uri
        log.info(f"         Run ID    : {run_id}")
        log.info(f"         Model URI : {model_uri}")

    # ── 7. Tulis run_id.txt & model_uri.txt ──────────────────────────────────
    with open("run_id.txt", "w") as f:
        f.write(run_id)
    with open("model_uri.txt", "w") as f:
        f.write(model_uri)

    log.info("[STEP 7] Metadata ditulis:")
    log.info(f"         run_id.txt   → {run_id}")
    log.info(f"         model_uri.txt→ {model_uri}")
    log.info("=" * 60)
    log.info("  TRAINING SELESAI ✅")
    log.info("=" * 60)


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Training pipeline (MLProject entry point).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--data", "-d",
        default="diabetes_prediction_preprocessing.csv",
        help="Path ke CSV hasil preprocessing.")
    parser.add_argument("--dagshub-repo-owner", required=True,
        help="DagsHub username.")
    parser.add_argument("--dagshub-repo-name", required=True,
        help="Nama repo DagsHub.")
    parser.add_argument("--dagshub-token", default="",
        help="Token akses DagsHub untuk otomatisasi headless.") 
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_training(
        data_path=args.data,
        repo_owner=args.dagshub_repo_owner,
        repo_name=args.dagshub_repo_name,
        dagshub_token=args.dagshub_token, # <-- Dioper langsung masuk memori parameter fungsi
    )