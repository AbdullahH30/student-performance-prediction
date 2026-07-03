"""
pipeline.py – Master script that runs every step in the correct order.

Usage:
    python pipeline.py

Steps executed:
    1. Generate synthetic dataset
    2. EDA plots
    3. Preprocessing + feature engineering
    4. Train all 9 models + hyperparameter tuning
    5. Evaluate best model
    6. SHAP explainability
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

# Ensure project root is on sys.path when run directly
sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.utils import get_logger

logger = get_logger("pipeline")


def run() -> None:
    t_start = time.time()

    # ── 1. Dataset ─────────────────────────────────────────────────────────
    logger.info("=" * 60)
    logger.info("STEP 1/6 – Generating dataset")
    import data.generate_data as gd
    gd.generate_dataset().to_csv("data/student_performance.csv", index=False)
    logger.info("Dataset saved.")

    # ── 2. EDA ─────────────────────────────────────────────────────────────
    logger.info("=" * 60)
    logger.info("STEP 2/6 – Exploratory Data Analysis")
    from src.eda import run_eda
    run_eda()

    # ── 3. Preprocessing ───────────────────────────────────────────────────
    logger.info("=" * 60)
    logger.info("STEP 3/6 – Preprocessing")
    from src.preprocessing import run_preprocessing
    (
        X_train_sc, X_val_sc, X_test_sc,
        y_train, y_val, y_test,
        feature_names, le, scaler,
    ) = run_preprocessing()

    # ── 4. Training ────────────────────────────────────────────────────────
    logger.info("=" * 60)
    logger.info("STEP 4/6 – Training all models")
    from src.train import train_all_models
    best_model, comparison_df = train_all_models(
        X_train_sc, y_train, X_val_sc, y_val
    )

    # ── 5. Evaluation ──────────────────────────────────────────────────────
    logger.info("=" * 60)
    logger.info("STEP 5/6 – Model evaluation")
    import numpy as np
    X_trainval = np.vstack([X_train_sc, X_val_sc])
    y_trainval = np.concatenate([y_train, y_val])
    from src.evaluate import run_evaluation
    metrics = run_evaluation(
        best_model,
        X_trainval, y_trainval,
        X_test_sc, y_test,
        feature_names,
        comparison_df,
    )
    logger.info("Test metrics: %s", metrics)

    # ── 6. Explainability ──────────────────────────────────────────────────
    logger.info("=" * 60)
    logger.info("STEP 6/6 – SHAP explainability")
    from src.explainability import run_explainability
    run_explainability(best_model, X_train_sc, X_test_sc, feature_names)

    elapsed = time.time() - t_start
    logger.info("=" * 60)
    logger.info("✅  Full pipeline complete in %.1f seconds.", elapsed)
    logger.info("Run `flask run` from the app/ directory to start the web server.")


if __name__ == "__main__":
    run()
