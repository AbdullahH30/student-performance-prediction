"""
routes.py – All Flask routes for the Student Performance Prediction web app.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from flask import Flask, jsonify, render_template, request, send_from_directory

from app.model_loader import get_label_encoder, get_model, get_scaler, models_ready

logger = logging.getLogger(__name__)

REPORTS_DIR = Path(__file__).resolve().parent.parent / "reports"
DATA_DIR = Path(__file__).resolve().parent.parent / "data"

CLASS_LABELS = ["Low", "Medium", "High"]
CLASS_COLORS = {"Low": "#ef4444", "Medium": "#f59e0b", "High": "#22c55e"}

BINARY_MAPS = {
    "gender": {"Male": 0, "Female": 1},
    "internet_access": {"No": 0, "Yes": 1},
    "extracurricular": {"No": 0, "Yes": 1},
    "tutoring": {"No": 0, "Yes": 1},
    "school_type": {"Public": 0, "Private": 1},
}
ORDINAL_MAPS = {
    "parental_education": {"None": 0, "High School": 1, "College": 2, "Postgraduate": 3},
    "family_support": {"Low": 0, "Medium": 1, "High": 2},
    "motivation_level": {"Low": 0, "Medium": 1, "High": 2},
    "distance_from_home": {"Near": 0, "Moderate": 1, "Far": 2},
}
FEATURE_ORDER = [
    "gender", "age", "school_type", "distance_from_home",
    "parental_education", "family_support", "internet_access",
    "study_hours", "attendance", "absences", "previous_grades",
    "sleep_hours", "extracurricular", "tutoring", "motivation_level",
]


def _encode_input(form_data: dict) -> np.ndarray:
    """Encode raw form data into a scaled numpy vector for inference."""
    row: dict = {}
    for key in FEATURE_ORDER:
        val = form_data.get(key, 0)
        if key in BINARY_MAPS:
            row[key] = BINARY_MAPS[key].get(str(val), 0)
        elif key in ORDINAL_MAPS:
            row[key] = ORDINAL_MAPS[key].get(str(val), 0)
        else:
            try:
                row[key] = float(val)
            except (ValueError, TypeError):
                row[key] = 0.0

    X = np.array([[row[k] for k in FEATURE_ORDER]], dtype=float)
    scaler = get_scaler()
    return scaler.transform(X)


def _load_json(filename: str) -> dict | list:
    path = REPORTS_DIR / filename
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return {}


def register_routes(app: Flask) -> None:
    """Attach all URL rules to *app*."""

    # ── Health check ──────────────────────────────────────────────────────
    @app.route("/health")
    def health():
        return jsonify({"status": "ok", "models_ready": models_ready()})

    # ── Home ──────────────────────────────────────────────────────────────
    @app.route("/")
    def index():
        metrics = _load_json("test_metrics.json")
        comparison = _load_json("model_comparison.json")
        return render_template(
            "index.html",
            metrics=metrics,
            comparison=comparison[:5] if comparison else [],
            models_ready=models_ready(),
        )

    # ── About ─────────────────────────────────────────────────────────────
    @app.route("/about")
    def about():
        return render_template("about.html")

    # ── Dataset ───────────────────────────────────────────────────────────
    @app.route("/dataset")
    def dataset():
        csv_path = DATA_DIR / "student_performance.csv"
        stats = {}
        preview = []
        if csv_path.exists():
            df = pd.read_csv(csv_path)
            stats = {
                "rows": len(df),
                "cols": df.shape[1],
                "classes": df["performance"].value_counts().to_dict() if "performance" in df else {},
                "missing": int(df.isnull().sum().sum()),
            }
            preview = df.head(8).to_dict(orient="records")
        return render_template("dataset.html", stats=stats, preview=preview)

    # ── Model performance ─────────────────────────────────────────────────
    @app.route("/performance")
    def performance():
        metrics = _load_json("test_metrics.json")
        comparison = _load_json("model_comparison.json")
        shap_imp = _load_json("shap_importance.json")
        return render_template(
            "performance.html",
            metrics=metrics,
            comparison=comparison,
            shap_importance=shap_imp,
        )

    # ── Visualisations ────────────────────────────────────────────────────
    @app.route("/visualizations")
    def visualizations():
        plots = [
            {"name": "Target Distribution", "file": "target_distribution.png"},
            {"name": "Attendance Distribution", "file": "attendance_distribution.png"},
            {"name": "Study Hours Distribution", "file": "study_hours_distribution.png"},
            {"name": "Grade Distribution", "file": "grade_distribution.png"},
            {"name": "Correlation Heatmap", "file": "correlation_heatmap.png"},
            {"name": "Boxplots", "file": "boxplots.png"},
            {"name": "Categorical Features", "file": "categorical_features.png"},
            {"name": "Confusion Matrix", "file": "confusion_matrix.png"},
            {"name": "ROC Curves", "file": "roc_curves.png"},
            {"name": "Learning Curve", "file": "learning_curve.png"},
            {"name": "Feature Importance", "file": "feature_importance.png"},
            {"name": "Permutation Importance", "file": "permutation_importance.png"},
            {"name": "SHAP Summary", "file": "shap_summary.png"},
            {"name": "SHAP Bar Chart", "file": "shap_bar.png"},
            {"name": "SHAP Waterfall", "file": "shap_waterfall.png"},
            {"name": "Model Comparison", "file": "model_comparison.png"},
            {"name": "Pairplot", "file": "pairplot.png"},
        ]
        available = [p for p in plots if (REPORTS_DIR / p["file"]).exists()]
        return render_template("visualizations.html", plots=available)

    # ── Serve report images ───────────────────────────────────────────────
    @app.route("/reports/<path:filename>")
    def report_image(filename: str):
        return send_from_directory(str(REPORTS_DIR), filename)

    # ── Prediction page ───────────────────────────────────────────────────
    @app.route("/predict", methods=["GET", "POST"])
    def predict():
        result = None
        error = None
        form_data = {}

        if request.method == "POST":
            form_data = request.form.to_dict()
            if not models_ready():
                error = "Model not found. Please run `python pipeline.py` first."
            else:
                try:
                    X = _encode_input(form_data)
                    model = get_model()
                    pred_int = int(model.predict(X)[0])
                    pred_class = CLASS_LABELS[pred_int]
                    proba = model.predict_proba(X)[0].tolist()
                    confidence = max(proba)

                    result = {
                        "predicted_class": pred_class,
                        "color": CLASS_COLORS[pred_class],
                        "probabilities": {
                            label: round(p * 100, 1)
                            for label, p in zip(CLASS_LABELS, proba)
                        },
                        "confidence": round(confidence * 100, 1),
                    }
                    logger.info("Prediction: %s (%.1f%%)", pred_class, confidence * 100)
                except Exception as exc:
                    error = f"Prediction error: {exc}"
                    logger.exception("Prediction failed.")

        return render_template("predict.html", result=result, error=error, form_data=form_data)

    # ── REST API: POST /api/predict ───────────────────────────────────────
    @app.route("/api/predict", methods=["POST"])
    def api_predict():
        if not models_ready():
            return jsonify({"error": "Model not ready. Run pipeline.py first."}), 503

        payload = request.get_json(silent=True)
        if not payload:
            return jsonify({"error": "Invalid JSON payload."}), 400

        try:
            X = _encode_input(payload)
            model = get_model()
            pred_int = int(model.predict(X)[0])
            pred_class = CLASS_LABELS[pred_int]
            proba = model.predict_proba(X)[0].tolist()

            return jsonify(
                {
                    "predicted_class": pred_class,
                    "probabilities": {
                        label: round(p, 4) for label, p in zip(CLASS_LABELS, proba)
                    },
                    "confidence": round(max(proba), 4),
                }
            )
        except Exception as exc:
            logger.exception("API prediction failed.")
            return jsonify({"error": str(exc)}), 500

    # ── Contact ───────────────────────────────────────────────────────────
    @app.route("/contact")
    def contact():
        return render_template("contact.html")
