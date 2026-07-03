"""
model_loader.py – Thread-safe lazy loading of ML artefacts for the Flask app.
"""

from __future__ import annotations

import threading
from pathlib import Path
from typing import Any

import joblib

_lock = threading.Lock()
_cache: dict[str, Any] = {}

MODELS_DIR = Path(__file__).resolve().parent.parent / "models"
REPORTS_DIR = Path(__file__).resolve().parent.parent / "reports"


def _load(key: str, filename: str) -> Any:
    """Load an artefact from disk once and cache it."""
    if key not in _cache:
        with _lock:
            if key not in _cache:
                path = MODELS_DIR / filename
                if not path.exists():
                    raise FileNotFoundError(
                        f"Artefact not found: {path}\n"
                        "Run `python pipeline.py` first to train the model."
                    )
                _cache[key] = joblib.load(path)
    return _cache[key]


def get_model() -> Any:
    """Return the cached trained model."""
    return _load("model", "trained_model.pkl")


def get_scaler() -> Any:
    """Return the cached StandardScaler."""
    return _load("scaler", "scaler.pkl")


def get_label_encoder() -> Any:
    """Return the cached LabelEncoder."""
    return _load("le", "label_encoder.pkl")


def models_ready() -> bool:
    """Return True if all required model files exist on disk."""
    return all(
        (MODELS_DIR / f).exists()
        for f in ("trained_model.pkl", "scaler.pkl", "label_encoder.pkl")
    )
