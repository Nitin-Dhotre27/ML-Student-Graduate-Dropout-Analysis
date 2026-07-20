"""Loads the saved pipeline once and exposes a single predict() function."""
from __future__ import annotations

import json
from functools import lru_cache

import joblib
import pandas as pd

from . import config


@lru_cache(maxsize=1)
def get_pipeline():
    if not config.ARTIFACT_PATH.exists():
        raise FileNotFoundError(
            f"No trained model found at {config.ARTIFACT_PATH}. "
            f"Run `python -m src.train` first."
        )
    return joblib.load(config.ARTIFACT_PATH)


@lru_cache(maxsize=1)
def get_feature_importance() -> dict:
    if config.FEATURE_IMPORTANCE_PATH.exists():
        with open(config.FEATURE_IMPORTANCE_PATH) as f:
            return json.load(f)
    return {}


def predict_one(record: dict) -> dict:
    """record: dict with the raw feature names (access, tests, tests_grade,
    exam, project, project_grade, assignments, result_points)."""
    pipeline = get_pipeline()
    df = pd.DataFrame([record])
    proba_non_dropout = float(pipeline.predict_proba(df)[0, 1])
    label = "non_dropout" if proba_non_dropout >= 0.5 else "dropout"
    risk_level = (
        "low" if proba_non_dropout >= 0.75 else
        "medium" if proba_non_dropout >= 0.5 else
        "high" if proba_non_dropout >= 0.25 else
        "critical"
    )
    return {
        "prediction": label,
        "probability_non_dropout": round(proba_non_dropout, 4),
        "probability_dropout": round(1 - proba_non_dropout, 4),
        "risk_level": risk_level,
    }
