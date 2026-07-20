"""
FastAPI backend for the student dropout predictor.

Run locally:
    cd dropout-predictor
    uvicorn backend.app:app --reload --port 8000

Endpoints:
    GET  /health            -> liveness check
    GET  /metrics           -> last training run's test metrics
    GET  /feature-importance -> RF feature importances (for the UI chart)
    POST /predict            -> single-student prediction
    POST /predict/batch      -> list of students -> list of predictions
"""
from __future__ import annotations

from typing import List, Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from src import config
from src.inference import predict_one, get_feature_importance

app = FastAPI(
    title="Student Dropout Prediction API",
    description="Two-layer stacking ensemble (RF + XGBoost + LightGBM -> FNN meta-learner)",
    version="1.0.0",
)

# Allow the static frontend (opened from file:// or a dev server) to call this API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class StudentFeatures(BaseModel):
    access: int = Field(..., ge=0, description="Number of course-portal accesses")
    tests: int = Field(..., ge=0, description="Number of tests taken")
    tests_grade: str = Field(..., description="Letter grade for tests, e.g. A/B/C/D/E/FX")
    exam: float = Field(..., ge=0, le=100, description="Exam score (0-100)")
    project: int = Field(..., ge=0, description="Number of project milestones completed")
    project_grade: str = Field(..., description="Letter grade for project, e.g. A/B/C/D/E/FX")
    assignments: int = Field(..., ge=0, description="Number of assignments submitted")
    result_points: float = Field(..., ge=0, le=100, description="Aggregate result points (0-100)")

    class Config:
        json_schema_extra = {
            "example": {
                "access": 120, "tests": 6, "tests_grade": "B", "exam": 72.5,
                "project": 3, "project_grade": "A", "assignments": 10,
                "result_points": 78.0,
            }
        }


class PredictionResponse(BaseModel):
    prediction: str
    probability_non_dropout: float
    probability_dropout: float
    risk_level: str


@app.get("/health")
def health():
    model_ready = config.ARTIFACT_PATH.exists()
    return {"status": "ok", "model_ready": model_ready}


@app.get("/metrics")
def metrics():
    if not config.METRICS_PATH.exists():
        raise HTTPException(404, "No metrics found. Train the model first.")
    import json
    with open(config.METRICS_PATH) as f:
        return json.load(f)


@app.get("/feature-importance")
def feature_importance():
    return get_feature_importance()


@app.post("/predict", response_model=PredictionResponse)
def predict(student: StudentFeatures):
    try:
        return predict_one(student.model_dump())
    except FileNotFoundError as e:
        raise HTTPException(503, str(e))
    except Exception as e:
        raise HTTPException(400, f"Prediction failed: {e}")


@app.post("/predict/batch", response_model=List[PredictionResponse])
def predict_batch(students: List[StudentFeatures]):
    try:
        return [predict_one(s.model_dump()) for s in students]
    except FileNotFoundError as e:
        raise HTTPException(503, str(e))
