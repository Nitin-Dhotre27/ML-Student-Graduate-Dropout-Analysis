"""
Layered stacking training pipeline.

Pipeline-by-layer design:
  Layer 0 (preprocessing): impute -> encode -> scale -> correlation filter
  Layer 1 (base learners):  RandomForest, XGBoost, LightGBM
                             — each independently fine-tuned with Optuna
                             (Bayesian/TPE search), 5-fold stratified CV,
                             optimizing ROC-AUC.
  Layer 2 (meta-learner):   MLPClassifier trained on the layer-1 models'
                             cross-validated out-of-fold probabilities
                             (via sklearn's StackingClassifier, which
                             handles the CV-to-avoid-leakage step
                             automatically and correctly — the exact
                             mechanism the original paper describes).

Everything (Layer 0 + Layer 1 + Layer 2) is fit inside ONE sklearn Pipeline
object and saved as a single .joblib artifact, so training and serving
can never drift apart.

Class imbalance is handled via class_weight='balanced' (RF, meta-learner)
and scale_pos_weight (XGBoost/LightGBM), computed from the training fold
only. This is used instead of SMOTE because SMOTE combined with
StackingClassifier's internal cross-validation risks leaking synthetic
samples across folds — a subtle bug the prototype notebook was exposed to
via its inconsistent SMOTE usage (see validation notes).

Run:
    python -m src.train
"""
from __future__ import annotations

import json
import time

import joblib
import numpy as np
import optuna
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.metrics import (accuracy_score, f1_score, precision_score,
                              recall_score, roc_auc_score)
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

from . import config
from .preprocessing import build_preprocessing_pipeline, load_raw, split_features_target

optuna.logging.set_verbosity(optuna.logging.WARNING)


def _cv():
    return StratifiedKFold(n_splits=config.N_SPLITS_CV, shuffle=True,
                            random_state=config.RANDOM_STATE)


def tune_random_forest(X, y, n_trials=20):
    def objective(trial):
        params = dict(
            n_estimators=trial.suggest_int("n_estimators", 200, 1000, step=100),
            max_depth=trial.suggest_int("max_depth", 3, 14),
            min_samples_split=trial.suggest_int("min_samples_split", 2, 10),
            min_samples_leaf=trial.suggest_int("min_samples_leaf", 1, 6),
            class_weight="balanced",
            random_state=config.RANDOM_STATE,
            n_jobs=-1,
        )
        model = RandomForestClassifier(**params)
        return cross_val_score(model, X, y, cv=_cv(), scoring="roc_auc", n_jobs=-1).mean()

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    best = study.best_params
    best.update(class_weight="balanced", random_state=config.RANDOM_STATE, n_jobs=-1)
    return RandomForestClassifier(**best), study.best_value


def tune_xgboost(X, y, n_trials=20):
    pos = y.sum()
    neg = len(y) - pos
    spw = (neg / pos) if pos > 0 else 1.0

    def objective(trial):
        params = dict(
            n_estimators=trial.suggest_int("n_estimators", 200, 1000, step=100),
            max_depth=trial.suggest_int("max_depth", 3, 10),
            learning_rate=trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
            subsample=trial.suggest_float("subsample", 0.6, 1.0),
            colsample_bytree=trial.suggest_float("colsample_bytree", 0.6, 1.0),
            scale_pos_weight=spw,
            eval_metric="logloss",
            random_state=config.RANDOM_STATE,
            n_jobs=-1,
        )
        model = XGBClassifier(**params)
        return cross_val_score(model, X, y, cv=_cv(), scoring="roc_auc", n_jobs=-1).mean()

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    best = study.best_params
    best.update(scale_pos_weight=spw, eval_metric="logloss",
                random_state=config.RANDOM_STATE, n_jobs=-1)
    return XGBClassifier(**best), study.best_value


def tune_lightgbm(X, y, n_trials=20):
    """LightGBM: histogram-based gradient boosting. Trains substantially
    faster than plain GradientBoosting/XGBoost at scale (leaf-wise growth +
    native categorical/na handling), which is why it replaces the
    notebook's/paper's plain GradientBoosting layer for a more scalable
    deployment."""
    pos = y.sum()
    neg = len(y) - pos
    spw = (neg / pos) if pos > 0 else 1.0

    def objective(trial):
        params = dict(
            n_estimators=trial.suggest_int("n_estimators", 200, 1000, step=100),
            max_depth=trial.suggest_int("max_depth", 3, 12),
            learning_rate=trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
            num_leaves=trial.suggest_int("num_leaves", 15, 127),
            subsample=trial.suggest_float("subsample", 0.6, 1.0),
            scale_pos_weight=spw,
            random_state=config.RANDOM_STATE,
            n_jobs=-1,
            verbosity=-1,
        )
        model = LGBMClassifier(**params)
        return cross_val_score(model, X, y, cv=_cv(), scoring="roc_auc", n_jobs=-1).mean()

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    best = study.best_params
    best.update(scale_pos_weight=spw, random_state=config.RANDOM_STATE,
                n_jobs=-1, verbosity=-1)
    return LGBMClassifier(**best), study.best_value


def build_stacking_model(X_pre: pd.DataFrame, y: pd.Series, n_trials: int = 20):
    print("Layer 1 — tuning Random Forest...")
    rf, rf_score = tune_random_forest(X_pre, y, n_trials)
    print(f"  best CV ROC-AUC: {rf_score:.4f}")

    print("Layer 1 — tuning XGBoost...")
    xgb, xgb_score = tune_xgboost(X_pre, y, n_trials)
    print(f"  best CV ROC-AUC: {xgb_score:.4f}")

    print("Layer 1 — tuning LightGBM...")
    lgbm, lgbm_score = tune_lightgbm(X_pre, y, n_trials)
    print(f"  best CV ROC-AUC: {lgbm_score:.4f}")

    print("Layer 2 — assembling stacked meta-learner (FNN)...")
    meta_learner = MLPClassifier(
        hidden_layer_sizes=(16, 8),
        activation="relu",
        solver="adam",
        alpha=1e-3,
        max_iter=3000,
        random_state=config.RANDOM_STATE,
    )

    stack = StackingClassifier(
        estimators=[("rf", rf), ("xgb", xgb), ("lgbm", lgbm)],
        final_estimator=meta_learner,
        cv=config.N_SPLITS_CV,   # generates out-of-fold predictions -> no leakage
        stack_method="predict_proba",
        n_jobs=-1,
        passthrough=False,
    )
    return stack


def evaluate(pipeline: Pipeline, X_test, y_test) -> dict:
    y_pred = pipeline.predict(X_test)
    y_proba = pipeline.predict_proba(X_test)[:, 1]
    return {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred, zero_division=0),
        "f1_score": f1_score(y_test, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_test, y_proba),
    }


def main(n_trials: int = 20):
    t0 = time.time()

    train_df = load_raw(config.TRAIN_CSV)
    X_train_raw, y_train = split_features_target(train_df)

    if config.TEST_CSV.exists():
        test_df = load_raw(config.TEST_CSV)
        X_test_raw, y_test = split_features_target(test_df)
    else:
        X_train_raw, X_test_raw, y_train, y_test = train_test_split(
            X_train_raw, y_train, test_size=config.TEST_SIZE,
            stratify=y_train, random_state=config.RANDOM_STATE,
        )

    preprocessor = build_preprocessing_pipeline(use_correlation_filter=True)
    X_train_pre = preprocessor.fit_transform(X_train_raw, y_train)
    print(f"Selected {X_train_pre.shape[1]} features after correlation filter: "
          f"{list(X_train_pre.columns)}")

    stack = build_stacking_model(X_train_pre, y_train, n_trials=n_trials)

    full_pipeline = Pipeline(steps=[
        ("preprocess", preprocessor),
        ("stack", stack),
    ])

    print("Fitting full pipeline on training data...")
    full_pipeline.fit(X_train_raw, y_train)

    metrics = evaluate(full_pipeline, X_test_raw, y_test)
    metrics["train_seconds"] = round(time.time() - t0, 1)
    metrics["n_train"] = int(len(y_train))
    metrics["n_test"] = int(len(y_test))
    print("Test metrics:", json.dumps(metrics, indent=2))

    joblib.dump(full_pipeline, config.ARTIFACT_PATH)
    with open(config.METRICS_PATH, "w") as f:
        json.dump(metrics, f, indent=2)

    # Feature importances from the RF base learner for explainability in the UI
    try:
        rf_model = full_pipeline.named_steps["stack"].named_estimators_["rf"]
        importances = dict(zip(X_train_pre.columns, rf_model.feature_importances_.tolist()))
        importances = dict(sorted(importances.items(), key=lambda kv: -kv[1]))
        with open(config.FEATURE_IMPORTANCE_PATH, "w") as f:
            json.dump(importances, f, indent=2)
    except Exception as e:
        print("Could not extract feature importances:", e)

    print(f"Saved model to {config.ARTIFACT_PATH}")
    return full_pipeline, metrics


if __name__ == "__main__":
    main()
