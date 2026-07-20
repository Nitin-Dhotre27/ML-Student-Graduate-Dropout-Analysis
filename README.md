# Student Dropout Predictor — Production-Ready Pipeline

Rebuilt from the `preProcessing.ipynb` prototype into a deployable system:
a single scikit-learn/imbalanced-learn pipeline, a layered stacking model
(RF + XGBoost + LightGBM → FNN meta-learner) with per-layer Optuna
fine-tuning, a FastAPI backend, and a usable web client.

## What changed from the prototype, and why

| Prototype issue | Fix in this version |
|---|---|
| `df1.columns` undefined variable (crashes) | Removed; all steps live inside one `Pipeline` object, no stray globals |
| `eval_matric` typo, `learning-rate` (hyphen) invalid grid key | Correct param names in `src/train.py`, validated by an actual training run |
| SMOTE applied inconsistently (some models, not others) → unfair ROC comparison | Replaced with `class_weight='balanced'` / `scale_pos_weight`, applied uniformly and computed from the training fold only — also avoids a subtle SMOTE + cross-validated-stacking leakage risk |
| Manual `dropna` / `get_dummies` / `StandardScaler` re-run in separate cells | One `ColumnTransformer`-based `Pipeline` (`src/preprocessing.py`), fit once, reused identically at inference time |
| No true stacking — 6 independent models compared by ROC curve only | Real 2-layer `StackingClassifier`: Layer 1 = tuned RF/XGBoost/LightGBM, Layer 2 = MLP meta-learner trained on out-of-fold Layer-1 probabilities |
| No persistence / no serving path | `joblib`-serialized single artifact + FastAPI `/predict` endpoint + web client |

## Architecture

```
data/DBS.csv, DBS_2020.csv
        │
        ▼
Layer 0  preprocessing.py   impute → one-hot encode → scale → correlation filter
        │
        ▼
Layer 1  train.py           RandomForest, XGBoost, LightGBM
                             (each independently Optuna-tuned, 5-fold CV, ROC-AUC objective)
        │  (out-of-fold predicted probabilities)
        ▼
Layer 2  train.py           MLPClassifier meta-learner (StackingClassifier)
        │
        ▼
models/dropout_stack_pipeline.joblib   ← single deployable artifact
        │
        ▼
backend/app.py (FastAPI)   /predict  /predict/batch  /metrics  /feature-importance
        │
        ▼
frontend/index.html         Form → risk badge, probability, feature-importance chart
```

**Why LightGBM was added:** it's the natural "recent, more scalable" upgrade
over the paper's plain `GradientBoosting` — histogram-based, leaf-wise growth,
native handling of larger feature counts, and materially faster to fine-tune
across many Optuna trials than sklearn's `GradientBoostingClassifier`.

**Why Optuna instead of `GridSearchCV`:** grid search scales combinatorially
with the number of hyperparameters; Optuna's Bayesian/TPE sampler finds
strong parameters in far fewer trials, which matters once you're tuning
three base learners independently.

## Setup

```bash
cd dropout-predictor
pip install -r requirements.txt
```

## 1. Train

```bash
python -m src.generate_synthetic_data   # only needed once, until you add real CSVs — see data/README.md
python -m src.train
```
This writes `models/dropout_stack_pipeline.joblib`, `models/metrics.json`,
and `models/feature_importance.json`.

## 2. Run the backend

```bash
uvicorn backend.app:app --reload --port 8000
```
Interactive API docs: http://localhost:8000/docs

## 3. Run the frontend

```bash
cd frontend
python -m http.server 5500
```
Open http://localhost:5500 — it talks to the backend at `http://localhost:8000`
(editable in the input box at the top of the page if you deploy the backend elsewhere).

## Or run everything with Docker

```bash
docker compose up --build
```
Backend on `:8000`, frontend on `:5500`.

## Extending further
- Swap `MLPClassifier` for a Keras/TensorFlow FNN if you need more control
  over the meta-learner's architecture — the `StackingClassifier` API accepts
  any scikit-learn-compatible estimator, including a Keras model wrapped with
  `scikeras.wrappers.KerasClassifier`.
- Add `CatBoost` as a fourth Layer-1 learner the same way RF/XGBoost/LightGBM
  were added in `src/train.py` — it handles categorical columns natively and
  is worth comparing against the one-hot + LightGBM approach used here.
- For real scale (thousands+ of students, frequent retraining), move
  `models/` to object storage (S3/GCS) and have the backend load the latest
  artifact by version tag instead of a fixed local path.
