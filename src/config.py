"""
Central configuration: paths, feature schema, and constants.
Keeping this in one place means preprocessing, training, and the API
all agree on the same column names — the #1 cause of prod/notebook drift.
"""
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "data"
MODELS_DIR = ROOT_DIR / "models"
MODELS_DIR.mkdir(exist_ok=True, parents=True)

TRAIN_CSV = DATA_DIR / "DBS.csv"
TEST_CSV = DATA_DIR / "DBS_2020.csv"
CSV_SEP = ";"

TARGET_COL = "graduate"

# Raw numeric columns (counts / continuous scores)
NUMERIC_FEATURES = [
    "access", "tests", "exam", "project", "assignments", "result_points",
]

# Raw categorical columns (letter-grade style features)
CATEGORICAL_FEATURES = ["tests_grade", "project_grade"]

# Columns that leak the target and must NEVER be used as model input
# (result_grade is a direct encoding of pass/fail; result_grade_* one-hot
#  columns derived from it must also be dropped)
LEAKAGE_COLUMNS = ["result_grade"]
LEAKAGE_PREFIX = "result_grade_"

# Correlation threshold used for optional secondary feature filtering
CORRELATION_THRESHOLD = 0.10

RANDOM_STATE = 42
N_SPLITS_CV = 5
TEST_SIZE = 0.2

ARTIFACT_PATH = MODELS_DIR / "dropout_stack_pipeline.joblib"
METRICS_PATH = MODELS_DIR / "metrics.json"
FEATURE_IMPORTANCE_PATH = MODELS_DIR / "feature_importance.json"
