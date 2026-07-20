"""
Single, reusable preprocessing pipeline.

Why this replaces the notebook's manual steps:
- The notebook re-ran dropna / get_dummies / StandardScaler by hand in
  separate cells with global variables. That can't be serialized or reused
  at inference time, and is easy to run out of order.
- Here, everything (impute -> encode -> scale -> correlation-filter) is a
  single sklearn Pipeline/ColumnTransformer object. Call `.fit()` once at
  training time and `.transform()` (or let the full model pipeline do it)
  at inference time — guaranteed identical behavior in both places.
"""
from __future__ import annotations

import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from . import config


class LeakageColumnDropper(BaseEstimator, TransformerMixin):
    """Drops the raw target-encoding column and any one-hot columns
    derived from it (e.g. 'result_grade_A'). This is the one correct
    safety check the prototype already had — kept and made reusable."""

    def __init__(self, leakage_columns=None, leakage_prefix=None):
        self.leakage_columns = leakage_columns or config.LEAKAGE_COLUMNS
        self.leakage_prefix = leakage_prefix or config.LEAKAGE_PREFIX

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        drop_cols = [c for c in X.columns if c in self.leakage_columns
                     or c.startswith(self.leakage_prefix)]
        return X.drop(columns=drop_cols, errors="ignore")


class CorrelationFeatureSelector(BaseEstimator, TransformerMixin):
    """Keeps only columns whose absolute Pearson correlation with the
    target (measured on the training fold only) is >= threshold.

    This mirrors the notebook's manual correlation-threshold step, but as
    a fittable transformer: the selected columns are learned on TRAIN data
    and then simply sliced on any future data (test set, or a live API
    request) — no leakage, no copy-pasted logic.
    """

    def __init__(self, threshold: float = config.CORRELATION_THRESHOLD):
        self.threshold = threshold

    def fit(self, X: pd.DataFrame, y):
        y = pd.Series(y).reset_index(drop=True)
        X = X.reset_index(drop=True)
        corr = X.apply(lambda col: col.astype(float).corr(y.astype(float)))
        self.selected_columns_ = corr[corr.abs() >= self.threshold].index.tolist()
        if not self.selected_columns_:
            # Safety net: never end up with zero features
            self.selected_columns_ = list(X.columns)
        return self

    def transform(self, X: pd.DataFrame):
        return X[self.selected_columns_]


def build_column_transformer() -> ColumnTransformer:
    numeric_pipe = Pipeline(steps=[
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
    ])
    categorical_pipe = Pipeline(steps=[
        ("impute", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])
    ct = ColumnTransformer(
        transformers=[
            ("num", numeric_pipe, config.NUMERIC_FEATURES),
            ("cat", categorical_pipe, config.CATEGORICAL_FEATURES),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )
    ct.set_output(transform="pandas")
    return ct


def build_preprocessing_pipeline(use_correlation_filter: bool = True) -> Pipeline:
    """Full preprocessing pipeline: drop leakage cols -> impute/encode/scale
    -> (optional) correlation-based feature selection.
    Returns raw, model-ready features. Does NOT include the model itself,
    so it can be unit-tested / reused independently.
    """
    steps = [
        ("drop_leakage", LeakageColumnDropper()),
        ("transform", build_column_transformer()),
    ]
    if use_correlation_filter:
        steps.append(("select", CorrelationFeatureSelector()))
    return Pipeline(steps=steps)


def load_raw(path, sep: str = config.CSV_SEP) -> pd.DataFrame:
    df = pd.read_csv(path, sep=sep)
    df = df.dropna(subset=[config.TARGET_COL])  # target must never be null
    return df


def split_features_target(df: pd.DataFrame):
    y = df[config.TARGET_COL].astype(int)
    X = df.drop(columns=[config.TARGET_COL])
    return X, y
