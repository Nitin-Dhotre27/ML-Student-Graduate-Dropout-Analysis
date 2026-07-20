"""
Generates placeholder DBS.csv / DBS_2020.csv files matching the schema used
in the original notebook, so the pipeline below is runnable end-to-end
immediately.

IMPORTANT: this is synthetic data for demonstration only. Replace the files
in data/DBS.csv and data/DBS_2020.csv with your real exports (semicolon-
separated, same column names) and re-run `python -m src.train` to get a
model trained on real data.
"""
import numpy as np
import pandas as pd

from . import config

GRADES = ["A", "B", "C", "D", "E", "FX"]
GRADE_WEIGHT = [0.20, 0.20, 0.20, 0.15, 0.10, 0.15]  # FX = fail


def _make_split(n: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    access = rng.integers(0, 250, n)
    tests = rng.integers(0, 10, n)
    exam = rng.uniform(0, 100, n)
    project = rng.integers(0, 5, n)
    assignments = rng.integers(0, 15, n)

    tests_grade = rng.choice(GRADES, n, p=GRADE_WEIGHT)
    project_grade = rng.choice(GRADES, n, p=GRADE_WEIGHT)

    grade_to_score = {"A": 95, "B": 85, "C": 75, "D": 65, "E": 55, "FX": 30}
    result_points = (
        0.35 * exam
        + 0.25 * np.vectorize(grade_to_score.get)(tests_grade)
        + 0.25 * np.vectorize(grade_to_score.get)(project_grade)
        + 0.15 * (assignments / 15 * 100)
        + rng.normal(0, 5, n)
    ).clip(0, 100)

    def points_to_grade(p):
        if p >= 90:
            return "A"
        if p >= 80:
            return "B"
        if p >= 70:
            return "C"
        if p >= 60:
            return "D"
        if p >= 50:
            return "E"
        return "FX"

    result_grade = np.array([points_to_grade(p) for p in result_points])
    graduate = (result_grade != "FX").astype(int)

    return pd.DataFrame({
        "access": access,
        "tests": tests,
        "tests_grade": tests_grade,
        "exam": exam,
        "project": project,
        "project_grade": project_grade,
        "assignments": assignments,
        "result_points": result_points,
        "result_grade": result_grade,
        "graduate": graduate,
        "year": rng.integers(1, 4, n),
        "acad_year": rng.choice(["2016/17", "2017/18", "2018/19", "2019/20"], n),
    })


def main():
    config.DATA_DIR.mkdir(exist_ok=True, parents=True)
    train_df = _make_split(400, seed=1)
    test_df = _make_split(120, seed=2)
    train_df.to_csv(config.TRAIN_CSV, sep=config.CSV_SEP, index=False)
    test_df.to_csv(config.TEST_CSV, sep=config.CSV_SEP, index=False)
    print(f"Wrote {config.TRAIN_CSV} ({len(train_df)} rows)")
    print(f"Wrote {config.TEST_CSV} ({len(test_df)} rows)")


if __name__ == "__main__":
    main()
