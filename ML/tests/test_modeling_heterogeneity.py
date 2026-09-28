import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import indusense.modeling.heterogeneity as het
from indusense.modeling.heterogeneity import leave_one_group_out_proba, pr_auc_by

PARAMS = {"n_estimators": 10, "max_depth": 2, "random_state": 42, "verbosity": 0}


@pytest.fixture
def df():
    rng = np.random.default_rng(42)
    n = 300
    out = pd.DataFrame(
        {
            "machine_id": np.repeat(["M1", "M2", "M3"], n // 3),
            "f1": rng.normal(size=n),
            "f2": rng.normal(size=n),
        }
    )
    out["y"] = (out["f1"] + rng.normal(scale=0.3, size=n) > 1).astype(int)
    return out


def test_every_row_is_scored(df):
    proba = leave_one_group_out_proba(df, ["f1", "f2"], "y", "machine_id", PARAMS)
    assert proba.notna().all()
    assert proba.between(0, 1).all()


def test_held_out_group_is_never_in_training(df, monkeypatch):
    seen = []
    real_build = het.build_xgb_pipeline

    def spy(params):
        pipe = real_build(params)
        real_fit = pipe.fit

        def fit(X, y):
            seen.append(set(df.loc[X.index, "machine_id"]))
            return real_fit(X, y)

        pipe.fit = fit
        return pipe

    monkeypatch.setattr(het, "build_xgb_pipeline", spy)
    leave_one_group_out_proba(df, ["f1", "f2"], "y", "machine_id", PARAMS)
    assert seen == [{"M2", "M3"}, {"M1", "M3"}, {"M1", "M2"}]


def test_pr_auc_by_is_nan_without_positive(df):
    df.loc[df["machine_id"] == "M3", "y"] = 0
    proba = pd.Series(np.linspace(0, 1, len(df)), index=df.index)
    scores = pr_auc_by(df, "y", proba, "machine_id")
    assert np.isnan(scores["M3"])
    assert scores[["M1", "M2"]].between(0, 1).all()
