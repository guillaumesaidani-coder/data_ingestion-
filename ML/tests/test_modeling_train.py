import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from indusense.modeling.train import B11_PARAMS, compute_scale_pos_weight, train_and_evaluate


def test_b11_params_are_the_documented_values():
    # Valeurs figées, cf. artifacts/model_card.md et le commit qui a extrait ce module.
    assert B11_PARAMS["n_estimators"] == 287
    assert B11_PARAMS["max_depth"] == 9
    assert B11_PARAMS["learning_rate"] == pytest.approx(0.02887139049912187)
    assert B11_PARAMS["random_state"] == 42


def test_compute_scale_pos_weight():
    y = pd.Series([0, 0, 0, 1])
    assert compute_scale_pos_weight(y) == pytest.approx(3.0)


def test_train_and_evaluate_returns_pipeline_and_metrics_smoke():
    rng = np.random.default_rng(42)
    n = 200
    X = pd.DataFrame({"f1": rng.normal(size=n), "f2": rng.normal(size=n)})
    y = pd.Series((X["f1"] + rng.normal(scale=0.1, size=n) > 0).astype(int))

    X_tv, X_test = X.iloc[:150], X.iloc[150:]
    y_tv, y_test = y.iloc[:150], y.iloc[150:]

    params = {"n_estimators": 10, "max_depth": 2, "random_state": 42, "verbosity": 0}
    pipe, metrics = train_and_evaluate(X_tv, y_tv, X_test, y_test, params)

    assert hasattr(pipe, "predict_proba")
    assert 0.0 <= metrics["pr_auc_test"] <= 1.0
    assert metrics["tp"] + metrics["tn"] + metrics["fp"] + metrics["fn"] == len(y_test)
