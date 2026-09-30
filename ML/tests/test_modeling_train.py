import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from indusense.modeling.pipeline import (
    CalibratedPipeline,
    alert_threshold,
    build_xgb_pipeline,
    predict_alert,
)
from indusense.modeling.train import (
    B11_PARAMS,
    compute_scale_pos_weight,
    evaluate,
    train_and_evaluate,
)


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


def test_evaluate_reproduces_train_and_evaluate_metrics():
    # La model card mesure le modèle servi avec evaluate(), sans le
    # ré-entraîner : mêmes métriques que celles écrites à l'entraînement.
    rng = np.random.default_rng(0)
    X = pd.DataFrame({"f1": rng.normal(size=200), "f2": rng.normal(size=200)})
    y = pd.Series((X["f1"] > 0).astype(int))
    X_tv, X_test, y_tv, y_test = X.iloc[:150], X.iloc[150:], y.iloc[:150], y.iloc[150:]
    params = {"n_estimators": 10, "max_depth": 2, "random_state": 42, "verbosity": 0}

    pipe, metrics = train_and_evaluate(X_tv, y_tv, X_test, y_test, params)

    assert evaluate(pipe, X_tv, y_tv, X_test, y_test) == metrics


def _grouped_data(n=600, seed=0):
    rng = np.random.default_rng(seed)
    X = pd.DataFrame({"f1": rng.normal(size=n), "f2": rng.normal(size=n)})
    y = pd.Series((X["f1"] + rng.normal(scale=0.8, size=n) > 1.2).astype(int))
    groups = pd.Series([f"MACH-{i % 6:02d}" for i in range(n)])
    return X, y, groups


def test_with_groups_the_model_is_calibrated_without_changing_ranking_or_alerts():
    # La calibration ne change que le chiffre affiché : même classement
    # (PR-AUC), mêmes alertes qu'au score brut 0,5.
    X, y, groups = _grouped_data()
    X_tv, X_test, y_tv, y_test = X.iloc[:450], X.iloc[450:], y.iloc[:450], y.iloc[450:]
    params = {"n_estimators": 20, "max_depth": 2, "random_state": 42, "verbosity": 0,
              "scale_pos_weight": compute_scale_pos_weight(y_tv)}

    raw, raw_metrics = train_and_evaluate(X_tv, y_tv, X_test, y_test, params)
    cal, cal_metrics = train_and_evaluate(X_tv, y_tv, X_test, y_test, params,
                                          groups=groups.iloc[:450])

    assert isinstance(cal, CalibratedPipeline)
    assert set(cal.named_steps) == {"imputer", "model"}  # explain.py / scoring inchangés
    p_raw = raw.predict_proba(X_test)[:, 1]
    p_cal = cal.predict_proba(X_test)[:, 1]
    assert not np.allclose(p_raw, p_cal)
    assert np.array_equal(np.argsort(p_raw, kind="stable"), np.argsort(p_cal, kind="stable"))
    assert np.array_equal(predict_alert(cal, X_test), (p_raw >= 0.5).astype(int))
    assert np.array_equal(cal.predict(X_test), predict_alert(cal, X_test))
    assert cal_metrics["pr_auc_test"] == raw_metrics["pr_auc_test"]
    assert cal_metrics["threshold"] == pytest.approx(cal.threshold_, abs=1e-4)
    assert evaluate(cal, X_tv, y_tv, X_test, y_test) == cal_metrics


def test_calibrated_pipeline_survives_joblib(tmp_path):
    X, y, groups = _grouped_data()
    params = {"n_estimators": 10, "max_depth": 2, "random_state": 42, "verbosity": 0}
    pipe, _ = train_and_evaluate(X, y, X, y, params, groups=groups)
    joblib.dump(pipe, tmp_path / "model.joblib")
    loaded = joblib.load(tmp_path / "model.joblib")
    assert loaded.threshold_ == pipe.threshold_
    assert np.allclose(loaded.predict_proba(X), pipe.predict_proba(X))


def test_calibration_refuses_a_non_positive_slope():
    with pytest.raises(ValueError):
        build_xgb_pipeline({}, calibrated=True).set_calibration(-1.0, 0.0)


def test_raw_pipeline_keeps_the_historical_threshold():
    assert alert_threshold(build_xgb_pipeline({})) == 0.5
