"""Feux verts du cycle de réentraînement (module 38) : chaque fonction
isolée de Prefect, testable sans base ni modèle réel — la preuve en
conditions réelles (Prefect + vraies données) est dans hitl_proof.md.
"""

import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from indusense.flows.retrain_flow import (
    check_confirmed_failures_quota,
    check_data_quality,
    check_drift_signal,
    check_validation_quota,
)


def _reviews(n_total, n_confirmed):
    return pd.DataFrame({"ground_truth": [True] * n_confirmed + [False] * (n_total - n_confirmed)})


def test_validation_quota_fails_below_threshold():
    ok, detail = check_validation_quota(_reviews(49, 0))
    assert not ok
    assert "49" in detail


def test_validation_quota_passes_at_threshold():
    ok, _ = check_validation_quota(_reviews(50, 0))
    assert ok


def test_confirmed_failures_quota_uses_ground_truth_only():
    ok, detail = check_confirmed_failures_quota(_reviews(100, 9))
    assert not ok  # 9 < seuil 10
    assert "9" in detail

    ok2, _ = check_confirmed_failures_quota(_reviews(100, 10))
    assert ok2


def _fake_gold(feature_values_early, feature_values_trainval):
    """Objet minimal imitant GoldDataset pour tester les feux verts 3/4
    sans base réelle."""
    cols = list(feature_values_early.keys())
    early_df = pd.DataFrame({c: feature_values_early[c] for c in cols})
    early_df["window_start"] = pd.Timestamp("2026-05-01", tz="UTC")
    trainval_df = pd.DataFrame({c: feature_values_trainval[c] for c in cols})
    return SimpleNamespace(test_df=early_df, trainval_df=trainval_df, feature_cols=cols)


def test_data_quality_gate_passes_on_plausible_values():
    values = {
        "temp_mean_24h": [48.0, 50.0, 47.0],
        "pressure_mean_24h": [195.0, 198.0, 200.0],
        "voltage_mean_24h": [228.0, 227.0, 229.0],
        "rotation_mean_24h": [1600.0, 1610.0, 1590.0],
    }
    gold = _fake_gold(values, values)
    ok, detail = check_data_quality(gold, pd.Timestamp("2026-06-01", tz="UTC"))
    assert ok
    assert "aucune" in detail


def test_data_quality_gate_fails_on_absurd_value():
    values = {
        "temp_mean_24h": [48.0, 9999.0],
        "pressure_mean_24h": [195.0, 198.0],
        "voltage_mean_24h": [228.0, 227.0],
        "rotation_mean_24h": [1600.0, 1610.0],
    }
    gold = _fake_gold(values, values)
    ok, detail = check_data_quality(gold, pd.Timestamp("2026-06-01", tz="UTC"))
    assert not ok
    assert "temp_mean_24h" in detail


def test_drift_gate_is_green_when_distribution_has_shifted():
    # PSI degenere sur une reference a variance nulle (tous les quantiles
    # de bin s'effondrent en un seul point) : un vrai etalement, comme
    # les vraies mesures capteurs, est necessaire pour que le test soit
    # significatif -- pas juste une moyenne differente.
    rng = np.random.default_rng(42)
    n = 200
    base = {
        "temp_std_24h": rng.normal(1.0, 0.1, n).tolist(),
        "pressure_mean_24h": rng.normal(195.0, 2.0, n).tolist(),
        "pressure_std_24h": rng.normal(1.0, 0.1, n).tolist(),
        "voltage_mean_24h": rng.normal(228.0, 1.0, n).tolist(),
        "rotation_mean_24h": rng.normal(1600.0, 20.0, n).tolist(),
        "pieces_produced_sum_24h": rng.normal(1000, 30, n).tolist(),
        "incident_count_prev_24h": rng.integers(0, 2, n).tolist(),
    }
    early = {**base, "temp_mean_24h": rng.normal(70.0, 3.0, n).tolist()}
    trainval = {
        **base,
        "temp_mean_24h": rng.normal(45.0, 3.0, n).tolist(),
    }  # seule la temperature a bouge
    gold = _fake_gold(early, trainval)
    ok, detail = check_drift_signal(gold, pd.Timestamp("2026-06-01", tz="UTC"))
    assert ok  # derive constatee = feu vert au sens de la feuille de route
    assert "temp_mean_24h" in detail


def test_drift_gate_is_red_when_nothing_moved():
    values = {
        "temp_mean_24h": [48.0] * 50,
        "temp_std_24h": [1.0] * 50,
        "pressure_mean_24h": [195.0] * 50,
        "pressure_std_24h": [1.0] * 50,
        "voltage_mean_24h": [228.0] * 50,
        "rotation_mean_24h": [1600.0] * 50,
        "pieces_produced_sum_24h": [1000] * 50,
        "incident_count_prev_24h": [0] * 50,
    }
    gold = _fake_gold(values, values)
    ok, _ = check_drift_signal(gold, pd.Timestamp("2026-06-01", tz="UTC"))
    assert not ok  # rien n'a bouge : pas de raison de reentrainer
