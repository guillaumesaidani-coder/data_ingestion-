"""gold_features.py extrait à l'identique de TP6.ipynb.

Vérifié avant extraction (pas reproduit ici, trop lent pour la suite
courante — voir le message du commit correspondant) : comparaison
stricte de build_gold_features() rejouée sur les vraies tables Silver
contre le contenu réel de gold_machine_hourly_feature (132 940 lignes,
toutes colonnes, tolérance 1e-6) -> aucun écart.

Ces tests couvrent chaque fonction isolément sur des fixtures
synthétiques, rapides, pour verrouiller le comportement pièce par pièce.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from indusense.processing.gold_features import (
    assign_time_split,
    build_gold_features,
    compute_capacity_utilization,
    compute_incident_lookback_features,
    compute_machine_zscore,
    compute_maintenance_lookback_features,
    compute_multi_horizon_labels,
    compute_rolling_features,
    compute_rolling_zscore,
    pivot_to_wide,
)


def _sensor_rows(machine_id, start, n_hours, temp=50.0, pressure=200.0, voltage=227.0, rotation=1590.0, pieces=10):
    rows = []
    for h in range(n_hours):
        ts = start + pd.Timedelta(hours=h)
        for sensor_type, val in [
            ("temperature_c", temp), ("pressure_bar", pressure),
            ("voltage_mean_v", voltage), ("rotation_mean_rpm", rotation),
            ("pieces_produced", pieces),
        ]:
            rows.append({"machine_id": machine_id, "observed_at": ts, "sensor_type": sensor_type, "sensor_value": val})
    return rows


def test_pivot_to_wide_fills_missing_sensor_columns_with_nan():
    df_sensors = pd.DataFrame([
        {"machine_id": "MACH-01", "observed_at": pd.Timestamp("2025-06-01"), "sensor_type": "temperature_c", "sensor_value": 50.0},
    ])
    wide = pivot_to_wide(df_sensors)
    for col in ["pressure_bar", "voltage_mean_v", "rotation_mean_rpm", "pieces_produced"]:
        assert col in wide.columns
        assert wide[col].isna().all()


def test_rolling_window_excludes_current_value():
    # closed='left' : la fenêtre 6h ne doit jamais inclure la valeur courante.
    rows = _sensor_rows("MACH-01", pd.Timestamp("2025-06-01", tz="UTC"), 3, temp=10.0)
    df_sensors = pd.DataFrame(rows)
    wide = pivot_to_wide(df_sensors)
    # dernière heure a une temp très différente : si elle fuitait dans son propre
    # rolling mean, la moyenne 6h de la dernière ligne serait égale à sa valeur.
    wide.loc[wide.index[-1], "temperature_c"] = 999.0
    out = compute_rolling_features(wide)
    last = out.sort_values("observed_at").iloc[-1]
    assert last["temp_mean_6h"] != 999.0
    assert last["temp_mean_6h"] == pytest.approx(10.0)  # moyenne des 2 heures précédentes


def test_rolling_deltas_and_trend():
    rows = _sensor_rows("MACH-01", pd.Timestamp("2025-06-01", tz="UTC"), 7, temp=0.0)
    df_sensors = pd.DataFrame(rows)
    wide = pivot_to_wide(df_sensors)
    wide["temperature_c"] = [10.0, 11.0, 13.0, 16.0, 20.0, 25.0, 31.0]
    out = compute_rolling_features(wide).sort_values("observed_at").reset_index(drop=True)
    # index 6 = 31.0 ; diff(k) compare a l'index (6-k) : diff(1)->25.0, diff(3)->16.0, diff(6)->10.0
    assert out.loc[6, "temp_delta_1h"] == pytest.approx(31.0 - 25.0)
    assert out.loc[6, "temp_delta_3h"] == pytest.approx(31.0 - 16.0)
    assert out.loc[6, "temp_trend_6h"] == pytest.approx(31.0 - 10.0)


def test_capacity_utilization_pct():
    df = pd.DataFrame({"machine_id": ["MACH-01"], "pieces_produced_sum_1h": [50.0]})
    df_machine = pd.DataFrame({"machine_id": ["MACH-01"], "max_hourly_capacity_pieces": [100.0]})
    out = compute_capacity_utilization(df, df_machine)
    assert out["capacity_utilization_pct"].iloc[0] == pytest.approx(50.0)
    assert "max_hourly_capacity_pieces" not in out.columns


def test_rolling_zscore_zero_std_gives_nan():
    df = pd.DataFrame({
        "temperature_c": [50.0], "temp_mean_24h": [50.0], "temp_std_24h": [0.0],
        "pressure_bar": [200.0], "pressure_mean_24h": [200.0], "pressure_std_24h": [1.0],
    })
    out = compute_rolling_zscore(df)
    assert pd.isna(out["temp_zscore_24h"].iloc[0])
    assert out["pressure_zscore_24h"].iloc[0] == pytest.approx(0.0)


def test_assign_time_split_ratios():
    dates = pd.date_range("2025-01-01", periods=100, freq="h", tz="UTC")
    df = pd.DataFrame({"observed_at": dates})
    out = assign_time_split(df)
    counts = out["split_set"].value_counts()
    assert counts["train"] == 70
    assert counts["validation"] == 15
    assert counts["test"] == 15


def test_machine_zscore_baseline_is_train_only():
    # Train : moyenne 10. Val/test : valeurs très différentes qui ne doivent
    # jamais influencer la baseline (anti-leakage).
    df = pd.DataFrame({
        "machine_id":  ["MACH-01"] * 3 + ["MACH-01"] * 2,
        "temperature_c": [8.0, 10.0, 12.0, 1000.0, -1000.0],
        "pressure_bar":  [200.0, 200.0, 200.0, 200.0, 200.0],
        "split_set":   ["train", "train", "train", "test", "test"],
    })
    out = compute_machine_zscore(df)
    train_mean, train_std = 10.0, np.std([8.0, 10.0, 12.0], ddof=1)
    expected_test_z = (1000.0 - train_mean) / train_std
    test_rows = out[out["split_set"] == "test"]
    assert test_rows["temp_zscore_machine"].iloc[0] == pytest.approx(expected_test_z)


def test_incident_lookback_window_is_strictly_past():
    df = pd.DataFrame({
        "machine_id":  ["MACH-01"],
        "observed_at": [pd.Timestamp("2025-06-02 12:00:00", tz="UTC")],
    })
    df_inc = pd.DataFrame({
        "machine_id":  ["MACH-01", "MACH-01"],
        # Un incident exactement à h (doit être exclu, < h strict) et un à h-1h (inclus).
        "occurred_at": [pd.Timestamp("2025-06-02 12:00:00", tz="UTC"), pd.Timestamp("2025-06-02 11:00:00", tz="UTC")],
        "severity":    [5, 2],
        "type_surchauffe": [1, 0], "type_baisse_pression": [0, 0], "type_vibration": [0, 0],
        "type_bruit_mecanique": [0, 0], "type_surconsommation": [0, 0], "type_blocage_mecanique": [0, 0],
        "type_alarme_capteur": [0, 0], "type_arret_urgence": [0, 0], "type_defaut_qualite": [0, 0],
    })
    out = compute_incident_lookback_features(df, df_inc)
    assert out["incident_count_prev_24h"].iloc[0] == 1
    assert out["incident_max_severity_prev_24h"].iloc[0] == 2  # celui à h, exclu -> seul severity=2 compte
    assert out["hours_since_last_incident"].iloc[0] == pytest.approx(1.0)


def test_maintenance_lookback_days_since_last():
    df = pd.DataFrame({
        "machine_id":  ["MACH-01"],
        "observed_at": [pd.Timestamp("2025-06-10 00:00:00", tz="UTC")],
    })
    df_maint = pd.DataFrame({
        "machine_id":   ["MACH-01"],
        "performed_at": [pd.Timestamp("2025-06-08 00:00:00", tz="UTC")],
    })
    out = compute_maintenance_lookback_features(df, df_maint)
    assert out["days_since_last_maintenance"].iloc[0] == pytest.approx(2.0)
    assert out["maintenance_count_prev_30d"].iloc[0] == 1


def test_multi_horizon_labels_lookahead_inverted():
    df = pd.DataFrame({
        "machine_id":  ["MACH-01"] * 3,
        "observed_at": pd.date_range("2025-06-02 00:00:00", periods=3, freq="h", tz="UTC"),
    })
    df_inc = pd.DataFrame({
        "machine_id":     ["MACH-01"],
        "occurred_at":    [pd.Timestamp("2025-06-02 02:00:00", tz="UTC")],
        "is_label_event": [True],
    })
    out = compute_multi_horizon_labels(df, df_inc)
    # Horizon 6h : toutes les heures dans [evt-6h, evt) sont True -> les 3 lignes (00h,01h,02h< evt=02h donc 00h,01h True, 02h False)
    assert list(out.sort_values("observed_at")["label_failure_next_6h"]) == [True, True, False]
    assert out["future_incident_count_6h"].max() == 1


def test_build_gold_features_end_to_end_smoke():
    rows = _sensor_rows("MACH-01", pd.Timestamp("2025-06-01", tz="UTC"), 200)
    df_sensors = pd.DataFrame(rows)
    df_inc = pd.DataFrame(columns=[
        "machine_id", "occurred_at", "severity", "is_label_event",
        "type_surchauffe", "type_baisse_pression", "type_vibration", "type_bruit_mecanique",
        "type_surconsommation", "type_blocage_mecanique", "type_alarme_capteur",
        "type_arret_urgence", "type_defaut_qualite",
    ])
    df_maint = pd.DataFrame(columns=["machine_id", "performed_at"])
    df_machine = pd.DataFrame({"machine_id": ["MACH-01"], "max_hourly_capacity_pieces": [100.0]})

    out = build_gold_features(df_sensors, df_inc, df_maint, df_machine)
    assert len(out) == 200
    assert set(["train", "validation", "test"]) >= set(out["split_set"].unique())
    assert not out["label_failure_next_24h"].any()  # aucun incident -> jamais de label
