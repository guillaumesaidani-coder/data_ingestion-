"""build_silver_telemetry() extrait à l'identique de TP2.ipynb (cellule
"Production du silver"). Vérifié par comparaison stricte (assert_frame_equal)
contre la logique inline originale rejouée sur telemetry.csv avant l'extraction.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from indusense.processing.silver_telemetry import (
    SILVER_COLUMNS,
    build_silver_telemetry,
)

ML_DIR = Path(__file__).resolve().parent.parent


def _bronze_row(machine_id, timestamp, temp, pressure=200.0, voltage=227.0, rotation=1590.0, pieces=10):
    return {
        "machine_id": machine_id,
        "timestamp": timestamp,
        "temperature_c": temp,
        "pressure_bar": pressure,
        "voltage_mean_v": voltage,
        "rotation_mean_rpm": rotation,
        "pieces_produced": pieces,
    }


def test_output_columns_match_expected_silver_schema():
    df = pd.DataFrame([
        _bronze_row("MACH-01", "2025-06-02 00:00:00", 50.0),  # lundi
        _bronze_row("MACH-01", "2025-06-02 01:00:00", 51.0),
    ])
    silver, _ = build_silver_telemetry(df)
    assert list(silver.columns) == SILVER_COLUMNS


def test_is_weekend_flag():
    df = pd.DataFrame([
        _bronze_row("MACH-01", "2025-06-02 00:00:00", 50.0),  # lundi -> False
        _bronze_row("MACH-01", "2025-06-07 00:00:00", 50.0),  # samedi -> True
    ])
    silver, _ = build_silver_telemetry(df)
    silver = silver.sort_values("timestamp").reset_index(drop=True)
    assert list(silver["is_weekend"]) == [False, True]


def test_is_stopped_flag_from_zero_production():
    df = pd.DataFrame([
        _bronze_row("MACH-01", "2025-06-02 00:00:00", 50.0, pieces=0),
        _bronze_row("MACH-01", "2025-06-02 01:00:00", 50.0, pieces=5),
    ])
    silver, _ = build_silver_telemetry(df)
    silver = silver.sort_values("timestamp").reset_index(drop=True)
    assert list(silver["is_stopped"]) == [True, False]


def test_outlier_is_replaced_by_linear_interpolation():
    # Série régulière (50, 51, 52, 53, 54) avec un point aberrant à l'index 2.
    rows = [_bronze_row("MACH-01", f"2025-06-02 0{h}:00:00", temp)
            for h, temp in enumerate([50.0, 51.0, 500.0, 53.0, 54.0])]
    df = pd.DataFrame(rows)
    silver, outlier_report = build_silver_telemetry(df)
    silver = silver.sort_values("timestamp").reset_index(drop=True)

    assert outlier_report["temperature_c"] == 1
    assert silver.loc[2, "temp_c"] == pytest.approx(52.0)  # interpolé entre 51 et 53


def test_interpolation_does_not_leak_across_machines():
    # MACH-01 : valeurs stables (pas d'outlier). MACH-02 : un seul point, NaN
    # après interpolation (limit_direction="both" ne peut rien combler seul,
    # mais surtout ne doit jamais piocher une valeur de MACH-01).
    rows = [_bronze_row("MACH-01", f"2025-06-02 0{h}:00:00", 50.0) for h in range(5)]
    rows += [_bronze_row("MACH-02", "2025-06-02 00:00:00", 9999.0)]  # outlier isolé
    df = pd.DataFrame(rows)
    silver, _ = build_silver_telemetry(df)

    mach02 = silver[silver["machine_id"] == "MACH-02"]
    # Un seul point outlier sur sa propre machine -> rien à interpoler depuis
    # MACH-01 : la valeur reste NaN, jamais 50.0 (qui serait une fuite inter-machine).
    assert mach02["temp_c"].isna().all()


def test_matches_known_outlier_counts_on_real_telemetry_csv():
    # Régression : mêmes comptes qu'annoncés dans
    # pipeline_artifacts/ANALYSE_ARTIFACTS.md (1201 outliers bronze au total).
    bronze = pd.read_csv(ML_DIR / "telemetry.csv")
    _, outlier_report = build_silver_telemetry(bronze)
    assert outlier_report == {
        "temperature_c": 52,
        "pressure_bar": 769,
        "voltage_mean_v": 77,
        "rotation_mean_rpm": 303,
    }
    assert sum(outlier_report.values()) == 1201
