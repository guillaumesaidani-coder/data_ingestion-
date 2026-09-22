"""check_sensor_quality() (module 38) : bornes physiques réutilisées du
contrat DAT (§5.4) — vérifie qu'une valeur absurde (feuille de route §6,
« température de 9 999 °C ») est bien détectée, et qu'une donnée réelle
plausible ne déclenche jamais de faux positif.
"""

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from indusense.data_quality import check_sensor_quality


def _plausible_row(**overrides):
    row = {
        "temp_mean_24h": 48.0,
        "pressure_mean_24h": 195.0,
        "voltage_mean_24h": 228.0,
        "rotation_mean_24h": 1600.0,
    }
    row.update(overrides)
    return row


def test_plausible_readings_pass():
    df = pd.DataFrame([_plausible_row(), _plausible_row(temp_mean_24h=52.0)])
    ok, errors = check_sensor_quality(df)
    assert ok
    assert errors == []


def test_absurd_temperature_is_caught():
    df = pd.DataFrame([_plausible_row(), _plausible_row(temp_mean_24h=9999.0)])
    ok, errors = check_sensor_quality(df)
    assert not ok
    assert any("temp_mean_24h" in e for e in errors)


def test_negative_pressure_is_caught():
    df = pd.DataFrame([_plausible_row(pressure_mean_24h=-5.0)])
    ok, errors = check_sensor_quality(df)
    assert not ok
    assert any("pressure_mean_24h" in e for e in errors)


def test_multiple_violations_are_all_reported():
    df = pd.DataFrame([_plausible_row(temp_mean_24h=-999.0, rotation_mean_24h=-10.0)])
    ok, errors = check_sensor_quality(df)
    assert not ok
    assert len(errors) >= 2  # les deux violations remontent, pas juste la premiere (lazy=True)
