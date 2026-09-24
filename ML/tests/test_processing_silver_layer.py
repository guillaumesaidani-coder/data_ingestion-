"""silver_sensor_reading.py / silver_events.py extraits de TP5.ipynb.

Fixtures synthétiques, sans base : la preuve d'identité avec les tables
réelles (Silver et Gold reconstruits contre ceux de TP5/TP6) est dans
tests/test_etl_flow.py, marqué requires_local_infra.
"""

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from indusense.processing.gold_features import GOLD_COLS, to_gold_rows
from indusense.processing.silver_events import (
    SILVER_INCIDENT_COLUMNS,
    SILVER_MAINTENANCE_COLUMNS,
    build_silver_incidents,
    build_silver_maintenance,
)
from indusense.processing.silver_sensor_reading import (
    SILVER_SENSOR_COLUMNS,
    build_silver_sensor_readings,
    normalize_machine_id,
)


def _telemetry(rows):
    return pd.DataFrame(
        rows,
        columns=[
            "machine_id",
            "timestamp",
            "temperature_c",
            "pressure_bar",
            "voltage_mean_v",
            "rotation_mean_rpm",
            "pieces_produced",
        ],
    )


def _incident(incident_id, machine_id="MACH-01", severity=2, operator_name="Alice", **types):
    row = {
        "incident_id": incident_id,
        "date": "2025-06-02",
        "time": "14:30",
        "operator_name": operator_name,
        "machine_id": machine_id,
        "severity": severity,
        "shift": "matin",
        "comment": "",
    }
    return {**row, **types}


def _maintenance(maintenance_id, machine_id="MACH-01", related=None):
    return {
        "maintenance_id": maintenance_id,
        "machine_id": machine_id,
        "maintenance_at": "2025-06-02 08:00:00",
        "maintenance_type": "proactive",
        "action_type": "inspection",
        "component": "pompe",
        "description": "RAS",
        "related_incident_id": related,
        "duration_hours": 1.5,
    }


# ---------------------------------------------------------------------------
# silver_sensor_reading
# ---------------------------------------------------------------------------


def test_normalize_machine_id_strips_and_uppercases():
    assert normalize_machine_id("  mach-01 ") == "MACH-01"
    assert normalize_machine_id(None) is None


def test_sensor_readings_are_unpivoted_one_row_per_sensor():
    bronze = _telemetry([[" mach-01", "2025-06-02 00:00:00", 50.0, 200.0, 227.0, 1590.0, 10]])
    silver = build_silver_sensor_readings(bronze)

    assert list(silver.columns) == SILVER_SENSOR_COLUMNS
    assert len(silver) == 5
    assert list(silver["sensor_type"]) == [
        "temperature_c",
        "pressure_bar",
        "voltage_mean_v",
        "rotation_mean_rpm",
        "pieces_produced",
    ]
    assert list(silver["unit"]) == ["°C", "bar", "V", "RPM", "pcs"]
    assert (silver["machine_id"] == "MACH-01").all()
    assert (silver["observed_at"] == pd.Timestamp("2025-06-02 00:00:00", tz="UTC")).all()


def test_missing_value_is_flagged_and_never_an_outlier():
    bronze = _telemetry([["MACH-01", "2025-06-02 00:00:00", None, 200.0, 227.0, 1590.0, 10]])
    silver = build_silver_sensor_readings(bronze)
    temp = silver[silver["sensor_type"] == "temperature_c"].iloc[0]

    assert temp["is_missing"]
    assert not temp["is_outlier"]
    assert silver["is_missing"].sum() == 1


def test_duplicate_reading_keeps_first_occurrence():
    bronze = _telemetry(
        [
            ["MACH-01", "2025-06-02 00:00:00", 50.0, 200.0, 227.0, 1590.0, 10],
            ["mach-01 ", "2025-06-02 00:00:00", 99.0, 200.0, 227.0, 1590.0, 10],
        ]
    )
    silver = build_silver_sensor_readings(bronze)
    temp = silver[silver["sensor_type"] == "temperature_c"]

    assert list(temp["is_duplicate"]) == [False, True]
    assert temp.loc[~temp["is_duplicate"], "sensor_value"].item() == 50.0


def test_outlier_detection_is_per_sensor_type():
    temps = [50.0] * 9 + [500.0]
    bronze = _telemetry(
        [
            ["MACH-01", f"2025-06-02 {h:02d}:00:00", t, 200.0, 227.0, 1590.0, 10]
            for h, t in enumerate(temps)
        ]
    )
    silver = build_silver_sensor_readings(bronze)

    outliers = silver[silver["is_outlier"]]
    assert len(outliers) == 1
    assert outliers.iloc[0]["sensor_type"] == "temperature_c"
    assert outliers.iloc[0]["sensor_value"] == 500.0


# ---------------------------------------------------------------------------
# silver_incident
# ---------------------------------------------------------------------------

OPERATORS = pd.DataFrame({"operator_id": [7], "operator_key": ["alice"]})


def test_incidents_output_schema_and_timestamp():
    silver = build_silver_incidents(pd.DataFrame([_incident("INC-1")]), OPERATORS)

    assert list(silver.columns) == SILVER_INCIDENT_COLUMNS
    row = silver.iloc[0]
    assert row["incident_code"] == "INC-1"
    assert row["occurred_at"] == pd.Timestamp("2025-06-02 14:30", tz="UTC")


def test_incidents_operator_is_resolved_by_normalized_name():
    bronze = pd.DataFrame(
        [_incident("INC-1", operator_name=" ALICE "), _incident("INC-2", operator_name="Bob")]
    )
    silver = build_silver_incidents(bronze, OPERATORS)

    assert silver.loc[silver["incident_code"] == "INC-1", "operator_id"].item() == 7
    assert pd.isna(silver.loc[silver["incident_code"] == "INC-2", "operator_id"].item())


def test_label_event_threshold_is_severity_4():
    bronze = pd.DataFrame([_incident(f"INC-{s}", severity=s) for s in range(1, 6)])
    silver = build_silver_incidents(bronze, OPERATORS)

    assert list(silver["is_label_event"]) == [False, False, False, True, True]


def test_incidents_duplicate_codes_keep_first():
    bronze = pd.DataFrame(
        [_incident("INC-1", severity=2), _incident("INC-1", severity=5), _incident("INC-2")]
    )
    silver = build_silver_incidents(bronze, OPERATORS)

    assert list(silver["incident_code"]) == ["INC-1", "INC-2"]
    assert silver.loc[0, "severity"] == 2


# ---------------------------------------------------------------------------
# silver_maintenance
# ---------------------------------------------------------------------------


def test_maintenance_output_schema_and_normalization():
    bronze = pd.DataFrame([_maintenance(12, machine_id=" mach-03", related="INC-9")])
    silver = build_silver_maintenance(bronze)

    assert list(silver.columns) == SILVER_MAINTENANCE_COLUMNS
    row = silver.iloc[0]
    assert row["maintenance_code"] == "12"
    assert row["machine_id"] == "MACH-03"
    assert row["related_incident_code"] == "INC-9"
    assert row["performed_at"] == pd.Timestamp("2025-06-02 08:00:00", tz="UTC")


def test_maintenance_duplicate_codes_keep_first():
    bronze = pd.DataFrame([_maintenance(1, "MACH-01"), _maintenance(1, "MACH-02")])
    silver = build_silver_maintenance(bronze)

    assert len(silver) == 1
    assert silver.iloc[0]["machine_id"] == "MACH-01"


# ---------------------------------------------------------------------------
# to_gold_rows
# ---------------------------------------------------------------------------


def test_to_gold_rows_adds_one_hour_window_and_selects_table_columns():
    observed_at = pd.Timestamp("2025-06-02 10:00", tz="UTC")
    df = pd.DataFrame({col: [0] for col in GOLD_COLS}).assign(
        observed_at=[observed_at], extra_col=[1]
    )
    gold = to_gold_rows(df)

    assert list(gold.columns) == GOLD_COLS
    assert gold.loc[0, "window_start"] == observed_at
    assert gold.loc[0, "window_end"] == observed_at + pd.Timedelta(hours=1)
