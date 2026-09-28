"""bronze_validation.py extrait de TP4.ipynb : une règle par test, sur des
fixtures synthétiques. La preuve de non-régression sur les vrais fichiers
(mêmes lignes rejetées que TP4) est dans
test_bronze_validation_matches_tp4_on_source_files, marqué
requires_local_infra (fichiers pilotés par DVC + base réelle).
"""

import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from indusense.processing.bronze_validation import (
    data_quality_rows,
    validate_bronze,
)

ML_DIR = Path(__file__).resolve().parent.parent

TELEMETRY_COLS = [
    "machine_id",
    "timestamp",
    "temperature_c",
    "pressure_bar",
    "voltage_mean_v",
    "rotation_mean_rpm",
    "pieces_produced",
]


def _telemetry(*rows) -> pd.DataFrame:
    """Lignes en texte, comme un CSV lu avec dtype=str."""
    return pd.DataFrame(
        [[str(v) if v is not None else None for v in r] for r in rows], columns=TELEMETRY_COLS
    )


def _tel_row(
    machine="MACH-01", ts="2026-06-15 00:00:00", temp="48.0", pressure="195.0", pieces="42"
):
    return [machine, ts, temp, pressure, "228.0", "1600.0", pieces]


def _incident(**overrides) -> dict:
    row = {
        "incident_id": "INC-1",
        "date": "2026-06-15",
        "time": "10:00",
        "operator_name": "Alice",
        "machine_id": "MACH-01",
        "severity": "3",
        "operator_badge": "B1",
        "comment": "",
        "shift": "matin",
        **{
            t: "0"
            for t in [
                "type_surchauffe",
                "type_baisse_pression",
                "type_vibration",
                "type_bruit_mecanique",
                "type_surconsommation",
                "type_blocage_mecanique",
                "type_alarme_capteur",
                "type_arret_urgence",
                "type_defaut_qualite",
            ]
        },
    }
    return {**row, **overrides}


def _maintenance(**overrides) -> dict:
    row = {
        "maintenance_id": "1",
        "machine_id": "MACH-01",
        "maintenance_at": "2026-06-15 08:00:00",
        "maintenance_type": "proactive",
        "action_type": "inspection",
        "component": "pompe",
        "description": "RAS",
        "related_incident_id": None,
        "duration_hours": "1.5",
    }
    return {**row, **overrides}


# ---------------------------------------------------------------------------
# Télémétrie
# ---------------------------------------------------------------------------


def test_valid_telemetry_row_is_typed_and_accepted():
    bronze = validate_bronze(_telemetry(_tel_row()), "telemetry")

    row = bronze.iloc[0]
    assert row["parse_ok"]
    assert row["parse_ok_reason"] == ""
    assert row["temperature_c"] == 48.0
    assert row["pieces_produced"] == 42


def test_non_numeric_value_is_rejected_with_reason():
    bronze = validate_bronze(_telemetry(_tel_row(temp="abc")), "telemetry")

    row = bronze.iloc[0]
    assert not row["parse_ok"]
    assert "temperature_c" in row["parse_ok_reason"]
    assert pd.isna(row["temperature_c"])  # valeur brute illisible -> NaN au typage


def test_missing_sensor_value_is_allowed():
    bronze = validate_bronze(_telemetry(_tel_row(temp=None)), "telemetry")
    assert bronze.iloc[0]["parse_ok"]


@pytest.mark.parametrize(
    ("pressure", "accepted"),
    [("0", True), ("350", True), ("400", True), ("400.1", False), ("-1", False)],
)
def test_pressure_range_covers_dat_contract(pressure, accepted):
    """350 bar : conforme au contrat DAT (<= 400), rejeté par l'ancienne
    borne de TP4 (300). Le Bronze ne doit jamais être plus strict que le
    contrat contrôlé par le feu vert n°3."""
    bronze = validate_bronze(_telemetry(_tel_row(pressure=pressure)), "telemetry")
    assert bool(bronze.iloc[0]["parse_ok"]) is accepted


def test_negative_pieces_produced_is_rejected():
    bronze = validate_bronze(_telemetry(_tel_row(pieces="-1")), "telemetry")
    assert not bronze.iloc[0]["parse_ok"]


def test_hourly_duplicate_rejects_every_copy():
    """Deux relevés MACH-01 à la même heure, valeurs différentes : on ne
    sait pas lequel est le bon, les deux sont rejetés."""
    bronze = validate_bronze(
        _telemetry(
            _tel_row(temp="46.348"),
            _tel_row(temp="46.332"),
            _tel_row(machine="MACH-02"),
        ),
        "telemetry",
    )

    flags = bronze.groupby("machine_id")["parse_ok"].sum().to_dict()
    assert flags == {"MACH-01": 0, "MACH-02": 1}
    reasons = bronze.loc[bronze["machine_id"] == "MACH-01", "parse_ok_reason"]
    assert reasons.str.startswith("DOUBLON_HORAIRE : 2 releves pour MACH-01").all()


def test_invalid_row_does_not_count_as_duplicate_copy():
    """La règle de doublon ne regarde que les lignes encore valides : une
    copie déjà rejetée par Pydantic ne fait pas tomber l'autre."""
    bronze = validate_bronze(_telemetry(_tel_row(), _tel_row(temp="abc")), "telemetry")
    assert bronze["parse_ok"].sum() == 1


# ---------------------------------------------------------------------------
# Incidents / maintenance
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("overrides", "accepted"),
    [({}, True), ({"severity": "6"}, False), ({"type_vibration": "2"}, False)],
)
def test_incident_rules(overrides, accepted):
    bronze = validate_bronze(pd.DataFrame([_incident(**overrides)]), "incidents")
    assert bool(bronze.iloc[0]["parse_ok"]) is accepted


@pytest.mark.parametrize(
    ("overrides", "accepted"),
    [
        ({}, True),
        ({"maintenance_type": "curative"}, False),
        ({"duration_hours": "0"}, False),
    ],
)
def test_maintenance_rules(overrides, accepted):
    bronze = validate_bronze(pd.DataFrame([_maintenance(**overrides)]), "maintenance")
    assert bool(bronze.iloc[0]["parse_ok"]) is accepted


# ---------------------------------------------------------------------------
# data_quality_issue
# ---------------------------------------------------------------------------


def test_data_quality_rows_one_per_rejected_row_with_severity():
    bronze = validate_bronze(
        _telemetry(_tel_row(), _tel_row(), _tel_row(machine="MACH-02", temp="abc")),
        "telemetry",
    )
    rows = data_quality_rows(bronze, "telemetry", "batch-1")

    assert len(rows) == 3
    by_rule = {(r["rule_code"], r["severity"]) for r in rows}
    assert by_rule == {("DOUBLON_HORAIRE", "WARNING"), ("PYDANTIC_ERROR", "ERROR")}
    assert all(r["dataset_name"] == "bronze_telemetry" for r in rows)
    assert all(r["ingestion_batch_id"] == "batch-1" for r in rows)
    assert "MACH-02|2026-06-15 00:00:00" in {r["entity_key"] for r in rows}


# ---------------------------------------------------------------------------
# Non-régression sur les vrais fichiers
# ---------------------------------------------------------------------------


@pytest.mark.requires_local_infra
def test_bronze_validation_matches_tp4_on_source_files(db_engine):
    """Les fichiers sources revalidés par le paquet donnent exactement les
    mêmes lignes acceptées/rejetées que TP4 (tables Bronze en base)."""
    from sqlalchemy import text

    for source, csv_name, key in [
        ("telemetry", "telemetry.csv", ["machine_id", "timestamp"]),
        ("incidents", "releves_incidents.csv", ["incident_id"]),
    ]:
        raw = pd.read_csv(ML_DIR / csv_name, dtype=str, encoding="utf-8")
        ours = validate_bronze(raw, source)
        theirs = pd.read_sql(
            text(f"SELECT {', '.join(key)}, parse_ok FROM bronze_{source}"), db_engine
        )

        pd.testing.assert_frame_equal(
            _parse_ok_by_key(ours, key), _parse_ok_by_key(theirs, key), check_dtype=False
        )


def _parse_ok_by_key(df: pd.DataFrame, key: list[str]) -> pd.DataFrame:
    """Par clé métier : nombre de lignes acceptées et nombre total."""
    return df.groupby(key)["parse_ok"].agg(["sum", "count"]).sort_index()
