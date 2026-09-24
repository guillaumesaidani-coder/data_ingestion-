"""Bronze incidents / maintenance -> `silver_incident` / `silver_maintenance`
(extrait de TP5.ipynb, sections 3b, 8 et 11).

Fonctions pures DataFrame -> DataFrame, sur le modèle de
`silver_telemetry.py` : l'I/O (lecture Bronze, batch d'ingestion,
écriture) vit dans `indusense.flows.etl_flow`.

Les deux tables Silver portent une contrainte UNIQUE sur leur code
(`incident_code`, `maintenance_code`). Comme le Bronze est alimenté en
append (`indusense.ingest`), un même code peut y figurer plusieurs fois :
on garde la première occurrence, comme TP5 le faisait déjà pour la
maintenance.
"""

import pandas as pd

from indusense.processing.silver_sensor_reading import normalize_machine_id

LABEL_SEVERITY_MIN = 4  # TP5 §8b : incident critique => événement de label supervisé

SILVER_INCIDENT_COLUMNS = [
    "incident_code",
    "machine_id",
    "operator_id",
    "occurred_at",
    "severity",
    "shift",
    "comment",
    "is_label_event",
]

SILVER_MAINTENANCE_COLUMNS = [
    "maintenance_code",
    "machine_id",
    "performed_at",
    "maintenance_type",
    "action_type",
    "component",
    "duration_hours",
    "related_incident_code",
]


def build_silver_incidents(bronze_incidents: pd.DataFrame, operators: pd.DataFrame) -> pd.DataFrame:
    """Bronze incidents (lignes `parse_ok=True`) -> Silver : machine_id
    normalisé, `occurred_at` = date + time en UTC, `operator_id` retrouvé
    via `operator_key` (nom normalisé, table `operator` alimentée par
    TP4), `is_label_event` si sévérité >= 4."""
    df = bronze_incidents.copy()
    df["machine_id"] = df["machine_id"].map(normalize_machine_id)
    df["occurred_at"] = pd.to_datetime(
        df["date"].str.strip() + " " + df["time"].str.strip(), errors="coerce"
    ).dt.tz_localize("UTC")

    df["operator_key"] = df["operator_name"].str.strip().str.lower()
    df = df.merge(operators[["operator_id", "operator_key"]], on="operator_key", how="left")
    df["operator_id"] = df["operator_id"].astype("Int64")

    df["severity"] = pd.to_numeric(df["severity"], errors="coerce").astype("Int64")
    df["is_label_event"] = (df["severity"] >= LABEL_SEVERITY_MIN).fillna(False).astype(bool)
    df["incident_code"] = df["incident_id"].astype(str)

    df = df.drop_duplicates(subset=["incident_code"], keep="first")
    return df[SILVER_INCIDENT_COLUMNS].reset_index(drop=True)


def build_silver_maintenance(bronze_maintenance: pd.DataFrame) -> pd.DataFrame:
    """Bronze maintenance (lignes `parse_ok=True`) -> Silver : machine_id
    normalisé, `performed_at` en UTC, dédoublonnage sur `maintenance_code`."""
    df = bronze_maintenance.copy()
    df["machine_id"] = df["machine_id"].map(normalize_machine_id)
    df["performed_at"] = pd.to_datetime(df["maintenance_at"], errors="coerce", utc=True)
    df["maintenance_code"] = df["maintenance_id"].astype(str)
    df = df.rename(columns={"related_incident_id": "related_incident_code"})

    df = df.drop_duplicates(subset=["maintenance_code"], keep="first")
    return df[SILVER_MAINTENANCE_COLUMNS].reset_index(drop=True)
