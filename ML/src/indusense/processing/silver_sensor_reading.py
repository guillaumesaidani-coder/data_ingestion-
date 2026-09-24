"""Bronze telemetry -> `silver_sensor_reading` (extrait de TP5.ipynb,
sections 2 à 5).

À ne pas confondre avec `silver_telemetry.py` (TP2) : ce dernier produit
un Silver *large* avec interpolation des outliers, utilisé pour l'analyse
exploratoire. Le Gold (TP6, `gold_features.py`) est construit à partir
de celui-ci : format *long* (1 ligne par capteur par relevé), valeurs
brutes, et trois flags qualité — les outliers sont seulement signalés,
jamais corrigés.

Fonction pure DataFrame -> DataFrame : l'I/O (lecture Bronze, batch
d'ingestion, écriture) vit dans `indusense.flows.etl_flow`.
"""

import pandas as pd

# (colonne Bronze, sensor_type, unité) — même ordre que TP5 : l'ordre de
# concaténation décide quel relevé garde `is_duplicate=False`.
SENSOR_MAP = [
    ("temperature_c", "temperature_c", "°C"),
    ("pressure_bar", "pressure_bar", "bar"),
    ("voltage_mean_v", "voltage_mean_v", "V"),
    ("rotation_mean_rpm", "rotation_mean_rpm", "RPM"),
    ("pieces_produced", "pieces_produced", "pcs"),
]

OUTLIER_IQR_FACTOR = 1.5  # TP5 : [Q1 - 1.5*IQR, Q3 + 1.5*IQR], par sensor_type

SILVER_SENSOR_COLUMNS = [
    "machine_id",
    "observed_at",
    "sensor_type",
    "sensor_value",
    "unit",
    "is_missing",
    "is_duplicate",
    "is_outlier",
]


def normalize_machine_id(mid) -> str | None:
    """strip + uppercase ; None si absent. Partagé avec `silver_events`."""
    if pd.isna(mid):
        return None
    return str(mid).strip().upper()


def _flag_outliers(values: pd.Series, sensor_types: pd.Series) -> pd.Series:
    """IQR par sensor_type, quartiles calculés hors NaN ; une valeur
    manquante n'est jamais un outlier."""
    grouped = values.groupby(sensor_types)
    q1 = grouped.transform(lambda v: v.quantile(0.25))
    q3 = grouped.transform(lambda v: v.quantile(0.75))
    iqr = q3 - q1
    low = q1 - OUTLIER_IQR_FACTOR * iqr
    high = q3 + OUTLIER_IQR_FACTOR * iqr
    return (values < low) | (values > high)


def build_silver_sensor_readings(bronze_telemetry: pd.DataFrame) -> pd.DataFrame:
    """Bronze telemetry (lignes `parse_ok=True`) -> format long flaggé :
    normalisation machine_id, parsing UTC de `timestamp`, dépivotage des
    5 capteurs, puis `is_missing`, `is_duplicate` (machine, heure,
    capteur ; le premier gagne) et `is_outlier`."""
    df = bronze_telemetry.copy()
    df["machine_id"] = df["machine_id"].map(normalize_machine_id)
    df["observed_at"] = pd.to_datetime(df["timestamp"], errors="coerce", utc=True)

    chunks = []
    for col, sensor_type, unit in SENSOR_MAP:
        tmp = df[["machine_id", "observed_at", col]].rename(columns={col: "sensor_value"})
        tmp["sensor_type"] = sensor_type
        tmp["unit"] = unit
        chunks.append(tmp)

    long = pd.concat(chunks, ignore_index=True)
    long["sensor_value"] = pd.to_numeric(long["sensor_value"], errors="coerce")
    long["is_missing"] = long["sensor_value"].isna()
    long["is_duplicate"] = long.duplicated(
        subset=["machine_id", "observed_at", "sensor_type"], keep="first"
    )
    long["is_outlier"] = _flag_outliers(long["sensor_value"], long["sensor_type"])
    return long[SILVER_SENSOR_COLUMNS]
