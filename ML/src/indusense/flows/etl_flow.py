"""Rupture ① du cadrage : flow Prefect Bronze -> Silver -> Gold, sans
notebook.

Enchaîne ce que TP5.ipynb (Silver) et TP6.ipynb (Gold) faisaient à la
main, avec les transformations du paquet (`processing.silver_sensor_reading`,
`processing.silver_events`, `processing.gold_features`) et le même
traçage `ingestion_batch` à chaque étage (mêmes `source_name` que les
notebooks : silver_telemetry, silver_incidents, silver_maintenance, gold).

Le Bronze n'est pas reconstruit : il s'alimente en append, lot par lot
(`indusense.ingest.ingest_terrain_batch`, idempotent par hash de fichier).
Silver et Gold, eux, sont reconstruits en entier à chaque exécution
(TRUNCATE + réécriture, comme TP5/TP6) : rejouer le flow laisse le même
nombre de lignes, jamais le double. Chaque table est vidée et réécrite
dans une seule transaction — un étage qui échoue laisse l'ancienne
version en place (et son batch en status 'running'), jamais une table
vide.

Postgres uniquement (TRUNCATE ... RESTART IDENTITY, tables créées par
les migrations alembic) : le test de bout en bout est marqué
requires_local_infra ; les transformations sont testées sans base.

Usage local : uv run --frozen indusense etl
Usage conteneur : docker compose run --rm api python -m indusense.flows.etl_flow
"""

import pandas as pd
from prefect import flow, get_run_logger, task
from prefect.cache_policies import NO_CACHE
from sqlalchemy import text
from sqlalchemy.engine import Engine

from indusense.config import get_engine
from indusense.ingest import close_batch, open_batch
from indusense.processing.gold_features import build_gold_features, to_gold_rows
from indusense.processing.silver_events import (
    build_silver_incidents,
    build_silver_maintenance,
)
from indusense.processing.silver_sensor_reading import build_silver_sensor_readings

# Lectures : mêmes requêtes que TP5 (Bronze parse_ok) et TP6 (Silver),
# ORDER BY ajouté pour que le « premier gagne » des dédoublonnages ne
# dépende pas de l'ordre physique des lignes.
BRONZE_TELEMETRY_QUERY = "SELECT * FROM bronze_telemetry WHERE parse_ok = true ORDER BY id"
BRONZE_INCIDENTS_QUERY = "SELECT * FROM bronze_incidents WHERE parse_ok = true ORDER BY id"
BRONZE_MAINTENANCE_QUERY = "SELECT * FROM bronze_maintenance WHERE parse_ok = true ORDER BY id"
OPERATOR_QUERY = "SELECT operator_id, operator_key FROM operator"

SILVER_SENSORS_QUERY = """
    SELECT machine_id, observed_at, sensor_type, sensor_value
    FROM silver_sensor_reading
    WHERE is_missing = false AND is_duplicate = false
"""
SILVER_INCIDENTS_QUERY = """
    SELECT si.machine_id, si.occurred_at, si.severity, si.is_label_event,
           bi.type_surchauffe, bi.type_baisse_pression, bi.type_vibration,
           bi.type_bruit_mecanique, bi.type_surconsommation, bi.type_blocage_mecanique,
           bi.type_alarme_capteur, bi.type_arret_urgence, bi.type_defaut_qualite
    FROM silver_incident si
    JOIN bronze_incidents bi ON bi.incident_id = si.incident_code
"""
SILVER_MAINTENANCE_QUERY = "SELECT machine_id, performed_at FROM silver_maintenance"
MACHINE_QUERY = "SELECT machine_code AS machine_id, max_hourly_capacity_pieces FROM machine"


def replace_table(engine: Engine, table: str, df: pd.DataFrame, chunksize: int = 2000) -> None:
    """TRUNCATE + réécriture dans une seule transaction : les lecteurs
    voient l'ancienne ou la nouvelle version de la table, jamais un état
    vide ou partiel."""
    with engine.begin() as conn:
        conn.execute(text(f"TRUNCATE TABLE {table} RESTART IDENTITY"))
        df.to_sql(table, conn, if_exists="append", index=False, method="multi", chunksize=chunksize)


# cache_policy=NO_CACHE : les tasks reçoivent un Engine (non hachable par
# Prefect, cf. predict_flow.store_predictions) et doivent toujours écrire.


@task(name="silver-sensor-reading", cache_policy=NO_CACHE)
def rebuild_silver_sensor_reading(engine: Engine) -> int:
    logger = get_run_logger()
    bronze = pd.read_sql(text(BRONZE_TELEMETRY_QUERY), engine)
    silver = build_silver_sensor_readings(bronze)

    batch_id = open_batch(engine, "silver_telemetry", "bronze_telemetry")
    replace_table(engine, "silver_sensor_reading", silver.assign(ingestion_batch_id=str(batch_id)))
    rejected = int((silver["is_missing"] | silver["is_duplicate"]).sum())
    close_batch(
        engine,
        batch_id,
        rows_read=len(silver),
        rows_loaded=len(silver) - rejected,
        rows_rejected=rejected,
    )
    logger.info("silver_sensor_reading : %d lignes (%d écartées du Gold)", len(silver), rejected)
    return len(silver)


@task(name="silver-incident", cache_policy=NO_CACHE)
def rebuild_silver_incident(engine: Engine) -> int:
    logger = get_run_logger()
    bronze = pd.read_sql(text(BRONZE_INCIDENTS_QUERY), engine)
    operators = pd.read_sql(text(OPERATOR_QUERY), engine)
    silver = build_silver_incidents(bronze, operators)

    batch_id = open_batch(engine, "silver_incidents", "bronze_incidents")
    replace_table(
        engine, "silver_incident", silver.assign(ingestion_batch_id=str(batch_id)), chunksize=500
    )
    close_batch(
        engine,
        batch_id,
        rows_read=len(bronze),
        rows_loaded=len(silver),
        rows_rejected=len(bronze) - len(silver),
    )
    logger.info("silver_incident : %d lignes", len(silver))
    return len(silver)


@task(name="silver-maintenance", cache_policy=NO_CACHE)
def rebuild_silver_maintenance(engine: Engine) -> int:
    logger = get_run_logger()
    bronze = pd.read_sql(text(BRONZE_MAINTENANCE_QUERY), engine)
    silver = build_silver_maintenance(bronze)

    batch_id = open_batch(engine, "silver_maintenance", "bronze_maintenance")
    replace_table(
        engine,
        "silver_maintenance",
        silver.assign(ingestion_batch_id=str(batch_id)),
        chunksize=1000,
    )
    close_batch(
        engine,
        batch_id,
        rows_read=len(bronze),
        rows_loaded=len(silver),
        rows_rejected=len(bronze) - len(silver),
    )
    logger.info("silver_maintenance : %d lignes", len(silver))
    return len(silver)


@task(name="gold", cache_policy=NO_CACHE)
def rebuild_gold(engine: Engine) -> int:
    logger = get_run_logger()
    df_sensors = pd.read_sql(text(SILVER_SENSORS_QUERY), engine, parse_dates=["observed_at"])
    df_inc = pd.read_sql(text(SILVER_INCIDENTS_QUERY), engine, parse_dates=["occurred_at"])
    df_maint = pd.read_sql(text(SILVER_MAINTENANCE_QUERY), engine, parse_dates=["performed_at"])
    df_machine = pd.read_sql(text(MACHINE_QUERY), engine)

    gold = to_gold_rows(build_gold_features(df_sensors, df_inc, df_maint, df_machine))

    # Le batch 'gold' est aussi ce que lit le feu vert n°5 du réentraînement
    # (flows.retrain_flow.check_new_terrain_batch).
    batch_id = open_batch(
        engine, "gold", "silver_sensor_reading + silver_incident + silver_maintenance"
    )
    replace_table(
        engine, "gold_machine_hourly_feature", gold.assign(ingestion_batch_id=str(batch_id))
    )
    close_batch(engine, batch_id, rows_read=len(gold), rows_loaded=len(gold))
    logger.info("gold_machine_hourly_feature : %d lignes", len(gold))
    return len(gold)


@flow(name="indusense-etl")
def etl_flow(engine: Engine | None = None) -> dict:
    """Bronze -> Silver (3 tables) -> Gold. Les trois Silver sont
    indépendants ; le Gold les lit tous les trois."""
    engine = engine or get_engine()
    return {
        "silver_sensor_reading": rebuild_silver_sensor_reading(engine),
        "silver_incident": rebuild_silver_incident(engine),
        "silver_maintenance": rebuild_silver_maintenance(engine),
        "gold_machine_hourly_feature": rebuild_gold(engine),
    }


def main() -> None:
    result = etl_flow()
    print(f"ETL terminé : {result}")


if __name__ == "__main__":
    main()
