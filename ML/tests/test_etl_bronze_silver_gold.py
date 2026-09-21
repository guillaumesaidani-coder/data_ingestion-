"""Vérifie que le pipeline Bronze -> Silver -> Gold (TP1/TP1bis, TP2, TP6)
produit toujours, dans la base réelle, ce que la documentation
(pipeline_artifacts/ANALYSE_ARTIFACTS.md) et le schéma (ML/models.py)
décrivent. Aucune logique de transformation n'est ré-implémentée ici.
"""
import pandas as pd
from sqlalchemy import text

from conftest import ML_DIR


# ---------------------------------------------------------------------------
# Bronze : les tables reflètent les sources brutes sans perte ni ajout
# ---------------------------------------------------------------------------

def test_bronze_telemetry_row_count_matches_source_csv(db_engine):
    source_rows = len(pd.read_csv(ML_DIR / "telemetry.csv"))
    with db_engine.connect() as conn:
        db_rows = conn.execute(text("SELECT count(*) FROM bronze_telemetry")).scalar()
    assert db_rows == source_rows


def test_bronze_incidents_row_count_matches_source_csv(db_engine):
    source_rows = len(pd.read_csv(ML_DIR / "releves_incidents.csv"))
    with db_engine.connect() as conn:
        db_rows = conn.execute(text("SELECT count(*) FROM bronze_incidents")).scalar()
    assert db_rows == source_rows


def test_bronze_maintenance_types_are_valid(db_engine):
    with db_engine.connect() as conn:
        bad = conn.execute(text(
            "SELECT count(*) FROM bronze_maintenance "
            "WHERE maintenance_type NOT IN ('proactive', 'reactive')"
        )).scalar()
    assert bad == 0


# ---------------------------------------------------------------------------
# Silver : dédoublonnage et absence de valeurs manquantes non signalées
# ---------------------------------------------------------------------------

def test_silver_sensor_reading_has_no_duplicate_keys(db_engine):
    with db_engine.connect() as conn:
        dup = conn.execute(text("""
            SELECT count(*) FROM (
                SELECT machine_id, observed_at, sensor_type
                FROM silver_sensor_reading
                GROUP BY machine_id, observed_at, sensor_type
                HAVING count(*) > 1
            ) d
        """)).scalar()
    assert dup == 0


def test_silver_sensor_reading_null_value_always_flagged_missing(db_engine):
    with db_engine.connect() as conn:
        unflagged_nulls = conn.execute(text(
            "SELECT count(*) FROM silver_sensor_reading "
            "WHERE sensor_value IS NULL AND NOT is_missing"
        )).scalar()
    assert unflagged_nulls == 0


def test_silver_incident_codes_are_unique(db_engine):
    with db_engine.connect() as conn:
        total = conn.execute(text("SELECT count(*) FROM silver_incident")).scalar()
        distinct = conn.execute(text("SELECT count(DISTINCT incident_code) FROM silver_incident")).scalar()
    assert total == distinct


# ---------------------------------------------------------------------------
# Gold : dataset d'entraînement conforme à la documentation
# (ANALYSE_ARTIFACTS.md : 132 940 lignes x 91 colonnes ;
#  model_card.md : 112 996 obs. train+validation)
# ---------------------------------------------------------------------------

def test_gold_row_count_matches_documentation(db_engine):
    with db_engine.connect() as conn:
        n = conn.execute(text("SELECT count(*) FROM gold_machine_hourly_feature")).scalar()
    assert n == 132_940


def test_gold_trainval_row_count_matches_model_card(db_engine):
    with db_engine.connect() as conn:
        n = conn.execute(text(
            "SELECT count(*) FROM gold_machine_hourly_feature "
            "WHERE split_set IN ('train', 'validation')"
        )).scalar()
    assert n == 112_996


def test_gold_split_set_has_only_expected_values(db_engine):
    with db_engine.connect() as conn:
        values = {row[0] for row in conn.execute(text(
            "SELECT DISTINCT split_set FROM gold_machine_hourly_feature"
        ))}
    assert values == {"train", "validation", "test"}


def test_gold_label_columns_are_never_null(db_engine):
    with db_engine.connect() as conn:
        n_null = conn.execute(text(
            "SELECT count(*) FROM gold_machine_hourly_feature "
            "WHERE label_failure_next_24h IS NULL"
        )).scalar()
    assert n_null == 0


def test_gold_covers_all_15_machines(db_engine):
    with db_engine.connect() as conn:
        n_machine_table = conn.execute(text("SELECT count(*) FROM machine")).scalar()
        n_machine_gold = conn.execute(text(
            "SELECT count(DISTINCT machine_id) FROM gold_machine_hourly_feature"
        )).scalar()
    assert n_machine_table == 15
    assert n_machine_gold == 15


def test_gold_windows_are_chronologically_ordered_per_machine(db_engine):
    """Anti-fuite de base : à l'intérieur d'une machine, window_start est strictement croissant
    (pas de doublon ni d'inversion de fenêtre)."""
    with db_engine.connect() as conn:
        offenders = conn.execute(text("""
            SELECT count(*) FROM (
                SELECT machine_id, window_start,
                       lag(window_start) OVER (PARTITION BY machine_id ORDER BY window_start) AS prev_start
                FROM gold_machine_hourly_feature
            ) w
            WHERE prev_start IS NOT NULL AND window_start <= prev_start
        """)).scalar()
    assert offenders == 0
