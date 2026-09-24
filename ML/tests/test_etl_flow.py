"""Flow ETL (rupture ① du cadrage) contre la vraie base : reconstruit
Silver et Gold à partir du Bronze, sans notebook.

Critères prouvés ici :
- le Gold reconstruit est identique à celui de TP6, figé dans
  data/gold/gold_dataset.csv (DVC, md5 47f3185b…), à la colonne
  `ingestion_batch_id` près — un UUID neuf à chaque reconstruction ;
- rejouer le flow ne duplique rien.

Réécrit réellement Silver et Gold dans indusense_db (TRUNCATE +
réécriture) : requires_local_infra, et lent (le Gold est reconstruit
deux fois).
"""

import pandas as pd
import pytest
from conftest import ML_DIR
from sqlalchemy import text

from indusense.data import GOLD_QUERY
from indusense.flows.etl_flow import etl_flow

pytestmark = pytest.mark.requires_local_infra

REFERENCE_GOLD_CSV = ML_DIR / "data" / "gold" / "gold_dataset.csv"
TABLES = [
    "silver_sensor_reading",
    "silver_incident",
    "silver_maintenance",
    "gold_machine_hourly_feature",
]


def _counts(engine) -> dict[str, int]:
    with engine.connect() as conn:
        return {t: conn.execute(text(f"SELECT count(*) FROM {t}")).scalar() for t in TABLES}


def _gold_batches(engine) -> int:
    with engine.connect() as conn:
        return conn.execute(
            text("SELECT count(*) FROM ingestion_batch WHERE source_name = 'gold'")
        ).scalar()


def test_etl_flow_rebuilds_tp6_gold_and_is_idempotent(db_engine, tmp_path):
    if not REFERENCE_GOLD_CSV.exists():
        pytest.skip("data/gold/gold_dataset.csv absent : lancer `dvc pull` d'abord")

    gold_batches_before = _gold_batches(db_engine)

    first = etl_flow(db_engine)
    counts_first = _counts(db_engine)
    second = etl_flow(db_engine)
    counts_second = _counts(db_engine)

    assert first == second == counts_first == counts_second
    assert counts_first["gold_machine_hourly_feature"] == 132_940
    assert _gold_batches(db_engine) == gold_batches_before + 2

    rebuilt_csv = tmp_path / "gold_rebuilt.csv"
    pd.read_sql(GOLD_QUERY, db_engine).to_csv(rebuilt_csv, index=False)
    rebuilt = pd.read_csv(rebuilt_csv).drop(columns=["ingestion_batch_id"])
    reference = pd.read_csv(REFERENCE_GOLD_CSV).drop(columns=["ingestion_batch_id"])
    pd.testing.assert_frame_equal(rebuilt, reference)
