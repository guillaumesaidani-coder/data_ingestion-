"""tests/test_flow_pipeline.py — le flow Prefect a besoin du Postgres réel
(build-gold-dataset) et du modèle déjà entraîné (ensure-model, reprise) :
marqué requires_local_infra comme test_etl_bronze_silver_gold.py, ne
tourne pas en CI. Le script scripts/demo_prefect_idempotence.py couvre le
même scénario pour une vérification manuelle rapide.
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from flows.pipeline import DEFAULT_GOLD_CSV, indusense_pipeline
from indusense.predictions_store import count_predictions

pytestmark = pytest.mark.requires_local_infra


def test_pipeline_replay_is_idempotent(tmp_path):
    db_path = tmp_path / "predictions.db"

    result1 = indusense_pipeline(gold_csv=DEFAULT_GOLD_CSV, predictions_db=db_path)
    n1 = count_predictions(db_path)

    result2 = indusense_pipeline(gold_csv=DEFAULT_GOLD_CSV, predictions_db=db_path)
    n2 = count_predictions(db_path)

    assert result1["rows_scored"] == result2["rows_scored"] > 0
    assert n1 == n2 == result1["rows_in_db"] == result2["rows_in_db"]


def test_pipeline_does_not_retrain_when_model_exists():
    from indusense.config import get_model_path

    model_path = get_model_path()
    assert model_path.exists(), "modèle déjà entraîné attendu (indusense train -o ...)"

    mtime_before = model_path.stat().st_mtime
    indusense_pipeline(gold_csv=DEFAULT_GOLD_CSV, predictions_db=Path("artifacts/predictions.db"))
    mtime_after = model_path.stat().st_mtime

    assert mtime_before == mtime_after  # jamais réécrit : reprise, pas réentraînement
