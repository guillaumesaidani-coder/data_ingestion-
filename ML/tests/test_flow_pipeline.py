"""tests/test_flow_pipeline.py — le flow Prefect a besoin du Postgres réel
(build-gold-dataset) et du modèle déjà entraîné (ensure-model, reprise) :
marqué requires_local_infra comme test_etl_bronze_silver_gold.py, ne
tourne pas en CI. scripts/demo_prefect_idempotence.py couvre le même
scénario en local ; le conteneur (docker compose run ... api python -m
indusense.flows.predict_flow) le rejoue contre un vrai Postgres — voir
pipeline_proof.md.
"""

import sys
from pathlib import Path

import pytest
from sqlalchemy import create_engine

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from indusense.flows.predict_flow import DEFAULT_GOLD_CSV, indusense_pipeline
from indusense.predictions_store import count_predictions

pytestmark = pytest.mark.requires_local_infra


def test_pipeline_replay_is_idempotent(tmp_path):
    engine = create_engine(f"sqlite:///{tmp_path / 'predictions.db'}")

    result1 = indusense_pipeline(gold_csv=DEFAULT_GOLD_CSV, predictions_engine=engine)
    n1 = count_predictions(engine)

    result2 = indusense_pipeline(gold_csv=DEFAULT_GOLD_CSV, predictions_engine=engine)
    n2 = count_predictions(engine)

    assert result1["rows_scored"] == result2["rows_scored"] > 0
    assert n1 == n2 == result1["rows_in_db"] == result2["rows_in_db"]


def test_pipeline_does_not_retrain_when_model_exists(tmp_path):
    from indusense.config import get_model_path

    model_path = get_model_path()
    assert model_path.exists(), "modèle déjà entraîné attendu (indusense train -o ...)"

    mtime_before = model_path.stat().st_mtime
    engine = create_engine(f"sqlite:///{tmp_path / 'predictions.db'}")
    indusense_pipeline(gold_csv=DEFAULT_GOLD_CSV, predictions_engine=engine)
    mtime_after = model_path.stat().st_mtime

    assert mtime_before == mtime_after  # jamais réécrit : reprise, pas réentraînement
