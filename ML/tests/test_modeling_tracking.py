"""log_training_run()/write_metrics_and_params() : le lien manquant entre
un run MLflow et la version de données (gold_md5) qui l'a produit.

Utilise une base MLflow sqlite temporaire (pas mlflow_tp7.db) : ce test
crée un vrai run, il ne doit jamais toucher l'historique réel.
"""

import json
import sys
from pathlib import Path

import mlflow
import pytest
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from indusense.modeling.tracking import log_training_run, write_metrics_and_params


@pytest.fixture()
def isolated_mlflow(tmp_path, monkeypatch):
    tracking_uri = f"sqlite:///{tmp_path / 'mlflow_test.db'}"
    monkeypatch.setattr("indusense.modeling.tracking.get_mlflow_tracking_uri", lambda: tracking_uri)
    yield tracking_uri
    mlflow.end_run()


def test_log_training_run_records_params_metrics_and_gold_md5(isolated_mlflow):
    run_id = log_training_run(
        run_name="test-run",
        params={"max_depth": 3, "random_state": 42},
        metrics={"pr_auc_test": 0.87, "tp": 10},
        gold_md5="abc123",
        gold_dataset_path="data/gold/gold_dataset.csv",
    )

    client = mlflow.tracking.MlflowClient(tracking_uri=isolated_mlflow)
    run = client.get_run(run_id)

    assert run.data.params["max_depth"] == "3"
    assert run.data.metrics["pr_auc_test"] == pytest.approx(0.87)
    assert run.data.tags["gold_md5"] == "abc123"
    assert run.data.tags["gold_dataset"] == "data/gold/gold_dataset.csv"


def test_log_training_run_without_gold_md5_sets_no_tag(isolated_mlflow):
    run_id = log_training_run(
        run_name="test-run-no-data-version",
        params={"max_depth": 3},
        metrics={"pr_auc_test": 0.5},
    )
    client = mlflow.tracking.MlflowClient(tracking_uri=isolated_mlflow)
    run = client.get_run(run_id)
    assert "gold_md5" not in run.data.tags


def test_write_metrics_and_params(tmp_path):
    metrics_path, params_path = write_metrics_and_params(
        metrics={"pr_auc_test": 0.87},
        params={"max_depth": 3},
        out_dir=tmp_path,
        gold_md5="abc123",
    )
    assert json.loads(metrics_path.read_text(encoding="utf-8")) == {"pr_auc_test": 0.87}
    params_out = yaml.safe_load(params_path.read_text(encoding="utf-8"))
    assert params_out == {"max_depth": 3, "gold_md5": "abc123"}
