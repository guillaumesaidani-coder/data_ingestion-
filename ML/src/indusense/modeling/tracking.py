"""Trace un entraînement dans MLflow, avec un lien explicite vers la
version des données (gold_md5) quand elle est connue — c'est ce lien
qui manquait pour retracer quelle donnée a produit quel modèle (cf.
l'écart 67/69/78 features découvert sur le run b11).

Écrit aussi metrics.json / params.yaml en clair, versionnés par Git (pas
par DVC : ce sont de petits fichiers texte, lisibles en diff).
"""

import json
from pathlib import Path

import mlflow
import yaml

from indusense.config import get_mlflow_tracking_uri

EXPERIMENT_NAME = "indusense_training"


def log_training_run(
    run_name: str,
    params: dict,
    metrics: dict,
    gold_md5: str | None = None,
    gold_dataset_path: str | None = None,
) -> str:
    """Loggue un run MLflow (params + métriques + tag gold_md5 si fourni).
    Retourne le run_id créé."""
    mlflow.set_tracking_uri(get_mlflow_tracking_uri())
    mlflow.set_experiment(EXPERIMENT_NAME)

    with mlflow.start_run(run_name=run_name) as run:
        mlflow.log_params(params)
        mlflow.log_metrics({k: v for k, v in metrics.items() if isinstance(v, (int, float))})
        if gold_md5:
            mlflow.set_tag("gold_md5", gold_md5)
            mlflow.set_tag("gold_dataset", gold_dataset_path or "")
        return run.info.run_id


def write_metrics_and_params(
    metrics: dict, params: dict, out_dir: Path, gold_md5: str | None = None
) -> tuple[Path, Path]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    metrics_path = out_dir / "metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2, sort_keys=True), encoding="utf-8")

    params_out = dict(params)
    if gold_md5:
        params_out["gold_md5"] = gold_md5
    params_path = out_dir / "params.yaml"
    params_path.write_text(yaml.safe_dump(params_out, sort_keys=True), encoding="utf-8")

    return metrics_path, params_path
