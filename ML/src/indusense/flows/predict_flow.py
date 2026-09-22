"""Module 29 (flow, cache, reprise) + module 30 (empaqueté, Postgres) :
flow Prefect orchestrant l'ETL Gold, l'entraînement (avec reprise), le
scoring des dernières mesures par machine, et le stockage idempotent des
prédictions — portable SQLite/Postgres (indusense.predictions_store).

Vit dans le paquet, pas à la racine du dépôt : le Dockerfile ne copie que
`src/` dans l'image (module 27, build multi-stage), donc c'est la seule
façon pour ce flow d'exister dans le conteneur. `ML/flows/pipeline.py`
n'est plus qu'une façade à trois lignes vers ce module — zéro logique
dupliquée entre l'exécution locale et `python -m indusense.flows.predict_flow`
dans le conteneur.

Rejouer ce flow ne doit jamais dupliquer de travail ni de données :
- build-gold-dataset est mis en cache tant que gold_machine_hourly_feature
  n'a pas changé (empreinte bon marché : nombre de lignes + dernière
  fenêtre — l'équivalent, pour une source Postgres, du hash mtime/taille
  qu'on utiliserait sur des fichiers plats).
- ensure-model saute l'entraînement si un modèle existe déjà sur disque
  (`retrain=False`, par défaut) — y compris le modèle déjà présent dans
  l'image Docker (module 27).
- store-predictions upsert par (machine_id, window_start) : un 2e passage
  laisse le même nombre de lignes, jamais le double, que le backend soit
  SQLite (local) ou PostgreSQL (conteneur, PREDICTIONS_DB_URL).

Usage local : uv run --frozen python flows/pipeline.py
Usage conteneur : docker compose run --rm --no-deps api python -m indusense.flows.predict_flow
(`--frozen` évite qu'uv ne tente de re-résoudre le lock avant de démarrer —
bloquant sans réseau sortant.)
"""

import hashlib
from datetime import UTC, datetime
from pathlib import Path

import joblib
import pandas as pd
from prefect import flow, get_run_logger, task
from prefect.cache_policies import NO_CACHE
from sqlalchemy import text

from indusense.config import get_engine, get_model_path, get_predictions_engine
from indusense.data import export_gold_dataset
from indusense.modeling.dataset import load_gold_dataset
from indusense.modeling.train import (
    B11_PARAMS,
    compute_scale_pos_weight,
    train_and_evaluate,
)
from indusense.predictions_store import upsert_predictions

DEFAULT_GOLD_CSV = Path("data/gold/gold_dataset.csv")


def _gold_freshness_cache_key(context, parameters) -> str:
    """Empreinte bon marché de l'état de gold_machine_hourly_feature (compte
    de lignes + dernière fenêtre) — l'équivalent, pour une source Postgres,
    du hash mtime/taille qu'on utiliserait sur des fichiers plats. Hachée :
    Prefect utilise cette clé comme nom de fichier de résultat, et le
    contenu brut (":", espaces) n'est pas un nom de fichier Windows valide."""
    engine = get_engine()
    with engine.connect() as conn:
        count, max_window = conn.execute(
            text("SELECT count(*), max(window_start) FROM gold_machine_hourly_feature")
        ).fetchone()
    raw = f"{count}:{max_window}:{parameters.get('out_path')}"
    return hashlib.sha256(raw.encode()).hexdigest()


@task(name="build-gold-dataset", cache_key_fn=_gold_freshness_cache_key)
def build_gold_dataset(out_path: Path = DEFAULT_GOLD_CSV) -> Path:
    logger = get_run_logger()
    result = export_gold_dataset(get_engine(), out_path)
    logger.info("Gold exporté : %s", result)
    return result


@task(name="ensure-model")
def ensure_model(model_path: Path | None = None, retrain: bool = False):
    logger = get_run_logger()
    model_path = Path(model_path) if model_path else get_model_path()

    if model_path.exists() and not retrain:
        logger.info("Reprise : modele existant reutilise -> %s", model_path)
        return joblib.load(model_path)

    gold = load_gold_dataset(get_engine())
    params = {**B11_PARAMS, "scale_pos_weight": compute_scale_pos_weight(gold.y_tv)}
    pipe, metrics = train_and_evaluate(gold.X_tv, gold.y_tv, gold.X_test, gold.y_test, params)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipe, model_path)
    logger.info(
        "Modele entraine et sauvegarde -> %s (PR-AUC test=%s)", model_path, metrics["pr_auc_test"]
    )
    return pipe


@task(name="load-latest-features")
def load_latest_features(gold_csv: Path) -> pd.DataFrame:
    df = pd.read_csv(gold_csv)
    df["window_start"] = pd.to_datetime(df["window_start"])
    return df.sort_values("window_start").groupby("machine_id", as_index=False).tail(1)


@task(name="predict-latest")
def predict_latest(model, latest: pd.DataFrame) -> pd.DataFrame:
    imputer = model.named_steps.get("imputer")
    feature_cols = (
        list(imputer.feature_names_in_)
        if imputer is not None and hasattr(imputer, "feature_names_in_")
        else [c for c in latest.columns if c not in ("machine_id", "window_start")]
    )
    proba = model.predict_proba(latest.reindex(columns=feature_cols))[:, 1]

    out = latest[["machine_id", "window_start"]].copy()
    out["failure_proba_24h"] = proba
    out["scored_at"] = datetime.now(UTC).isoformat()
    return out


@task(name="store-predictions", cache_policy=NO_CACHE)
def store_predictions(predictions: pd.DataFrame, engine=None) -> int:
    # cache_policy=NO_CACHE : cette task reçoit un Engine SQLAlchemy (verrou
    # de thread interne, non sérialisable) — Prefect essaierait sinon de le
    # hacher pour sa politique de cache par défaut et échouerait à chaque
    # appel (HashError, non fatal mais loggé en erreur à chaque run). Cette
    # task doit de toute façon toujours s'exécuter : c'est elle qui écrit.
    logger = get_run_logger()
    engine = engine or get_predictions_engine()
    rows = [
        {
            "machine_id": row.machine_id,
            "window_start": row.window_start.isoformat(),
            "failure_proba_24h": float(row.failure_proba_24h),
            "scored_at": row.scored_at,
        }
        for row in predictions.itertuples()
    ]
    rows_in_db = upsert_predictions(engine, rows)
    logger.info(
        "%d prédictions upsertées -> %s (%d lignes en base)", len(rows), engine.url, rows_in_db
    )
    return rows_in_db


def _flow_run_name() -> str:
    from prefect.runtime import flow_run

    gold_csv = flow_run.parameters.get("gold_csv", DEFAULT_GOLD_CSV)
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    return f"indusense-pipeline-{Path(gold_csv).stem}-{timestamp}"


@flow(name="indusense-pipeline", flow_run_name=_flow_run_name)
def indusense_pipeline(
    gold_csv: Path = DEFAULT_GOLD_CSV,
    model_path: Path | None = None,
    predictions_engine=None,
    retrain: bool = False,
) -> dict:
    gold_path = build_gold_dataset(gold_csv)
    model = ensure_model(model_path, retrain=retrain)
    latest = load_latest_features(gold_path)
    predictions = predict_latest(model, latest)
    engine = predictions_engine or get_predictions_engine()
    rows_in_db = store_predictions(predictions, engine)

    return {
        "rows_scored": len(predictions),
        "rows_in_db": rows_in_db,
        "db_url": str(engine.url),
    }


def main() -> None:
    result = indusense_pipeline()
    print(f"Pipeline terminé : {result}")


if __name__ == "__main__":
    main()
