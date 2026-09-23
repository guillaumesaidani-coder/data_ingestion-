"""Stockage idempotent des prédictions (module 29), rendu portable
SQLite/Postgres via SQLAlchemy (module 30) : un seul jeu de requêtes, pas
de branche `if backend == "postgres"`. `INSERT ... ON CONFLICT ... DO
UPDATE` est supporté nativement par SQLite (>=3.24) et PostgreSQL — même
syntaxe des deux côtés. `TEXT`/`DOUBLE PRECISION` : valides en Postgres,
acceptés par l'affinité de type de SQLite sans erreur.

Rejouer le flow deux fois sur les mêmes données laisse le même nombre de
lignes, jamais le double — clé composite (machine_id, window_start).

Module 35 (boucle HITL, plan_action_hitl_champion_challenger.md) : ajoute
`model_version` (indusense.config.get_model_version) et `features_payload`
(photo JSON des features au moment du scoring — jusqu'ici seule la
probabilité était gardée, impossible de rejouer une fenêtre passée contre
un nouveau modèle sans ça) et les colonnes de retour terrain
(review_status/ground_truth/reviewer_comment/reviewed_at), vides tant
qu'aucun technicien (réel ou simulé, module 36) n'a statué. `TEXT` pour
features_payload, pas `JSONB` : ce fichier reste un seul jeu de requêtes
pour les deux backends, comme le reste du module.

Piège évité sur l'upsert : `ON CONFLICT ... DO UPDATE` ne touche QUE les
colonnes de prédiction (probabilité, modèle, features, horodatage) —
jamais les colonnes de revue. Sans ça, rejouer le flow sur une fenêtre
déjà validée par un technicien effacerait silencieusement son verdict.
"""

import json
from pathlib import Path

from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine

CREATE_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS predictions (
    machine_id TEXT NOT NULL,
    window_start TEXT NOT NULL,
    failure_proba_24h DOUBLE PRECISION NOT NULL,
    scored_at TEXT NOT NULL,
    model_version TEXT NOT NULL,
    features_payload TEXT,
    review_status TEXT NOT NULL DEFAULT 'A_VALIDER',
    ground_truth BOOLEAN,
    reviewer_comment TEXT,
    reviewed_at TEXT,
    PRIMARY KEY (machine_id, window_start)
)
"""

UPSERT_SQL = """
INSERT INTO predictions (
    machine_id, window_start, failure_proba_24h, scored_at, model_version, features_payload
)
VALUES (
    :machine_id, :window_start, :failure_proba_24h, :scored_at, :model_version, :features_payload
)
ON CONFLICT (machine_id, window_start) DO UPDATE SET
    failure_proba_24h = excluded.failure_proba_24h,
    scored_at = excluded.scored_at,
    model_version = excluded.model_version,
    features_payload = excluded.features_payload
"""

REVIEW_SQL = """
UPDATE predictions
SET review_status = :review_status,
    ground_truth = :ground_truth,
    reviewer_comment = :reviewer_comment,
    reviewed_at = :reviewed_at
WHERE machine_id = :machine_id AND window_start = :window_start
"""


def get_default_engine(db_path: Path | None = None) -> Engine:
    """Moteur SQLite local par défaut — indusense.config.get_engine() donne
    l'équivalent Postgres (voir predict_flow.py, PREDICTIONS_DB_URL)."""
    db_path = Path(db_path) if db_path else Path("artifacts/predictions.db")
    db_path.parent.mkdir(parents=True, exist_ok=True)
    return create_engine(f"sqlite:///{db_path}")


def upsert_predictions(engine: Engine, rows: list[dict]) -> int:
    """Insère ou met à jour chaque ligne de `rows` (clés : machine_id,
    window_start, failure_proba_24h, scored_at, model_version,
    features_payload — ce dernier un dict, sérialisé ici). Retourne le
    nombre total de lignes en base après l'opération (compté
    indépendamment, pas déduit de `len(rows)`)."""
    prepared = [{**row, "features_payload": json.dumps(row["features_payload"])} for row in rows]
    with engine.begin() as conn:
        conn.execute(text(CREATE_TABLE_SQL))
        if prepared:
            conn.execute(text(UPSERT_SQL), prepared)
        return conn.execute(text("SELECT COUNT(*) FROM predictions")).scalar()


def count_predictions(engine: Engine) -> int:
    with engine.begin() as conn:
        conn.execute(text(CREATE_TABLE_SQL))
        return conn.execute(text("SELECT COUNT(*) FROM predictions")).scalar()


def review_prediction(
    engine: Engine,
    machine_id: str,
    window_start: str,
    review_status: str,
    ground_truth: bool | None,
    reviewer_comment: str | None,
    reviewed_at: str,
) -> None:
    """Enregistre le verdict d'un technicien (réel, module 36 Streamlit,
    ou simulé, module 36 scripts/simulate_feedback.py) sur une prédiction
    déjà scorée. N'écrit rien sur la ligne si (machine_id, window_start)
    n'existe pas encore — rowcount vérifié plutôt qu'un succès supposé."""
    with engine.begin() as conn:
        conn.execute(text(CREATE_TABLE_SQL))
        result = conn.execute(
            text(REVIEW_SQL),
            {
                "machine_id": machine_id,
                "window_start": window_start,
                "review_status": review_status,
                "ground_truth": ground_truth,
                "reviewer_comment": reviewer_comment,
                "reviewed_at": reviewed_at,
            },
        )
        if result.rowcount == 0:
            raise ValueError(f"Aucune prédiction pour ({machine_id}, {window_start})")
