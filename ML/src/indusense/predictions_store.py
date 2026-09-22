"""Stockage idempotent des prédictions (module 29), rendu portable
SQLite/Postgres via SQLAlchemy (module 30) : un seul jeu de requêtes, pas
de branche `if backend == "postgres"`. `INSERT ... ON CONFLICT ... DO
UPDATE` est supporté nativement par SQLite (>=3.24) et PostgreSQL — même
syntaxe des deux côtés. `TEXT`/`DOUBLE PRECISION` : valides en Postgres,
acceptés par l'affinité de type de SQLite sans erreur.

Rejouer le flow deux fois sur les mêmes données laisse le même nombre de
lignes, jamais le double — clé composite (machine_id, window_start).
"""

from pathlib import Path

from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine

CREATE_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS predictions (
    machine_id TEXT NOT NULL,
    window_start TEXT NOT NULL,
    failure_proba_24h DOUBLE PRECISION NOT NULL,
    scored_at TEXT NOT NULL,
    PRIMARY KEY (machine_id, window_start)
)
"""

UPSERT_SQL = """
INSERT INTO predictions (machine_id, window_start, failure_proba_24h, scored_at)
VALUES (:machine_id, :window_start, :failure_proba_24h, :scored_at)
ON CONFLICT (machine_id, window_start) DO UPDATE SET
    failure_proba_24h = excluded.failure_proba_24h,
    scored_at = excluded.scored_at
"""


def get_default_engine(db_path: Path | None = None) -> Engine:
    """Moteur SQLite local par défaut — indusense.config.get_engine() donne
    l'équivalent Postgres (voir predict_flow.py, PREDICTIONS_DB_URL)."""
    db_path = Path(db_path) if db_path else Path("artifacts/predictions.db")
    db_path.parent.mkdir(parents=True, exist_ok=True)
    return create_engine(f"sqlite:///{db_path}")


def upsert_predictions(engine: Engine, rows: list[dict]) -> int:
    """Insère ou met à jour chaque ligne de `rows` (clés : machine_id,
    window_start, failure_proba_24h, scored_at). Retourne le nombre total
    de lignes en base après l'opération (compté indépendamment, pas déduit
    de `len(rows)`)."""
    with engine.begin() as conn:
        conn.execute(text(CREATE_TABLE_SQL))
        if rows:
            conn.execute(text(UPSERT_SQL), rows)
        return conn.execute(text("SELECT COUNT(*) FROM predictions")).scalar()


def count_predictions(engine: Engine) -> int:
    with engine.begin() as conn:
        conn.execute(text(CREATE_TABLE_SQL))
        return conn.execute(text("SELECT COUNT(*) FROM predictions")).scalar()
