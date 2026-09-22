"""Stockage idempotent des prédictions (module 29) : rejouer le flow deux
fois sur les mêmes données laisse le même nombre de lignes, jamais le
double. Clé composite (machine_id, window_start) + upsert SQL.

`closing()` explicite : `with sqlite3.connect(...) as conn` ne fait que
committer/annuler la transaction, il ne ferme pas la connexion — sous
Windows le fichier reste verrouillé (PermissionError au nettoyage d'un
dossier temporaire, par ex.). `closing()` garantit la fermeture réelle.
"""

import sqlite3
from contextlib import closing
from pathlib import Path

CREATE_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS predictions (
    machine_id TEXT NOT NULL,
    window_start TEXT NOT NULL,
    failure_proba_24h REAL NOT NULL,
    scored_at TEXT NOT NULL,
    PRIMARY KEY (machine_id, window_start)
)
"""

UPSERT_SQL = """
INSERT INTO predictions (machine_id, window_start, failure_proba_24h, scored_at)
VALUES (:machine_id, :window_start, :failure_proba_24h, :scored_at)
ON CONFLICT(machine_id, window_start) DO UPDATE SET
    failure_proba_24h = excluded.failure_proba_24h,
    scored_at = excluded.scored_at
"""


def upsert_predictions(db_path: Path, rows: list[dict]) -> int:
    """Insère ou met à jour chaque ligne de `rows` (clés : machine_id,
    window_start, failure_proba_24h, scored_at). Retourne le nombre total
    de lignes en base après l'opération (compté indépendamment, pas déduit
    de `len(rows)`)."""
    db_path = Path(db_path)
    db_path.parent.mkdir(parents=True, exist_ok=True)

    with closing(sqlite3.connect(db_path)) as conn:
        conn.execute(CREATE_TABLE_SQL)
        conn.executemany(UPSERT_SQL, rows)
        conn.commit()
        return conn.execute("SELECT COUNT(*) FROM predictions").fetchone()[0]


def count_predictions(db_path: Path) -> int:
    with closing(sqlite3.connect(db_path)) as conn:
        conn.execute(CREATE_TABLE_SQL)
        return conn.execute("SELECT COUNT(*) FROM predictions").fetchone()[0]
