"""R1 (stack unique) : fabrique les fichiers d'initialisation de la base
de la stack compose, à partir de la base de développement déjà peuplée.

Deux fichiers, joués par l'image postgres au PREMIER démarrage du volume
(`/docker-entrypoint-initdb.d`, ordre alphabétique) :

- 01_indusense_seed.sql.gz : pg_dump de la base (schéma alembic, Bronze,
  Silver, Gold, référentiel machines, registre ingestion_batch) ;
- 02_predictions.sql.gz : table `predictions` recopiée depuis la base
  SQLite locale (artifacts/predictions.db), avec les verdicts
  techniciens — pour qu'API, Streamlit et Grafana lisent la même base.

Le dump est volumineux et régénérable : il est versionné par DVC
(docker/initdb.dvc), pas par Git.

Usage : uv run --frozen python scripts/build_seed_dump.py [--container docker-db-1]
"""

import argparse
import csv
import gzip
import io
import sqlite3
import subprocess
from pathlib import Path

ML_DIR = Path(__file__).resolve().parent.parent
OUT_DIR = ML_DIR / "docker" / "initdb"
SQLITE_PREDICTIONS = ML_DIR / "artifacts" / "predictions.db"

PREDICTION_COLS = [
    "machine_id",
    "window_start",
    "failure_proba_24h",
    "scored_at",
    "model_version",
    "features_payload",
    "review_status",
    "ground_truth",
    "reviewer_comment",
    "reviewed_at",
]

# Même schéma que indusense.predictions_store.CREATE_TABLE_SQL (non importé
# ici : le script ne doit dépendre que de la stdlib et de Docker).
CREATE_PREDICTIONS_SQL = """
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
);
"""


def dump_database(container: str, user: str, database: str) -> Path:
    out = OUT_DIR / "01_indusense_seed.sql.gz"
    dump = subprocess.run(
        [
            "docker",
            "exec",
            container,
            "pg_dump",
            "-U",
            user,
            "-d",
            database,
            "--no-owner",
            "--no-privileges",
        ],
        check=True,
        capture_output=True,
    ).stdout
    with gzip.open(out, "wb", compresslevel=6) as f:
        f.write(dump)
    return out


def dump_predictions() -> tuple[Path, int]:
    out = OUT_DIR / "02_predictions.sql.gz"
    with sqlite3.connect(SQLITE_PREDICTIONS) as conn:
        rows = conn.execute(f"SELECT {', '.join(PREDICTION_COLS)} FROM predictions").fetchall()

    buffer = io.StringIO()
    writer = csv.writer(buffer, lineterminator="\n")
    gt = PREDICTION_COLS.index("ground_truth")
    for row in rows:
        row = list(row)
        # SQLite stocke les booléens en 0/1 ; NULL -> champ vide (NULL en CSV COPY).
        if row[gt] is not None:
            row[gt] = "true" if row[gt] else "false"
        writer.writerow(["" if v is None else v for v in row])

    with gzip.open(out, "wt", encoding="utf-8") as f:
        f.write(CREATE_PREDICTIONS_SQL)
        f.write(f"COPY predictions ({', '.join(PREDICTION_COLS)}) FROM stdin WITH (FORMAT csv);\n")
        f.write(buffer.getvalue())
        f.write("\\.\n")
    return out, len(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--container", default="docker-db-1")
    parser.add_argument("--user", default="indusense_user")
    parser.add_argument("--database", default="indusense_db")
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    seed = dump_database(args.container, args.user, args.database)
    predictions, n = dump_predictions()
    print(f"{seed.relative_to(ML_DIR)} : {seed.stat().st_size / 1e6:.1f} Mo")
    print(f"{predictions.relative_to(ML_DIR)} : {n} prédictions")
    print("Suite : uv run dvc add docker/initdb  (puis dvc push)")


if __name__ == "__main__":
    main()
