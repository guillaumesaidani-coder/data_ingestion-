"""R1 (stack unique) : fige une photo de la base de la stack compose
(ml-db-1) en fichiers d'initialisation, joués par l'image postgres au
PREMIER démarrage du volume (`/docker-entrypoint-initdb.d`, ordre
alphabétique) :

- 01_indusense_seed.sql.gz : pg_dump de la base SANS la table
  `predictions` (schéma alembic, Bronze, Silver, Gold, référentiels,
  registre ingestion_batch) ;
- 02_predictions.sql.gz : la table `predictions`, scores et verdicts
  techniciens, lue dans la même base.

Les deux fichiers viennent de la même base, au même moment : la table
`predictions` n'est écrite qu'une fois. Jusqu'ici 01 venait de Postgres
et 02 de l'ancien SQLite (artifacts/predictions.db) : un dump de ml-db-1,
qui contient déjà `predictions`, aurait produit la table deux fois et
fait échouer l'initialisation (clé primaire en double).

`--check` ne dumpe rien : il liste les verdicts présents dans la base mais
absents de 02, c'est-à-dire ce qu'un `docker compose down -v` effacerait.

Le dump est volumineux et régénérable : il est versionné par DVC
(docker/initdb.dvc), pas par Git.

Usage :
    uv run --frozen python scripts/build_seed_dump.py            # fige ml-db-1
    uv run --frozen python scripts/build_seed_dump.py --check    # avant un down -v
"""

import argparse
import csv
import gzip
import io
import subprocess
import sys
from pathlib import Path

ML_DIR = Path(__file__).resolve().parent.parent
OUT_DIR = ML_DIR / "docker" / "initdb"
SEED_FILE = OUT_DIR / "01_indusense_seed.sql.gz"
PREDICTIONS_FILE = OUT_DIR / "02_predictions.sql.gz"

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
REVIEW_COLS = [
    "machine_id",
    "window_start",
    "review_status",
    "ground_truth",
    "reviewer_comment",
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


def _docker_exec(container: str, *cmd: str) -> bytes:
    return subprocess.run(
        ["docker", "exec", container, *cmd], check=True, capture_output=True
    ).stdout


def dump_database(container: str, user: str, database: str) -> Path:
    dump = _docker_exec(
        container,
        "pg_dump",
        "-U",
        user,
        "-d",
        database,
        "--no-owner",
        "--no-privileges",
        # Écrite une seule fois, dans 02 : jamais dans les deux fichiers.
        "--exclude-table=public.predictions",
    )
    with gzip.open(SEED_FILE, "wb", compresslevel=6) as f:
        f.write(dump)
    return SEED_FILE


def export_predictions_csv(container: str, user: str, database: str, cols: list[str]) -> str:
    """Table `predictions` de la base, en CSV (format COPY : NULL = champ
    vide, booléens en t/f)."""
    query = f"COPY (SELECT {', '.join(cols)} FROM predictions ORDER BY machine_id, window_start) TO STDOUT WITH (FORMAT csv)"
    return _docker_exec(container, "psql", "-U", user, "-d", database, "-c", query).decode("utf-8")


def dump_predictions(container: str, user: str, database: str) -> tuple[Path, int]:
    body = export_predictions_csv(container, user, database, PREDICTION_COLS)
    with gzip.open(PREDICTIONS_FILE, "wt", encoding="utf-8") as f:
        f.write(CREATE_PREDICTIONS_SQL)
        f.write(f"COPY predictions ({', '.join(PREDICTION_COLS)}) FROM stdin WITH (FORMAT csv);\n")
        f.write(body)
        f.write("\\.\n")
    return PREDICTIONS_FILE, sum(1 for _ in csv.reader(io.StringIO(body)))


def _normalize_bool(value: str) -> str:
    # Postgres exporte t/f, l'ancien seed (converti de SQLite) true/false.
    return {"t": "true", "f": "false"}.get(value, value)


def _review_keys(rows) -> set[tuple[str, ...]]:
    """Un verdict = une prédiction revue, identifiée par sa clé et son
    contenu (un verdict modifié compte comme nouveau)."""
    gt = REVIEW_COLS.index("ground_truth")
    status = REVIEW_COLS.index("review_status")
    keys = set()
    for row in rows:
        if row[status] == "A_VALIDER":
            continue
        row = list(row)
        row[gt] = _normalize_bool(row[gt])
        keys.add(tuple(row))
    return keys


def seeded_reviews(path: Path = PREDICTIONS_FILE) -> set[tuple[str, ...]]:
    """Verdicts figés dans 02 : lignes du bloc COPY, lues avec les colonnes
    annoncées par l'en-tête COPY du fichier."""
    with gzip.open(path, "rt", encoding="utf-8") as f:
        lines = iter(f)
        for line in lines:
            if line.startswith("COPY predictions ("):
                cols = [c.strip() for c in line.split("(", 1)[1].split(")", 1)[0].split(",")]
                break
        else:
            raise ValueError(f"{path} : aucun bloc COPY predictions")
        body = []
        for line in lines:
            if line.startswith("\\."):
                break
            body.append(line)
    idx = [cols.index(c) for c in REVIEW_COLS]
    return _review_keys([row[i] for i in idx] for row in csv.reader(body))


def unsaved_reviews(container: str, user: str, database: str) -> set[tuple[str, ...]]:
    live = export_predictions_csv(container, user, database, REVIEW_COLS)
    return _review_keys(csv.reader(io.StringIO(live))) - seeded_reviews()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--container", default="ml-db-1")
    parser.add_argument("--user", default="indusense_user")
    parser.add_argument("--database", default="indusense_db")
    parser.add_argument(
        "--check",
        action="store_true",
        help="Ne dumpe rien : échoue si la base contient des verdicts absents du seed",
    )
    args = parser.parse_args()

    if args.check:
        missing = unsaved_reviews(args.container, args.user, args.database)
        if missing:
            print(
                f"{len(missing)} verdict(s) de {args.container} absent(s) du seed : "
                "un down -v les effacerait. Régénérer d'abord : make seed-dump"
            )
            return 1
        print(f"Seed à jour : tous les verdicts de {args.container} y sont.")
        return 0

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    seed = dump_database(args.container, args.user, args.database)
    predictions, n = dump_predictions(args.container, args.user, args.database)
    print(f"{seed.relative_to(ML_DIR)} : {seed.stat().st_size / 1e6:.1f} Mo")
    print(f"{predictions.relative_to(ML_DIR)} : {n} prédictions")
    print("Suite : uv run dvc add docker/initdb  (puis dvc push)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
