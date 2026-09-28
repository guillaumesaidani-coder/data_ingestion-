"""Ingestion de lots terrain (Bronze) tracés par `ingestion_batch`.

Factorise le pattern répété dans TP4.ipynb/TP5.ipynb/TP6.ipynb (ouvrir un
batch -> charger/transformer -> clôturer) pour permettre d'injecter un
nouveau lot de données terrain de façon reproductible, en dehors d'une
ré-exécution manuelle de notebook.

Contrairement au rebuild complet de TP4.ipynb (TRUNCATE + réécriture),
`ingest_terrain_batch()` fait un append : chaque appel ajoute un lot
traçable sans toucher aux données déjà en base -- c'est ce qui permet
d'injecter un nouveau lot pour challenger le modèle sans tout regénérer.

`ingestion_batch.content_hash` distingue un vrai nouveau fichier terrain
d'un simple rejeu du même fichier -- même algorithme MD5 que
`indusense.data.hash_file()`, pour rester cohérent avec le hash déjà
utilisé côté export Gold (DVC/MLflow).

Chaque ligne est validée comme dans TP4 (`processing.bronze_validation`) :
une ligne invalide est quand même écrite, avec `parse_ok=False`, et
tracée dans `data_quality_issue` -- le Silver ne la lira pas.

Un lot d'incidents met aussi à jour le référentiel `operator`
(`processing.operators`, comme TP4 §4) : sans ça, un opérateur nouveau
dans le lot resterait sans `operator_id` au Silver.
"""

import uuid
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd
from sqlalchemy import text
from sqlalchemy.engine import Connection, Engine

from indusense.data import hash_file
from indusense.processing.bronze_validation import data_quality_rows, validate_bronze
from indusense.processing.operators import UPSERT_OPERATOR_SQL, build_operators

BRONZE_TABLES = {
    "telemetry": "bronze_telemetry",
    "incidents": "bronze_incidents",
    "maintenance": "bronze_maintenance",
}


def open_batch(
    engine: Engine, source_name: str, source_file: str, content_hash: str | None = None
) -> uuid.UUID:
    """Ouvre un batch (status='running'). Même requête que TP4/TP5/TP6,
    factorisée -- l'id est généré côté Python (pas `gen_random_uuid()`
    serveur) pour que la même fonction tourne contre SQLite (tests) comme
    contre Postgres (module 30)."""
    batch_id = uuid.uuid4()
    with engine.begin() as conn:
        conn.execute(
            text(
                "INSERT INTO ingestion_batch "
                "(ingestion_batch_id, source_name, source_file, content_hash, started_at, status) "
                "VALUES (:id, :source_name, :source_file, :content_hash, :started_at, 'running')"
            ),
            {
                "id": str(batch_id),
                "source_name": source_name,
                "source_file": source_file,
                "content_hash": content_hash,
                "started_at": datetime.now(UTC).isoformat(),
            },
        )
    return batch_id


def close_batch(
    engine: Engine,
    batch_id: uuid.UUID,
    rows_read: int,
    rows_loaded: int,
    rows_rejected: int = 0,
) -> None:
    """Clôture un batch (status='done'). Même requête que TP6.ipynb (section
    « Clôture batch »)."""
    with engine.begin() as conn:
        _close_batch(conn, batch_id, rows_read, rows_loaded, rows_rejected)


def _close_batch(
    conn: Connection, batch_id: uuid.UUID, rows_read: int, rows_loaded: int, rows_rejected: int
) -> None:
    conn.execute(
        text(
            "UPDATE ingestion_batch "
            "SET finished_at=:finished_at, rows_read=:rows_read, rows_loaded=:rows_loaded, "
            "rows_rejected=:rows_rejected, status='done' "
            "WHERE ingestion_batch_id=:id"
        ),
        {
            "finished_at": datetime.now(UTC).isoformat(),
            "rows_read": rows_read,
            "rows_loaded": rows_loaded,
            "rows_rejected": rows_rejected,
            "id": str(batch_id),
        },
    )


def find_batch_by_hash(engine: Engine, source_name: str, content_hash: str) -> uuid.UUID | None:
    """Détecte un rejeu du même fichier terrain (même source + même hash)
    déjà clos avec succès -- évite de dupliquer silencieusement les lignes
    Bronze à chaque injection répétée du même CSV."""
    with engine.connect() as conn:
        row = conn.execute(
            text(
                "SELECT ingestion_batch_id FROM ingestion_batch "
                "WHERE source_name=:source_name AND content_hash=:content_hash AND status='done' "
                "ORDER BY started_at DESC LIMIT 1"
            ),
            {"source_name": source_name, "content_hash": content_hash},
        ).fetchone()
    return uuid.UUID(str(row[0])) if row else None


def ingest_terrain_batch(engine: Engine, csv_path: Path, source_name: str) -> uuid.UUID:
    """Injecte un CSV terrain dans la table Bronze correspondante, en append.
    Rejouer le même fichier (même hash) réutilise le batch existant plutôt
    que de dupliquer les lignes -- idempotent comme `upsert_predictions()`
    (predictions_store.py), mais côté Bronze."""
    csv_path = Path(csv_path)
    if source_name not in BRONZE_TABLES:
        raise ValueError(f"source_name inconnu : {source_name} (attendu {list(BRONZE_TABLES)})")

    content_hash = hash_file(csv_path)
    existing = find_batch_by_hash(engine, source_name, content_hash)
    if existing is not None:
        return existing

    # Lu en texte, comme TP4 : c'est Pydantic qui décide si "12,5" ou "abc"
    # est un nombre, pas l'inférence de type de pandas.
    raw = pd.read_csv(csv_path, dtype=str)
    bronze = validate_bronze(raw, source_name)

    # Ouvert dans sa propre transaction : la tentative reste tracée
    # (status 'running') même si le chargement échoue.
    batch_id = open_batch(engine, source_name, csv_path.name, content_hash)
    bronze["ingestion_batch_id"] = str(batch_id)
    issues = data_quality_rows(bronze, source_name, batch_id)
    n_ok = int(bronze["parse_ok"].sum())
    operators = build_operators(bronze) if source_name == "incidents" else None

    # Lignes Bronze, anomalies et clôture dans UNE transaction. Sinon un
    # échec en cours de route laisse des lignes Bronze d'un batch jamais
    # clos : find_batch_by_hash ne voit que les batches 'done', le rejeu du
    # fichier les ajoute une 2e fois, et le Silver (qui lit tout le Bronze
    # parse_ok) les reçoit en double.
    with engine.begin() as conn:
        bronze.to_sql(BRONZE_TABLES[source_name], conn, if_exists="append", index=False)
        if issues:
            pd.DataFrame(issues).to_sql("data_quality_issue", conn, if_exists="append", index=False)
        if operators is not None and not operators.empty:
            conn.execute(text(UPSERT_OPERATOR_SQL), operators.to_dict("records"))
        _close_batch(conn, batch_id, len(bronze), n_ok, len(bronze) - n_ok)
    return batch_id
