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
"""

import uuid
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd
from sqlalchemy import text
from sqlalchemy.engine import Engine

from indusense.data import hash_file

BRONZE_TABLES = {
    "telemetry": "bronze_telemetry",
    "incidents": "bronze_incidents",
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

    df = pd.read_csv(csv_path)
    batch_id = open_batch(engine, source_name, csv_path.name, content_hash)
    df["ingestion_batch_id"] = str(batch_id)
    df.to_sql(BRONZE_TABLES[source_name], engine, if_exists="append", index=False)
    close_batch(engine, batch_id, rows_read=len(df), rows_loaded=len(df))
    return batch_id
