"""ingest_terrain_batch() : injection d'un lot terrain en Bronze, tracée
par `ingestion_batch` -- isolée d'un vrai Postgres via un moteur SQLite
temporaire portant un schéma minimal aligné sur les colonnes que ce module
touche réellement (voir alembic/versions/b74e16597ef9_create_bronze_tables.py,
cc83a31393a7_add_ingestion_batch_operator_dq.py et
e3a9f1c7b2d4_add_ingestion_batch_content_hash.py -- à tenir à jour si ces
migrations changent les colonnes utilisées ici).
"""

import sys
from pathlib import Path

import pandas as pd
from sqlalchemy import create_engine, text

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from indusense.ingest import (
    close_batch,
    find_batch_by_hash,
    ingest_terrain_batch,
    open_batch,
)

SCHEMA_SQL = """
CREATE TABLE ingestion_batch (
    ingestion_batch_id TEXT PRIMARY KEY,
    source_name TEXT,
    source_file TEXT,
    content_hash TEXT,
    started_at TEXT,
    finished_at TEXT,
    rows_read INTEGER,
    rows_loaded INTEGER,
    rows_rejected INTEGER,
    status TEXT
);
CREATE TABLE bronze_telemetry (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    machine_id TEXT,
    timestamp TEXT,
    temperature_c REAL,
    pressure_bar REAL,
    voltage_mean_v REAL,
    rotation_mean_rpm REAL,
    pieces_produced INTEGER,
    ingestion_batch_id TEXT
);
CREATE TABLE bronze_maintenance (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ingestion_batch_id TEXT,
    maintenance_id INTEGER,
    machine_id TEXT,
    maintenance_at TEXT,
    maintenance_type TEXT,
    action_type TEXT,
    component TEXT,
    description TEXT,
    related_incident_id TEXT,
    duration_hours REAL
);
"""


def _engine(tmp_path):
    engine = create_engine(f"sqlite:///{tmp_path / 'ingest.db'}")
    with engine.begin() as conn:
        for statement in SCHEMA_SQL.strip().split(";"):
            if statement.strip():
                conn.execute(text(statement))
    return engine


def _terrain_csv(tmp_path, name="terrain.csv", n_rows=3):
    path = tmp_path / name
    pd.DataFrame(
        {
            "machine_id": [f"MACH-0{i}" for i in range(1, n_rows + 1)],
            "timestamp": ["2026-06-15T00:00:00"] * n_rows,
            "temperature_c": [48.0] * n_rows,
            "pressure_bar": [195.0] * n_rows,
            "voltage_mean_v": [228.0] * n_rows,
            "rotation_mean_rpm": [1600.0] * n_rows,
            "pieces_produced": [42] * n_rows,
        }
    ).to_csv(path, index=False)
    return path


def test_open_and_close_batch_round_trip(tmp_path):
    engine = _engine(tmp_path)
    batch_id = open_batch(engine, "telemetry", "terrain.csv", content_hash="abc123")
    close_batch(engine, batch_id, rows_read=3, rows_loaded=3)

    with engine.connect() as conn:
        row = conn.execute(
            text(
                "SELECT status, rows_loaded, content_hash FROM ingestion_batch WHERE ingestion_batch_id=:id"
            ),
            {"id": str(batch_id)},
        ).fetchone()
    assert row.status == "done"
    assert row.rows_loaded == 3
    assert row.content_hash == "abc123"


def test_ingest_terrain_batch_appends_rows_and_closes_batch(tmp_path):
    engine = _engine(tmp_path)
    csv_path = _terrain_csv(tmp_path)

    batch_id = ingest_terrain_batch(engine, csv_path, "telemetry")

    with engine.connect() as conn:
        n_rows = conn.execute(
            text("SELECT COUNT(*) FROM bronze_telemetry WHERE ingestion_batch_id=:id"),
            {"id": str(batch_id)},
        ).scalar()
        status = conn.execute(
            text("SELECT status FROM ingestion_batch WHERE ingestion_batch_id=:id"),
            {"id": str(batch_id)},
        ).scalar()
    assert n_rows == 3
    assert status == "done"


def test_ingest_terrain_batch_does_not_duplicate_same_file(tmp_path):
    engine = _engine(tmp_path)
    csv_path = _terrain_csv(tmp_path)

    first = ingest_terrain_batch(engine, csv_path, "telemetry")
    second = ingest_terrain_batch(engine, csv_path, "telemetry")  # rejeu du meme fichier

    assert first == second  # meme hash -> meme batch reutilise, pas de doublon
    with engine.connect() as conn:
        n_rows = conn.execute(text("SELECT COUNT(*) FROM bronze_telemetry")).scalar()
    assert n_rows == 3  # pas 6


def test_ingest_terrain_batch_appends_new_content_as_new_batch(tmp_path):
    engine = _engine(tmp_path)
    first_csv = _terrain_csv(tmp_path, name="terrain_j1.csv", n_rows=3)
    second_csv = _terrain_csv(tmp_path, name="terrain_j2.csv", n_rows=2)

    first = ingest_terrain_batch(engine, first_csv, "telemetry")
    second = ingest_terrain_batch(engine, second_csv, "telemetry")  # contenu different

    assert first != second
    with engine.connect() as conn:
        n_rows = conn.execute(text("SELECT COUNT(*) FROM bronze_telemetry")).scalar()
        n_batches = conn.execute(text("SELECT COUNT(*) FROM ingestion_batch")).scalar()
    assert n_rows == 5  # 3 + 2, les deux lots coexistent
    assert n_batches == 2


def test_find_batch_by_hash_ignores_unfinished_batches(tmp_path):
    engine = _engine(tmp_path)
    open_batch(engine, "telemetry", "terrain.csv", content_hash="same-hash")  # jamais clos

    assert find_batch_by_hash(engine, "telemetry", "same-hash") is None


def test_ingest_terrain_batch_accepts_maintenance_source(tmp_path):
    engine = _engine(tmp_path)
    csv_path = tmp_path / "maintenance.csv"
    pd.DataFrame(
        {
            "maintenance_id": [1, 2],
            "machine_id": ["MACH-01", "MACH-02"],
            "maintenance_at": ["2026-06-15 08:00:00"] * 2,
            "maintenance_type": ["proactive", "reactive"],
            "action_type": ["inspection", "remplacement"],
            "component": ["pompe", "joint"],
            "description": ["RAS", "fuite"],
            "related_incident_id": [None, "INC-1"],
            "duration_hours": [1.0, 2.5],
        }
    ).to_csv(csv_path, index=False)

    batch_id = ingest_terrain_batch(engine, csv_path, "maintenance")
    ingest_terrain_batch(engine, csv_path, "maintenance")  # rejeu : aucun doublon

    with engine.connect() as conn:
        n_rows = conn.execute(text("SELECT COUNT(*) FROM bronze_maintenance")).scalar()
        source = conn.execute(
            text("SELECT source_name FROM ingestion_batch WHERE ingestion_batch_id=:id"),
            {"id": str(batch_id)},
        ).scalar()
    assert n_rows == 2
    assert source == "maintenance"


def test_ingest_terrain_batch_rejects_unknown_source(tmp_path):
    engine = _engine(tmp_path)
    csv_path = _terrain_csv(tmp_path)
    try:
        ingest_terrain_batch(engine, csv_path, "gold")
        assert False, "devrait lever ValueError pour une source non geree"
    except ValueError as exc:
        assert "gold" in str(exc)
