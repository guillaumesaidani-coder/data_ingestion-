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
from indusense.processing.operators import badge_hash
from indusense.processing.silver_events import build_silver_incidents

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
    parse_ok BOOLEAN NOT NULL DEFAULT 1,
    parse_ok_reason TEXT NOT NULL DEFAULT '',
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
    duration_hours REAL,
    parse_ok BOOLEAN NOT NULL DEFAULT 1,
    parse_ok_reason TEXT NOT NULL DEFAULT ''
);
CREATE TABLE data_quality_issue (
    dq_issue_id INTEGER PRIMARY KEY AUTOINCREMENT,
    ingestion_batch_id TEXT,
    dataset_name TEXT,
    rule_code TEXT,
    severity TEXT,
    entity_key TEXT,
    details TEXT
);
CREATE TABLE bronze_incidents (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    incident_id TEXT,
    date TEXT,
    time TEXT,
    operator_name TEXT,
    machine_id TEXT,
    severity INTEGER,
    operator_badge TEXT,
    comment TEXT,
    shift TEXT,
    type_surchauffe INTEGER,
    type_baisse_pression INTEGER,
    type_vibration INTEGER,
    type_bruit_mecanique INTEGER,
    type_surconsommation INTEGER,
    type_blocage_mecanique INTEGER,
    type_alarme_capteur INTEGER,
    type_arret_urgence INTEGER,
    type_defaut_qualite INTEGER,
    parse_ok BOOLEAN NOT NULL DEFAULT 1,
    parse_ok_reason TEXT NOT NULL DEFAULT '',
    ingestion_batch_id TEXT
);
CREATE TABLE operator (
    operator_id INTEGER PRIMARY KEY AUTOINCREMENT,
    operator_key TEXT UNIQUE,
    badge_hash TEXT,
    is_active BOOLEAN NOT NULL DEFAULT 1
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


def test_ingest_terrain_batch_writes_rejected_rows_and_traces_them(tmp_path):
    """Une ligne invalide et un doublon horaire : écrits quand même en
    Bronze (miroir du fichier), mais parse_ok=False, comptés dans
    rows_rejected et tracés dans data_quality_issue."""
    engine = _engine(tmp_path)
    csv_path = tmp_path / "terrain_mixte.csv"
    csv_path.write_text(
        "machine_id,timestamp,temperature_c,pressure_bar,voltage_mean_v,rotation_mean_rpm,"
        "pieces_produced\n"
        "MACH-01,2026-06-15 00:00:00,48.0,195.0,228.0,1600.0,42\n"
        "MACH-02,2026-06-15 00:00:00,abc,195.0,228.0,1600.0,42\n"
        "MACH-03,2026-06-15 00:00:00,48.0,195.0,228.0,1600.0,42\n"
        "MACH-03,2026-06-15 00:00:00,48.1,195.0,228.0,1600.0,42\n",
        encoding="utf-8",
    )

    batch_id = ingest_terrain_batch(engine, csv_path, "telemetry")

    with engine.connect() as conn:
        flags = dict(
            conn.execute(
                text("SELECT machine_id, SUM(parse_ok) FROM bronze_telemetry GROUP BY machine_id")
            ).fetchall()
        )
        batch = conn.execute(
            text(
                "SELECT rows_read, rows_loaded, rows_rejected FROM ingestion_batch "
                "WHERE ingestion_batch_id=:id"
            ),
            {"id": str(batch_id)},
        ).fetchone()
        issues = conn.execute(
            text("SELECT rule_code, severity FROM data_quality_issue ORDER BY rule_code")
        ).fetchall()

    assert flags == {"MACH-01": 1, "MACH-02": 0, "MACH-03": 0}
    assert tuple(batch) == (4, 1, 3)
    assert [tuple(i) for i in issues] == [
        ("DOUBLON_HORAIRE", "WARNING"),
        ("DOUBLON_HORAIRE", "WARNING"),
        ("PYDANTIC_ERROR", "ERROR"),
    ]


def test_ingest_terrain_batch_failure_leaves_no_bronze_rows_and_replay_loads_once(
    tmp_path, monkeypatch
):
    """Échec avant la clôture : aucune ligne Bronze orpheline (le batch
    reste tracé en 'running'), et le rejeu du même fichier charge le lot
    une seule fois -- pas de doublon qui passerait ensuite au Silver."""
    from indusense import ingest

    engine = _engine(tmp_path)
    csv_path = _terrain_csv(tmp_path)

    def boom(*args, **kwargs):
        raise RuntimeError("coupure pendant la clôture")

    monkeypatch.setattr(ingest, "_close_batch", boom)
    try:
        ingest_terrain_batch(engine, csv_path, "telemetry")
        assert False, "l'échec de clôture devait remonter"
    except RuntimeError:
        pass

    with engine.connect() as conn:
        assert conn.execute(text("SELECT COUNT(*) FROM bronze_telemetry")).scalar() == 0
        assert conn.execute(text("SELECT status FROM ingestion_batch")).scalar() == "running"

    monkeypatch.undo()
    ingest_terrain_batch(engine, csv_path, "telemetry")
    with engine.connect() as conn:
        assert conn.execute(text("SELECT COUNT(*) FROM bronze_telemetry")).scalar() == 3


def test_ingest_terrain_batch_rejects_unknown_source(tmp_path):
    engine = _engine(tmp_path)
    csv_path = _terrain_csv(tmp_path)
    try:
        ingest_terrain_batch(engine, csv_path, "gold")
        assert False, "devrait lever ValueError pour une source non geree"
    except ValueError as exc:
        assert "gold" in str(exc)


INCIDENT_HEADER = (
    "incident_id,date,time,operator_name,machine_id,severity,operator_badge,comment,shift,"
    "type_surchauffe,type_baisse_pression,type_vibration,type_bruit_mecanique,"
    "type_surconsommation,type_blocage_mecanique,type_alarme_capteur,type_arret_urgence,"
    "type_defaut_qualite\n"
)


def _incidents_csv(tmp_path, name, *rows):
    """rows : (incident_id, operator_name, operator_badge, severity)."""
    path = tmp_path / name
    lines = [
        f"{inc},2026-06-15,10:00,{op},MACH-01,{sev},{badge},,matin,0,0,0,0,0,0,0,0,0\n"
        for inc, op, badge, sev in rows
    ]
    path.write_text(INCIDENT_HEADER + "".join(lines), encoding="utf-8")
    return path


def _operators(engine) -> dict[str, str]:
    with engine.connect() as conn:
        return dict(conn.execute(text("SELECT operator_key, badge_hash FROM operator")).fetchall())


def test_ingest_incidents_creates_unknown_operator_and_silver_resolves_it(tmp_path):
    """Opérateur absent du référentiel : créé à l'ingestion, si bien que
    le Silver (même jointure que etl_flow) lui trouve un operator_id."""
    engine = _engine(tmp_path)
    with engine.begin() as conn:
        conn.execute(text("INSERT INTO operator (operator_key, badge_hash) VALUES ('alice', 'h')"))
    csv_path = _incidents_csv(
        tmp_path, "inc.csv", ("INC-1", "Alice", "B1", 3), ("INC-2", " Zoé Nouvelle ", "B9", 2)
    )

    ingest_terrain_batch(engine, csv_path, "incidents")

    ops = _operators(engine)
    assert set(ops) == {"alice", "zoé nouvelle"}
    assert ops["zoé nouvelle"] == badge_hash("Zoé Nouvelle", "B9")
    bronze = pd.read_sql(text("SELECT * FROM bronze_incidents WHERE parse_ok = 1"), engine)
    operators = pd.read_sql(text("SELECT operator_id, operator_key FROM operator"), engine)
    assert build_silver_incidents(bronze, operators)["operator_id"].notna().all()


def test_ingest_incidents_replay_and_new_batch_do_not_duplicate_operators(tmp_path):
    engine = _engine(tmp_path)
    first = _incidents_csv(tmp_path, "j1.csv", ("INC-1", "Alice", "B1", 3))
    second = _incidents_csv(tmp_path, "j2.csv", ("INC-2", "ALICE", "B1", 4))

    ingest_terrain_batch(engine, first, "incidents")
    ingest_terrain_batch(engine, first, "incidents")  # rejeu : batch réutilisé
    ingest_terrain_batch(engine, second, "incidents")  # nouveau lot, même opérateur

    with engine.connect() as conn:
        assert conn.execute(text("SELECT COUNT(*) FROM operator")).scalar() == 1


def test_ingest_incidents_skips_rejected_rows_and_empty_names(tmp_path):
    """Ligne rejetée (sévérité 9) ou nom vide : aucun opérateur créé."""
    engine = _engine(tmp_path)
    csv_path = _incidents_csv(
        tmp_path, "inc.csv", ("INC-1", "Alcie", "B1", 9), ("INC-2", "", "B2", 2)
    )

    ingest_terrain_batch(engine, csv_path, "incidents")

    assert _operators(engine) == {}


def test_ingest_incidents_reassigned_badge_updates_hash_missing_badge_keeps_it(tmp_path):
    engine = _engine(tmp_path)
    ingest_terrain_batch(
        engine, _incidents_csv(tmp_path, "j1.csv", ("INC-1", "Alice", "B1", 3)), "incidents"
    )
    ingest_terrain_batch(
        engine, _incidents_csv(tmp_path, "j2.csv", ("INC-2", "Alice", "B2", 3)), "incidents"
    )
    assert _operators(engine) == {"alice": badge_hash("Alice", "B2")}

    ingest_terrain_batch(
        engine, _incidents_csv(tmp_path, "j3.csv", ("INC-3", "Alice", "", 3)), "incidents"
    )
    assert _operators(engine) == {"alice": badge_hash("Alice", "B2")}
