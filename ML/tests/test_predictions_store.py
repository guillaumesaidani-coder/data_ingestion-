"""upsert_predictions() : la preuve d'idempotence des modules 29/30, isolée
du flow Prefect — moteur SQLAlchemy neuf (SQLite temporaire) à chaque
test, jamais réutilisé. Le même code (predictions_store.py) tourne aussi
contre Postgres (module 30) — voir tests/test_flow_pipeline.py.
"""

import sys
from pathlib import Path

from sqlalchemy import create_engine

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from indusense.predictions_store import count_predictions, upsert_predictions


def _engine(tmp_path):
    return create_engine(f"sqlite:///{tmp_path / 'predictions.db'}")


def _row(machine_id, window_start, proba, scored_at="2026-01-01T00:00:00"):
    return {
        "machine_id": machine_id,
        "window_start": window_start,
        "failure_proba_24h": proba,
        "scored_at": scored_at,
    }


def test_upsert_inserts_new_rows(tmp_path):
    engine = _engine(tmp_path)
    rows = [
        _row("MACH-01", "2026-01-01T00:00:00", 0.1),
        _row("MACH-02", "2026-01-01T00:00:00", 0.2),
    ]
    n = upsert_predictions(engine, rows)
    assert n == 2
    assert count_predictions(engine) == 2


def test_upsert_is_idempotent_on_replay(tmp_path):
    engine = _engine(tmp_path)
    rows = [
        _row("MACH-01", "2026-01-01T00:00:00", 0.1),
        _row("MACH-02", "2026-01-01T00:00:00", 0.2),
    ]

    n1 = upsert_predictions(engine, rows)
    n2 = upsert_predictions(engine, rows)  # même clé, même contenu

    assert n1 == n2 == 2  # pas 4 : la clé composite empêche le doublon


def test_upsert_updates_existing_key_instead_of_duplicating(tmp_path):
    engine = _engine(tmp_path)
    upsert_predictions(engine, [_row("MACH-01", "2026-01-01T00:00:00", 0.1, "2026-01-01T00:00:01")])
    n = upsert_predictions(
        engine, [_row("MACH-01", "2026-01-01T00:00:00", 0.9, "2026-01-01T00:05:00")]
    )

    assert n == 1  # toujours une seule ligne pour cette clé

    from sqlalchemy import text

    with engine.connect() as conn:
        proba, scored_at = conn.execute(
            text("SELECT failure_proba_24h, scored_at FROM predictions WHERE machine_id='MACH-01'")
        ).fetchone()
    assert proba == 0.9  # la valeur la plus récente a bien remplacé l'ancienne
    assert scored_at == "2026-01-01T00:05:00"


def test_different_window_start_is_a_different_row(tmp_path):
    engine = _engine(tmp_path)
    upsert_predictions(
        engine,
        [
            _row("MACH-01", "2026-01-01T00:00:00", 0.1),
            _row("MACH-01", "2026-01-01T01:00:00", 0.2),
        ],
    )
    assert count_predictions(engine) == 2  # même machine, deux heures -> deux lignes
