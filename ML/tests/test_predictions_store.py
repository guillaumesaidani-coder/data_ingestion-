"""upsert_predictions() : la preuve d'idempotence des modules 29/30, isolée
du flow Prefect — moteur SQLAlchemy neuf (SQLite temporaire) à chaque
test, jamais réutilisé. Le même code (predictions_store.py) tourne aussi
contre Postgres (module 30) — voir tests/test_flow_pipeline.py.

Module 35 : review_prediction() ne doit jamais être effacée par un
replay de upsert_predictions() — testé explicitement ci-dessous.
"""

import sys
from pathlib import Path

import pytest
from sqlalchemy import create_engine

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from indusense.predictions_store import (
    count_predictions,
    review_prediction,
    upsert_predictions,
)


def _engine(tmp_path):
    return create_engine(f"sqlite:///{tmp_path / 'predictions.db'}")


def _row(
    machine_id,
    window_start,
    proba,
    scored_at="2026-01-01T00:00:00",
    model_version="abc123",
    features_payload=None,
):
    return {
        "machine_id": machine_id,
        "window_start": window_start,
        "failure_proba_24h": proba,
        "scored_at": scored_at,
        "model_version": model_version,
        "features_payload": features_payload or {"temp_mean_24h": 48.1},
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


def test_replay_does_not_erase_an_existing_review(tmp_path):
    engine = _engine(tmp_path)
    upsert_predictions(engine, [_row("MACH-01", "2026-01-01T00:00:00", 0.9, model_version="v1")])
    review_prediction(
        engine,
        "MACH-01",
        "2026-01-01T00:00:00",
        review_status="PANNE_CONFIRMEE",
        ground_truth=True,
        reviewer_comment="BI-2026-042",
        reviewed_at="2026-01-02T08:00:00",
    )

    # Le flow rejoue la meme fenetre avec un modele plus recent
    upsert_predictions(engine, [_row("MACH-01", "2026-01-01T00:00:00", 0.95, model_version="v2")])

    from sqlalchemy import text

    with engine.connect() as conn:
        row = conn.execute(
            text(
                "SELECT model_version, review_status, ground_truth, reviewer_comment "
                "FROM predictions WHERE machine_id='MACH-01'"
            )
        ).fetchone()
    assert row.model_version == "v2"  # la prediction, elle, est bien mise a jour
    assert row.review_status == "PANNE_CONFIRMEE"  # la revue survit au replay
    assert bool(row.ground_truth) is True
    assert row.reviewer_comment == "BI-2026-042"


def test_review_prediction_fails_on_unknown_window(tmp_path):
    engine = _engine(tmp_path)
    upsert_predictions(engine, [_row("MACH-01", "2026-01-01T00:00:00", 0.1)])
    with pytest.raises(ValueError):
        review_prediction(
            engine,
            "MACH-01",
            "2099-01-01T00:00:00",
            review_status="PANNE_CONFIRMEE",
            ground_truth=True,
            reviewer_comment=None,
            reviewed_at="2026-01-02T08:00:00",
        )
