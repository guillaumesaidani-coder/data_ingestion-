"""scripts/streamlit_review.py, prouvé pour de vrai avec
streamlit.testing.v1.AppTest — pas juste un survol visuel : on clique
réellement les boutons et on relit la base après coup, comme le reste
du projet (module 30, "comptage indépendant, pas la valeur que le
script prétend avoir écrite").
"""

import json
import sys
from pathlib import Path

import pytest
from sqlalchemy import create_engine, text

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from indusense.predictions_store import upsert_predictions

streamlit_testing = pytest.importorskip("streamlit.testing.v1")
AppTest = streamlit_testing.AppTest

SCRIPT = str(Path(__file__).resolve().parent.parent / "scripts" / "streamlit_review.py")


def _seed_db(tmp_path, monkeypatch):
    db_path = tmp_path / "predictions.db"
    engine = create_engine(f"sqlite:///{db_path}")
    upsert_predictions(
        engine,
        [
            {
                "machine_id": "MACH-07",
                "window_start": "2026-05-01T00:00:00",
                "failure_proba_24h": 0.62,
                "scored_at": "2026-05-01T01:00:00",
                "model_version": "test-version",
                "features_payload": {"temp_mean_1h": 55.2, "rotation_mean_1h": 1610.0},
            },
            {
                "machine_id": "MACH-03",
                "window_start": "2026-05-02T00:00:00",
                "failure_proba_24h": 0.05,
                "scored_at": "2026-05-02T01:00:00",
                "model_version": "test-version",
                "features_payload": {"temp_mean_1h": 40.1, "rotation_mean_1h": 1500.0},
            },
        ],
    )
    with engine.begin() as conn:
        conn.execute(
            text("UPDATE predictions SET review_status='FAUSSE_ALERTE' WHERE machine_id='MACH-07'")
        )
    monkeypatch.setenv("PREDICTIONS_DB_URL", f"sqlite:///{db_path}")
    return engine


@pytest.mark.parametrize(
    "libelle_choix,statut_attendu",
    [
        ("Capteur défaillant", "CAPTEUR_DEFAILLANT"),
        ("Maintenance préventive effectuée", "MAINTENANCE_PREVENTIVE"),
    ],
)
def test_refine_false_alarm_writes_reclassification(
    tmp_path, monkeypatch, libelle_choix, statut_attendu
):
    engine = _seed_db(tmp_path, monkeypatch)

    at = AppTest.from_file(SCRIPT)
    at.run()
    assert not at.exception

    assert at.radio(key="verdict_fausse_alerte").value == "Fausse alerte confirmée (machine saine)"
    at.radio(key="verdict_fausse_alerte").set_value(libelle_choix)
    at.text_input(key="commentaire_fausse_alerte").set_value("sonde temperature HS")
    at.button(key="valider_fausse_alerte").click().run()
    assert not at.exception

    with engine.connect() as conn:
        row = conn.execute(
            text(
                "SELECT review_status, reviewer_comment FROM predictions WHERE machine_id='MACH-07'"
            )
        ).fetchone()
    assert row.review_status == statut_attendu
    assert row.reviewer_comment == "sonde temperature HS"


def test_declare_unpredicted_incident_writes_review(tmp_path, monkeypatch):
    engine = _seed_db(tmp_path, monkeypatch)

    at = AppTest.from_file(SCRIPT)
    at.run()
    assert not at.exception

    options = at.selectbox(key="select_incident").options
    assert any("MACH-03" in opt for opt in options)
    at.selectbox(key="select_incident").set_value(next(o for o in options if "MACH-03" in o))
    at.text_input(key="commentaire_incident").set_value("arret machine constate en atelier")
    at.button(key="declarer_incident").click().run()
    assert not at.exception

    with engine.connect() as conn:
        row = conn.execute(
            text(
                "SELECT review_status, ground_truth, reviewer_comment "
                "FROM predictions WHERE machine_id='MACH-03'"
            )
        ).fetchone()
    assert row.review_status == "INCIDENT_NON_PREDIT"
    assert bool(row.ground_truth) is True
    assert row.reviewer_comment == "arret machine constate en atelier"


def test_metrics_read_real_features_payload(tmp_path, monkeypatch):
    """Les métriques affichées viennent bien de features_payload (module 35),
    pas de valeurs inventées dans le script Streamlit."""
    engine = _seed_db(tmp_path, monkeypatch)
    with engine.connect() as conn:
        payload = conn.execute(
            text("SELECT features_payload FROM predictions WHERE machine_id='MACH-07'")
        ).scalar()
    assert json.loads(payload)["temp_mean_1h"] == 55.2

    at = AppTest.from_file(SCRIPT)
    at.run()
    metrics_values = {m.label: m.value for m in at.get("metric")}
    assert metrics_values["Température (temp_mean_1h)"] == "55.2 °C"
