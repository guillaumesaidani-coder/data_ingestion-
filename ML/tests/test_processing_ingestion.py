"""create_ingestion_dir() extrait à l'identique de TP1.ipynb/TP2.ipynb
(copié-collé dans les deux, cf. pipeline_artifacts/ANALYSE_ARTIFACTS.md).
Comportement figé : même signature, même format de dossier horodaté.
"""

import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from indusense.processing.ingestion import create_ingestion_dir


def test_create_ingestion_dir_default_topic(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    run_dir = create_ingestion_dir(base="artifacts/ingestions")
    assert run_dir.exists()
    assert run_dir.parent.name == "incidents"  # défaut historique de TP1/TP2


def test_create_ingestion_dir_custom_topic_matches_timestamp_format(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    before = datetime.now().strftime("%Y%m%d%H%M")
    run_dir = create_ingestion_dir(base="artifacts/ingestions", topic="telemetry")
    after = datetime.now().strftime("%Y%m%d%H%M")

    assert run_dir.parent.name == "telemetry"
    assert run_dir.name in (before, after)  # tolère un changement de minute pendant le test
    assert len(run_dir.name) == 12 and run_dir.name.isdigit()


def test_create_ingestion_dir_is_idempotent(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    first = create_ingestion_dir(base="artifacts/ingestions", topic="incidents")
    second = create_ingestion_dir(base="artifacts/ingestions", topic="incidents")
    assert first == second  # même minute -> même dossier, pas d'exception (exist_ok=True)
