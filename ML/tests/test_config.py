"""get_db_url() centralise ce qui était dupliqué (identique) dans chaque
notebook/script : les valeurs par défaut DOIVENT reproduire exactement
les credentials en dur historiques, sinon toute connexion existante casse.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from indusense.config import get_db_url, get_predictions_db_url


def test_default_db_url_matches_historical_hardcoded_values():
    url = get_db_url()
    assert url.drivername == "postgresql+psycopg2"
    assert url.username == "indusense_user"
    assert url.password == "ThEP@ssW0rd"
    assert url.host == "localhost"
    assert url.port == 5432
    assert url.database == "indusense_db"


def test_env_var_overrides_default(monkeypatch):
    monkeypatch.setenv("DB_HOST", "some-other-host")
    monkeypatch.setenv("DB_PORT", "6543")
    url = get_db_url()
    assert url.host == "some-other-host"
    assert url.port == 6543


def test_predictions_default_to_the_central_database(monkeypatch):
    # Plus de repli silencieux sur artifacts/predictions.db (SQLite archivé) :
    # sans surcharge, prédictions et verdicts vont dans la même base que le Gold.
    monkeypatch.delenv("PREDICTIONS_DB_URL", raising=False)
    monkeypatch.delenv("DB_HOST", raising=False)
    monkeypatch.delenv("DB_PASSWORD", raising=False)
    assert get_predictions_db_url() == (
        "postgresql+psycopg2://indusense_user:ThEP%40ssW0rd@localhost:5432/indusense_db"
    )
