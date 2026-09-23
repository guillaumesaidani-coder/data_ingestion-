"""get_db_url() centralise ce qui était dupliqué (identique) dans chaque
notebook/script : les valeurs par défaut DOIVENT reproduire exactement
les credentials en dur historiques, sinon toute connexion existante casse.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from indusense.config import get_db_url


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
