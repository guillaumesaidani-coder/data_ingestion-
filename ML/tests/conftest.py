"""Fixtures pour les tests de non-régression du socle ETL (Bronze->Silver->Gold)
et du modèle documenté b11-gkf.

Ces tests ne réimplémentent aucune logique métier : ils interrogent la base
Postgres et le tracking MLflow tels que produits par TP1/TP2/TP4/TP5/TP6 (ETL)
et TP9/TP11 (modèle b11), et vérifient des invariants contre ce qui est
documenté (ANALYSE_ARTIFACTS.md, artifacts/model_card.md).
"""

import sys
from pathlib import Path

import mlflow
import numpy as np
import pandas as pd
import pytest
from sqlalchemy import create_engine, text
from sqlalchemy.engine import URL

ML_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ML_DIR / "src"))

DB_URL = URL.create(
    drivername="postgresql+psycopg2",
    username="indusense_user",
    password="ThEP@ssW0rd",
    host="localhost",
    port=5432,
    database="indusense_db",
)

B11_RUN_ID = "fa336bc0880b48bc849a4a6b1e6f412d"
B11_MODEL_ARTIFACT_DIR = (
    ML_DIR.parent / "mlruns" / "1" / "models" / "m-649e4e35dac14192b37d34ba1ae9ff81" / "artifacts"
)


@pytest.fixture(scope="session")
def db_engine():
    engine = create_engine(DB_URL)
    try:
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
    except Exception as exc:  # noqa: BLE001
        pytest.fail(
            f"Base indusense_db injoignable ({exc}). "
            "Le container docker-db-1 (postgres:15.1, port 5432) doit tourner : "
            "le socle ETL ne peut pas être prouvé sans lui.",
            pytrace=False,
        )
    yield engine
    engine.dispose()


@pytest.fixture(scope="session")
def mlflow_client():
    mlflow.set_tracking_uri(f"sqlite:///{ML_DIR / 'mlflow_tp7.db'}")
    return mlflow.tracking.MlflowClient()


# --------------------------------------------------------------------------
# API (tests/test_api.py, tests/test_security.py) : jamais le vrai modèle
# (artifacts/models/model.joblib, piloté par DVC, absent d'un checkout CI
# brut) — un petit pipeline synthétique est injecté à la place.
# --------------------------------------------------------------------------

API_FEATURE_NAMES = ["temp_mean_24h", "pressure_mean_24h"]


@pytest.fixture()
def client():
    from fastapi.testclient import TestClient

    import indusense.api.main as api_main

    # Le rate limiter est un dict en mémoire, par IP : sans ce nettoyage,
    # son état fuiterait d'un test à l'autre (résultat dépendant de l'ordre).
    api_main._rate_limit_state.clear()
    yield TestClient(api_main.app)
    api_main._rate_limit_state.clear()


@pytest.fixture()
def fake_model(monkeypatch):
    from sklearn.impute import SimpleImputer
    from sklearn.pipeline import Pipeline
    from xgboost import XGBClassifier

    import indusense.api.main as api_main

    rng = np.random.default_rng(42)
    X = pd.DataFrame({name: rng.normal(size=40) for name in API_FEATURE_NAMES})
    y = (X[API_FEATURE_NAMES[0]] > 0).astype(int)
    pipe = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("model", XGBClassifier(n_estimators=10, max_depth=2, random_state=42)),
        ]
    )
    pipe.fit(X, y)
    monkeypatch.setattr(api_main, "_model", pipe)
    return pipe


@pytest.fixture()
def no_model(monkeypatch):
    import indusense.api.main as api_main

    monkeypatch.setattr(api_main, "_model", None)


def auth_headers():
    from indusense.config import get_api_key

    return {"X-API-Key": get_api_key()}
