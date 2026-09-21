"""Fixtures pour les tests de non-régression du socle ETL (Bronze->Silver->Gold)
et du modèle documenté b11-gkf.

Ces tests ne réimplémentent aucune logique métier : ils interrogent la base
Postgres et le tracking MLflow tels que produits par TP1/TP2/TP4/TP5/TP6 (ETL)
et TP9/TP11 (modèle b11), et vérifient des invariants contre ce qui est
documenté (ANALYSE_ARTIFACTS.md, artifacts/model_card.md).
"""

from pathlib import Path

import mlflow
import pytest
from sqlalchemy import create_engine, text
from sqlalchemy.engine import URL

ML_DIR = Path(__file__).resolve().parent.parent

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
