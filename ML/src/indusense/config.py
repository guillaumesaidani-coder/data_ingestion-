"""Configuration transverse : connexion PostgreSQL.

Centralise ce qui était dupliqué (identique) dans chaque notebook et
script : TP1/TP2/TP4/TP5/TP6/TP9/TP11/generate_model_card.py
construisaient chacun leur propre `URL.create(...)` avec les mêmes
valeurs en dur. Les valeurs par défaut ci-dessous reproduisent
exactement ces valeurs — le comportement ne change pas tant qu'aucune
variable d'environnement n'est définie ; un `.env` (non commité) permet
de les surcharger sans toucher au code.
"""

import os
from pathlib import Path

from dotenv import load_dotenv
from sqlalchemy import create_engine
from sqlalchemy.engine import URL, Engine

load_dotenv()

ML_DIR = Path(__file__).resolve().parent.parent.parent


def get_db_url() -> URL:
    return URL.create(
        drivername="postgresql+psycopg2",
        username=os.getenv("DB_USER", "indusense_user"),
        password=os.getenv("DB_PASSWORD", "ThEP@ssW0rd"),
        host=os.getenv("DB_HOST", "localhost"),
        port=int(os.getenv("DB_PORT", "5432")),
        database=os.getenv("DB_NAME", "indusense_db"),
    )


def get_engine() -> Engine:
    return create_engine(get_db_url())


def get_mlflow_tracking_uri() -> str:
    """Reproduit l'URI utilisée par conftest.py et generate_model_card.py :
    sqlite:///ML/mlflow_tp7.db (chemin absolu, résolu quel que soit le cwd)."""
    return os.getenv("MLFLOW_TRACKING_URI", f"sqlite:///{ML_DIR / 'mlflow_tp7.db'}")


def get_api_key() -> str:
    """Clé attendue dans l'en-tête X-API-Key de l'API. Valeur de dev par
    défaut (même logique que les credentials DB ci-dessus) — à surcharger
    par variable d'environnement pour tout usage au-delà du poste local."""
    return os.getenv("API_KEY", "dev-local-key")


def get_model_path() -> Path:
    """Chemin du modèle entraîné servi par l'API (produit par `indusense train -o ...`,
    versionné par DVC — voir artifacts/models/model.joblib.dvc)."""
    return Path(os.getenv("MODEL_PATH", str(ML_DIR / "artifacts" / "models" / "model.joblib")))


def get_predictions_db_url() -> str:
    """URL SQLAlchemy du stockage des prédictions (module 30) : une seule
    URL, pas de branche if backend == "postgres". SQLite en local par
    défaut ; `compose.yaml` la surcharge en Postgres pour le service api
    (même base indusense_db que le Gold dataset)."""
    default = f"sqlite:///{ML_DIR / 'artifacts' / 'predictions.db'}"
    return os.getenv("PREDICTIONS_DB_URL", default)


def get_predictions_engine() -> Engine:
    return create_engine(get_predictions_db_url())
