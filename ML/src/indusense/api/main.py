"""API de scoring InduSense (module 25) : /health, /ready, /predict-tabular,
et le middleware request-id qui trace chaque requête.

Aucune nouvelle logique de prédiction : /predict-tabular appelle le
pipeline sklearn déjà entraîné et testé (indusense.modeling), chargé
depuis le fichier produit par `indusense train -o ...` (versionné par
DVC — artifacts/models/model.joblib.dvc).
"""

import uuid
from pathlib import Path

import joblib
import pandas as pd
from fastapi import Depends, FastAPI, HTTPException, Request, Security
from fastapi.security import APIKeyHeader
from pydantic import BaseModel

from indusense.config import get_api_key, get_model_path

app = FastAPI(title="InduSense — API de maintenance prédictive")

_api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)

# Chargé une fois au démarrage ; None si le fichier n'existe pas (ex. avant
# le premier `indusense train`, ou dans un environnement qui ne l'a pas).
_model = None


def load_model(path: Path | None = None):
    """(Re)charge le modèle depuis disque. Retourne None si absent —
    ne lève jamais : c'est /ready qui traduit cette absence en 503."""
    global _model
    model_path = path or get_model_path()
    _model = joblib.load(model_path) if Path(model_path).exists() else None
    return _model


load_model()


@app.middleware("http")
async def add_request_id(request: Request, call_next):
    request_id = request.headers.get("X-Request-ID", str(uuid.uuid4()))
    response = await call_next(request)
    response.headers["X-Request-ID"] = request_id
    return response


def require_api_key(api_key: str = Security(_api_key_header)) -> str:
    if api_key != get_api_key():
        raise HTTPException(status_code=401, detail="Clé API manquante ou invalide")
    return api_key


class PredictRequest(BaseModel):
    features: dict[str, float | None]


class PredictResponse(BaseModel):
    failure_proba_24h: float


@app.get("/health")
def health():
    """Le processus tourne — ne dit rien du modèle. Pas d'authentification :
    une sonde de liveness ne doit jamais dépendre d'une clé API."""
    return {"status": "ok"}


@app.get("/ready")
def ready():
    """Le service peut vraiment servir des prédictions : le modèle est chargé."""
    if _model is None:
        raise HTTPException(status_code=503, detail="Modèle non chargé")
    return {"status": "ready"}


@app.post("/predict-tabular", response_model=PredictResponse)
def predict_tabular(payload: PredictRequest, api_key: str = Depends(require_api_key)):
    if _model is None:
        raise HTTPException(status_code=503, detail="Modèle non chargé")

    if not payload.features:
        raise HTTPException(status_code=422, detail="features ne peut pas être vide")

    imputer = _model.named_steps.get("imputer")
    expected_cols = (
        list(imputer.feature_names_in_)
        if imputer is not None and hasattr(imputer, "feature_names_in_")
        else list(payload.features)
    )

    known = set(expected_cols) & set(payload.features)
    if not known:
        raise HTTPException(
            status_code=422,
            detail="Aucune des features envoyées ne correspond aux features du modèle",
        )

    row = pd.DataFrame([payload.features]).reindex(columns=expected_cols)
    proba = float(_model.predict_proba(row)[:, 1][0])
    return PredictResponse(failure_proba_24h=proba)
