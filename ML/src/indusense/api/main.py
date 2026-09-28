"""API de scoring InduSense : /health, /ready, /predict-tabular.

Explicabilité (indusense.explain) : `?explain=true` ajoute à la réponse le
pourquoi du score (décision, facteurs de hausse et de baisse,
avertissements) ; sans ce paramètre la réponse est inchangée. /ready
refuse (503) un modèle non conforme à la base de connaissance : variable
exclue parmi ses entrées, ou certification (scripts/certify_model.py)
bloquante ou faite pour un autre fichier modèle.

Module 25 (service) + module 26 (contrôles de sécurité — voir
security_controls.md pour le registre et threat_model.md pour l'analyse
STRIDE). Quatre contrôles prouvés par test : auth, validation, rate
limit, taille de payload. Le cinquième (audit logging) reste Planifié
v0 — ce fichier logue method/path/status/request_id à titre informatif,
ce n'est pas un événement d'audit structuré et durable.

Aucune nouvelle logique de prédiction : /predict-tabular appelle le
pipeline sklearn déjà entraîné et testé (indusense.modeling) et délègue
à indusense.scoring le choix des colonnes attendues (même implémentation
que le scoring horaire et le backfill). Le modèle est chargé
depuis le fichier produit par `indusense train -o ...` (versionné par
DVC — artifacts/models/model.joblib.dvc).
"""

import json
import logging
import time
import uuid
from collections import defaultdict
from pathlib import Path

import joblib
import pandas as pd
from fastapi import Depends, FastAPI, HTTPException, Request, Response, Security
from fastapi.responses import JSONResponse
from fastapi.security import APIKeyHeader
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Histogram, generate_latest
from pydantic import BaseModel

from indusense.config import get_api_key, get_model_path, get_model_version
from indusense.explain import (
    controler_modele,
    est_conforme,
    expliquer,
    variables_du_modele,
)
from indusense.scoring import model_feature_cols

app = FastAPI(title="InduSense — API de maintenance prédictive")
logger = logging.getLogger("indusense.api")

_api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)

# Chargé une fois au démarrage ; None si le fichier n'existe pas (ex. avant
# le premier `indusense train`, ou dans un environnement qui ne l'a pas).
_model = None
# Certification du modèle chargé (explication_certification.json, à côté
# du modèle) ; None si absente.
_certification = None
_model_version = None

MAX_BODY_BYTES = 64 * 1024
RATE_LIMIT_PER_MINUTE = 60
RATE_LIMIT_WINDOW_SECONDS = 60

# IP -> horodatages des appels dans la fenêtre courante. En mémoire, par
# process : suffisant pour une seule instance, pas pour un déploiement
# multi-instance (cf. security_controls.md, risque résiduel du rate limit).
_rate_limit_state: dict[str, list[float]] = defaultdict(list)

# Instrumentation Prometheus (module 28) : scrapée sur /metrics.
_http_requests_total = Counter(
    "indusense_http_requests_total",
    "Nombre de requêtes HTTP reçues",
    ["method", "path", "status"],
)
_http_request_duration_seconds = Histogram(
    "indusense_http_request_duration_seconds",
    "Durée des requêtes HTTP",
    ["method", "path"],
)


def load_model(path: Path | None = None):
    """(Re)charge le modèle depuis disque. Retourne None si absent —
    ne lève jamais : c'est /ready qui traduit cette absence en 503."""
    global _model, _certification, _model_version
    model_path = Path(path or get_model_path())
    _model = joblib.load(model_path) if model_path.exists() else None
    _model_version = get_model_version(model_path) if _model is not None else None
    cert_path = model_path.parent / "explication_certification.json"
    _certification = (
        json.loads(cert_path.read_text(encoding="utf-8")) if cert_path.exists() else None
    )
    return _model


def non_conformites() -> list[str]:
    """Raisons de refuser de servir le modèle chargé. Sans données :
    exclusions (liste des entrées du modèle) ; concentration et autres
    mesures d'entraînement : lues dans la certification, si présente."""
    constats = controler_modele(variables_du_modele(_model))
    raisons = [f"{c['variable']} : {c['message']}" for c in constats if not est_conforme([c])]
    if _certification is not None:
        if _certification.get("model_version") != _model_version:
            raisons.append(
                f"certification faite pour le modèle {_certification.get('model_version')}, "
                f"modèle chargé {_model_version} : relancer scripts/certify_model.py"
            )
        elif not _certification.get("conforme", False):
            raisons += [
                f"{c.get('variable') or c['controle']} : {c['message']}"
                for c in _certification["constats"]
                if c["gravite"] == "bloquant"
            ]
    return raisons


load_model()


@app.middleware("http")
async def limit_body_size(request: Request, call_next):
    content_length = request.headers.get("content-length")
    if content_length is not None:
        try:
            length = int(content_length)
        except ValueError:
            return JSONResponse(status_code=400, content={"detail": "Content-Length illisible"})
        if length > MAX_BODY_BYTES:
            return JSONResponse(
                status_code=413,
                content={"detail": f"Payload supérieur à {MAX_BODY_BYTES} octets"},
            )
    return await call_next(request)


@app.middleware("http")
async def add_request_id(request: Request, call_next):
    request_id = request.headers.get("X-Request-ID", str(uuid.uuid4()))
    start = time.monotonic()
    response = await call_next(request)
    duration = time.monotonic() - start
    response.headers["X-Request-ID"] = request_id
    # Volontairement limité à method/path/status/request_id : jamais les en-têtes
    # (donc jamais X-API-Key) ni le corps de la requête. Ne remplace pas un
    # audit logging structuré (Planifié v0) — juste une trace non sensible.
    logger.info(
        "request_id=%s method=%s path=%s status=%s",
        request_id,
        request.method,
        request.url.path,
        response.status_code,
    )
    _http_requests_total.labels(
        method=request.method, path=request.url.path, status=response.status_code
    ).inc()
    _http_request_duration_seconds.labels(method=request.method, path=request.url.path).observe(
        duration
    )
    return response


def require_api_key(api_key: str = Security(_api_key_header)) -> str:
    if api_key != get_api_key():
        raise HTTPException(status_code=401, detail="Clé API manquante ou invalide")
    return api_key


def rate_limit(client_id: str) -> None:
    now = time.monotonic()
    window_start = now - RATE_LIMIT_WINDOW_SECONDS
    timestamps = _rate_limit_state[client_id]
    while timestamps and timestamps[0] < window_start:
        timestamps.pop(0)
    if len(timestamps) >= RATE_LIMIT_PER_MINUTE:
        raise HTTPException(status_code=429, detail="Trop de requêtes — réessayez plus tard")
    timestamps.append(now)


def rate_limit_dependency(request: Request) -> None:
    client_id = request.client.host if request.client else "unknown"
    rate_limit(client_id)


class PredictRequest(BaseModel):
    features: dict[str, float | None]


class PredictResponse(BaseModel):
    failure_proba_24h: float
    explication: dict | None = None


@app.get("/metrics")
def metrics():
    """Scrapé par Prometheus (voir prometheus.yml, job indusense-api)."""
    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.get("/health")
def health():
    """Le processus tourne — ne dit rien du modèle. Pas d'authentification :
    une sonde de liveness ne doit jamais dépendre d'une clé API."""
    return {"status": "ok"}


@app.get("/ready")
def ready():
    """Le service peut vraiment servir des prédictions : le modèle est
    chargé ET conforme à la base de connaissance."""
    if _model is None:
        raise HTTPException(status_code=503, detail="Modèle non chargé")
    raisons = non_conformites()
    if raisons:
        raise HTTPException(
            status_code=503,
            detail={"message": "Modèle non conforme à la base de connaissance", "raisons": raisons},
        )
    return {"status": "ready"}


@app.post(
    "/predict-tabular",
    response_model=PredictResponse,
    response_model_exclude_none=True,
    dependencies=[Depends(require_api_key), Depends(rate_limit_dependency)],
)
def predict_tabular(payload: PredictRequest, explain: bool = False):
    if _model is None:
        raise HTTPException(status_code=503, detail="Modèle non chargé")

    if not payload.features:
        raise HTTPException(status_code=422, detail="features ne peut pas être vide")

    expected_cols = model_feature_cols(_model, list(payload.features))

    known = set(expected_cols) & set(payload.features)
    if not known:
        raise HTTPException(
            status_code=422,
            detail="Aucune des features envoyées ne correspond aux features du modèle",
        )

    row = pd.DataFrame([payload.features]).reindex(columns=expected_cols)
    proba = float(_model.predict_proba(row)[:, 1][0])
    if not explain:
        return PredictResponse(failure_proba_24h=proba)

    explication = expliquer(_model, row)
    inconnues = sorted(set(payload.features) - set(expected_cols))
    if inconnues:
        explication["avertissements"].append(
            f"Variables inconnues du modèle, ignorées : {', '.join(inconnues)}"
        )
    return PredictResponse(failure_proba_24h=proba, explication=explication)
