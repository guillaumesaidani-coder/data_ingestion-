"""tests/test_api.py — /health, /ready, /predict-tabular, et le middleware
X-Request-ID. N'utilise jamais le vrai modèle (artifacts/models/model.joblib,
piloté par DVC, absent d'un checkout CI brut) : un petit pipeline synthétique
est injecté à la place, ce qui rend ces tests indépendants de toute infra
locale — aucun marker requires_local_infra nécessaire ici.
"""

import sys
import uuid
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import indusense.api.main as api_main
from indusense.config import get_api_key

FEATURE_NAMES = ["temp_mean_24h", "pressure_mean_24h"]


@pytest.fixture()
def client():
    return TestClient(api_main.app)


@pytest.fixture()
def fake_model(monkeypatch):
    rng = np.random.default_rng(42)
    X = pd.DataFrame({name: rng.normal(size=40) for name in FEATURE_NAMES})
    y = (X[FEATURE_NAMES[0]] > 0).astype(int)
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
    monkeypatch.setattr(api_main, "_model", None)


def auth_headers():
    return {"X-API-Key": get_api_key()}


# --------------------------------------------------------------------- health
def test_health_returns_ok_without_auth(client):
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok"}


# --------------------------------------------------------------------- ready
def test_ready_returns_503_when_model_not_loaded(client, no_model):
    resp = client.get("/ready")
    assert resp.status_code == 503


def test_ready_returns_ok_when_model_loaded(client, fake_model):
    resp = client.get("/ready")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ready"}


# ------------------------------------------------------------ predict-tabular
def test_predict_tabular_requires_api_key(client, fake_model):
    resp = client.post("/predict-tabular", json={"features": {"temp_mean_24h": 1.0}})
    assert resp.status_code == 401


def test_predict_tabular_rejects_wrong_api_key(client, fake_model):
    resp = client.post(
        "/predict-tabular",
        json={"features": {"temp_mean_24h": 1.0}},
        headers={"X-API-Key": "wrong-key"},
    )
    assert resp.status_code == 401


def test_predict_tabular_rejects_empty_features(client, fake_model):
    resp = client.post("/predict-tabular", json={"features": {}}, headers=auth_headers())
    assert resp.status_code == 422


def test_predict_tabular_rejects_malformed_payload(client, fake_model):
    resp = client.post("/predict-tabular", json={"not_features": 1}, headers=auth_headers())
    assert resp.status_code == 422


def test_predict_tabular_returns_503_when_model_not_loaded(client, no_model):
    resp = client.post(
        "/predict-tabular", json={"features": {"temp_mean_24h": 1.0}}, headers=auth_headers()
    )
    assert resp.status_code == 503


def test_predict_tabular_returns_prediction_with_valid_payload(client, fake_model):
    resp = client.post(
        "/predict-tabular",
        json={"features": {"temp_mean_24h": 1.5, "pressure_mean_24h": -0.5}},
        headers=auth_headers(),
    )
    assert resp.status_code == 200
    body = resp.json()
    assert 0.0 <= body["failure_proba_24h"] <= 1.0


def test_predict_tabular_works_with_partial_features(client, fake_model):
    # Une seule des deux features connues du modèle -> l'imputer comble le reste.
    resp = client.post(
        "/predict-tabular",
        json={"features": {"temp_mean_24h": 1.5}},
        headers=auth_headers(),
    )
    assert resp.status_code == 200


# -------------------------------------------------------------- X-Request-ID
def test_response_has_generated_request_id(client):
    resp = client.get("/health")
    request_id = resp.headers["X-Request-ID"]
    assert uuid.UUID(request_id)  # lève si ce n'est pas un UUID valide


def test_response_echoes_client_request_id(client):
    resp = client.get("/health", headers={"X-Request-ID": "demo-trace-42"})
    assert resp.headers["X-Request-ID"] == "demo-trace-42"
