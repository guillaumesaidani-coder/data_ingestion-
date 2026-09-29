"""tests/test_api.py — /health, /ready, /predict-tabular, et le middleware
X-Request-ID. Fixtures (client, fake_model, no_model, auth_headers) dans
conftest.py, partagées avec tests/test_security.py.
"""

import uuid

import pandas as pd
from conftest import auth_headers


# --------------------------------------------------------------------- health
def test_health_returns_ok_without_auth(client):
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json()["status"] == "ok"


def test_health_reports_code_revision(client, monkeypatch):
    # Une image en retard sur le dépôt se voit dans /health.
    monkeypatch.setenv("INDUSENSE_REVISION", "abc1234")
    assert client.get("/health").json()["revision"] == "abc1234"
    monkeypatch.delenv("INDUSENSE_REVISION")
    assert client.get("/health").json()["revision"] == "dépôt local"


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


# ------------------------------------------------------------ explicabilité
def test_predict_tabular_without_explain_is_unchanged(client, fake_model):
    resp = client.post(
        "/predict-tabular",
        json={"features": {"temp_mean_24h": 1.5, "pressure_mean_24h": -0.5}},
        headers=auth_headers(),
    )
    assert set(resp.json()) == {"failure_proba_24h"}


def test_predict_tabular_explain_returns_why(client, fake_model):
    resp = client.post(
        "/predict-tabular?explain=true",
        json={"features": {"temp_mean_24h": 1.5, "inconnue": 3.0}},
        headers=auth_headers(),
    )
    assert resp.status_code == 200
    explication = resp.json()["explication"]
    assert explication["decision"]
    assert {"facteurs_hausse", "facteurs_baisse", "avertissements"} <= set(explication)
    avert = " ".join(explication["avertissements"])
    assert "Pression moyenne sur 24 h : valeur manquante" in avert
    assert "inconnue" in avert


def test_ready_refuses_model_with_excluded_feature(client, fake_model, monkeypatch):
    from sklearn.impute import SimpleImputer
    from sklearn.pipeline import Pipeline
    from xgboost import XGBClassifier

    import indusense.api.main as api_main

    X = pd.DataFrame(
        {"temp_mean_24h": [1.0, 2.0, 3.0, 4.0], "future_incident_count_24h": [0, 1, 0, 1]}
    )
    pipe = Pipeline([("imputer", SimpleImputer()), ("model", XGBClassifier(n_estimators=2))]).fit(
        X, [0, 1, 0, 1]
    )
    monkeypatch.setattr(api_main, "_model", pipe)
    resp = client.get("/ready")
    assert resp.status_code == 503
    assert "future_incident_count_24h" in " ".join(resp.json()["detail"]["raisons"])


def test_ready_refuses_certification_of_another_model(client, fake_model, monkeypatch):
    import indusense.api.main as api_main

    monkeypatch.setattr(api_main, "_model_version", "aaaaaaaaaaaa")
    monkeypatch.setattr(
        api_main,
        "_certification",
        {"model_version": "bbbbbbbbbbbb", "conforme": True, "constats": []},
    )
    assert client.get("/ready").status_code == 503


def test_ready_refuses_blocking_certification(client, fake_model, monkeypatch):
    import indusense.api.main as api_main

    monkeypatch.setattr(api_main, "_model_version", "aaaaaaaaaaaa")
    monkeypatch.setattr(
        api_main,
        "_certification",
        {
            "model_version": "aaaaaaaaaaaa",
            "conforme": False,
            "constats": [
                {
                    "controle": "concentration",
                    "gravite": "bloquant",
                    "variable": "x",
                    "message": "83 %",
                }
            ],
        },
    )
    assert client.get("/ready").status_code == 503
