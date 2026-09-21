"""tests/test_security.py — module 26 : les 4 contrôles "Implémenté" du
registre security_controls.md, chacun avec sa preuve (code HTTP observé).
Le 5e contrôle (audit logging) reste "Planifié v0" : rien ici ne le teste
comme fait, seulement que ce qui est déjà logué ne fuite pas de secret.
"""

import pytest
from conftest import API_FEATURE_NAMES, auth_headers
from fastapi import HTTPException

from indusense.api.main import RATE_LIMIT_PER_MINUTE, rate_limit


# --------------------------------------------------------------- rate limit
def test_rate_limit_function_blocks_after_limit():
    """Preuve directe sur la fonction interne : les RATE_LIMIT_PER_MINUTE
    premiers appels passent, le suivant lève 429."""
    for _ in range(RATE_LIMIT_PER_MINUTE):
        rate_limit("test-client-direct")

    with pytest.raises(HTTPException) as exc_info:
        rate_limit("test-client-direct")
    assert exc_info.value.status_code == 429


def test_rate_limit_burst_returns_429_over_http(client, fake_model):
    """Retour formateur appliqué dès le départ : un test sur la fonction seule
    ne prouve pas qu'elle est branchée sur la route. Rafale HTTP réelle."""
    status_codes = []
    for _ in range(RATE_LIMIT_PER_MINUTE + 10):
        resp = client.post(
            "/predict-tabular",
            json={"features": {name: 1.0 for name in API_FEATURE_NAMES}},
            headers=auth_headers(),
        )
        status_codes.append(resp.status_code)

    assert 429 in status_codes
    assert status_codes[:RATE_LIMIT_PER_MINUTE] == [200] * RATE_LIMIT_PER_MINUTE


# ------------------------------------------------------------ payload size
def test_predict_tabular_rejects_oversized_payload(client, fake_model):
    # ~65 octets/entrée une fois sérialisé (clé longue + JSON) x 2000 > 64 Ko.
    huge_features = {f"feature_padding_key_number_{i:06d}": 1.0 for i in range(2000)}
    resp = client.post("/predict-tabular", json={"features": huge_features}, headers=auth_headers())
    assert resp.status_code == 413


def test_predict_tabular_rejects_unreadable_content_length(client, fake_model):
    resp = client.post(
        "/predict-tabular",
        content=b'{"features": {}}',
        headers={**auth_headers(), "Content-Type": "application/json", "Content-Length": "abc"},
    )
    assert resp.status_code == 400


# -------------------------------------------------------- non-divulgation
def test_no_secret_or_payload_leak_in_logs(client, fake_model, caplog):
    secret_value = auth_headers()["X-API-Key"]
    payload_marker = "MARKER_MACH-TEST-99999"

    with caplog.at_level("DEBUG", logger="indusense.api"):
        client.post(
            "/predict-tabular",
            json={"features": {API_FEATURE_NAMES[0]: 1.0, "marker": payload_marker}},
            headers={"X-API-Key": secret_value},
        )

    logs_text = caplog.text
    assert secret_value not in logs_text
    assert payload_marker not in logs_text
