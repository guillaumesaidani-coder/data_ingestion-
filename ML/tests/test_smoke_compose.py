"""tests/test_smoke_compose.py — module 28 : frappe le réseau réel de la
stack `docker compose up` (pas un TestClient in-process). Si la stack n'est
pas lancée, chaque test SKIP dès le premier ConnectionError — jamais
d'échec, jamais d'attente de 60s : `uv run pytest -q` reste vert en local
et en CI sans Docker Compose.

Ports (voir compose.yaml) : 8010 (api), 9091 (prometheus), 3010 (grafana) —
remappés depuis 8000/9090/3000, déjà pris par une autre stack sur ce poste.
"""

import os
import time

import httpx
import pytest

API_URL = os.getenv("SMOKE_API_URL", "http://localhost:8010")
PROMETHEUS_URL = os.getenv("SMOKE_PROMETHEUS_URL", "http://localhost:9091")
GRAFANA_URL = os.getenv("SMOKE_GRAFANA_URL", "http://localhost:3010")

TIMEOUT = 3.0


def _get(url: str) -> httpx.Response:
    try:
        return httpx.get(url, timeout=TIMEOUT)
    except httpx.ConnectError:
        pytest.skip(f"Stack Compose non lancée ({url} injoignable) — `docker compose up -d`")


def test_api_health():
    resp = _get(f"{API_URL}/health")
    assert resp.status_code == 200


def test_api_ready():
    resp = _get(f"{API_URL}/ready")
    assert resp.status_code == 200


def test_prometheus_ready():
    resp = _get(f"{PROMETHEUS_URL}/-/ready")
    assert resp.status_code == 200


def test_grafana_health():
    resp = _get(f"{GRAFANA_URL}/api/health")
    assert resp.status_code == 200
    assert resp.json()["database"] == "ok"


def test_prometheus_scrapes_api_target_as_up():
    # Juste après le démarrage, la cible peut encore afficher "unknown" : le
    # premier scrape (scrape_interval: 15s) n'a pas eu le temps de jouer.
    # On retente sur le CONTENU, pas seulement sur le code HTTP.
    deadline = time.monotonic() + 30
    last_health = None
    while time.monotonic() < deadline:
        resp = _get(f"{PROMETHEUS_URL}/api/v1/targets")
        assert resp.status_code == 200
        targets = resp.json()["data"]["activeTargets"]
        api_targets = [t for t in targets if t["labels"].get("job") == "indusense-api"]
        if api_targets:
            last_health = api_targets[0]["health"]
            if last_health == "up":
                return
        time.sleep(2)

    pytest.fail(f"Cible indusense-api jamais 'up' après 30s (dernier statut : {last_health})")
