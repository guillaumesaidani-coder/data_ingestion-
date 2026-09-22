# Preuve Compose — 4 services, readiness gating

Règle du module : `condition: service_healthy` fait attendre le *HEALTHCHECK*
du service amont, pas seulement son démarrage. Chaque ligne ci-dessous a été
observée pendant cette séance, pas déduite de `compose.yaml`.

| Contrôle | Statut | Preuve |
|---|---|---|
| Config Compose valide | Implémenté | `docker compose config -q` → aucune erreur |
| Readiness gating (db → api → prometheus → grafana) | Implémenté | `docker compose up -d --build` : chaque service `Healthy` avant que le suivant ne passe `Starting` |
| Scrape Prometheus réussi | Implémenté | `/api/v1/targets` → `indusense-api` `up` |
| Smoke test automatisé | Implémenté | `uv run pytest tests/test_smoke_compose.py -q` → 5 passed |

## Ports remappés

`8000`/`9090`/`3000` étaient déjà occupés sur ce poste par une autre stack
Docker (`cisia_j6_gameday_guigui-*`, sans rapport avec ce projet). Remappés
côté hôte uniquement (le réseau interne Compose, lui, utilise toujours les
ports de conteneur d'origine — `api:8000` dans `prometheus.yml` n'a pas
changé) :

| Service | Port conteneur | Port hôte |
|---|---|---|
| api | 8000 | **8010** |
| prometheus | 9090 | **9091** |
| grafana | 3000 | **3010** |

## TP1 — Config valide

```
docker compose config -q
→ (rien : configuration valide)
```

`db` (`postgres:16`, healthcheck `pg_isready -U indusense_user`, volume
`pgdata`), `api` (build local, réutilise le `HEALTHCHECK` du `Dockerfile`,
reçoit `DB_HOST=db` — résolution DNS interne à Compose — plutôt qu'une URL
unique, pour correspondre à `indusense.config.get_db_url()` qui lit des
variables séparées, pas une chaîne de connexion), `prometheus`
(`prom/prometheus:v3.1.0`, healthcheck écrit à la main via `wget --spider`
contre `/-/ready`, absent nativement), `grafana` (`grafana/grafana:11.4.0`,
même logique contre `/api/health`). Volumes nommés `pgdata`,
`prometheus_data`, `grafana_data`.

**Deux ajouts nécessaires, hors `compose.yaml`, pour que ce module soit
vérifiable** (le `Dockerfile` du jalon 05 n'avait ni l'un ni l'autre) :
- `HEALTHCHECK` dans le `Dockerfile` (`python -c "urllib.request.urlopen(...)"`
  contre `/health` — pas de `curl`/`wget` dans `python:3.13-slim`, éviter de
  les installer juste pour la sonde).
- Endpoint `GET /metrics` (bibliothèque `prometheus_client`, un `Counter` et
  un `Histogram` alimentés depuis le middleware `add_request_id` déjà en
  place) : sans lui, Prometheus n'aurait jamais eu la cible `indusense-api`
  à scraper.

## TP2 — Ordre réel de démarrage

```
docker compose up -d --build
...
 Container ml-db-1 Starting
 Container ml-db-1 Started
 Container ml-db-1 Waiting
 Container ml-db-1 Healthy
 Container ml-api-1 Starting          # seulement maintenant
 Container ml-api-1 Started
 Container ml-api-1 Waiting
 Container ml-api-1 Healthy
 Container ml-prometheus-1 Starting   # seulement maintenant
 Container ml-prometheus-1 Started
 Container ml-prometheus-1 Waiting
 Container ml-prometheus-1 Healthy
 Container ml-grafana-1 Starting      # seulement maintenant
 Container ml-grafana-1 Started
```

```
docker compose ps
NAME              STATUS
ml-api-1          Up 16 seconds (healthy)
ml-db-1           Up 22 seconds (healthy)
ml-grafana-1      Up 5 seconds (healthy)
ml-prometheus-1   Up 10 seconds (healthy)
```

Les 4 services `healthy` dès le premier `up` (aucune course observée) —
chaque maillon a bien attendu le `Healthy` du précédent avant de démarrer.

## TP3 — Smoke test réel + down propre

```
uv run pytest tests/test_smoke_compose.py -v
tests/test_smoke_compose.py::test_api_health PASSED
tests/test_smoke_compose.py::test_api_ready PASSED
tests/test_smoke_compose.py::test_prometheus_ready PASSED
tests/test_smoke_compose.py::test_grafana_health PASSED
tests/test_smoke_compose.py::test_prometheus_scrapes_api_target_as_up PASSED
5 passed
```

`curl http://localhost:9091/api/v1/targets` → `indusense-api up` (confirmé
aussi en JSON, `activeTargets[0].health == "up"`).

**Sans la stack lancée** (`docker compose down` exécuté avant de rejouer) :
```
uv run pytest tests/test_smoke_compose.py -v
5 skipped
```
Chaque test `SKIP` dès le premier `httpx.ConnectError` — pas d'échec, la
suite complète (`uv run pytest -q`) reste verte en local et en CI sans
Docker Compose (76 passed, 5 skipped, 1 xfailed).

```
docker compose down
 Container ml-grafana-1 Removed
 Container ml-prometheus-1 Removed
 Container ml-api-1 Removed
 Container ml-db-1 Removed
 Network ml_default Removed
```

```
docker volume ls | grep ml_
ml_grafana_data
ml_pgdata
ml_prometheus_data
```
Les 3 volumes nommés survivent à ce `down` (sans `-v`) — confirmé.
