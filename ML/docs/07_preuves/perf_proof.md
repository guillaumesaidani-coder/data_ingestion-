# Preuve — harnais de charge (doc "sous la jauge", modules 33-34)

`perf/locustfile.py` : trafic réel vers `/predict-tabular` (payload =
une vraie ligne du Gold, MACH-01, 2026-04-14, split test — pas un dict
inventé), pour observer les panneaux Grafana/Prometheus en conditions
réelles plutôt qu'en lisant du code. Mesuré contre la vraie stack
Compose (`docker compose up -d --wait`), pas en local isolé.

```
uv run --frozen locust -f perf/locustfile.py --host http://127.0.0.1:8010 \
    --headless -u 3 -r 1 -t 100s

150 requêtes, 36 échecs (24.00%) — tous des 429 (rate limit, module 26,
60/min, franchi par 3 users soutenus sur le même endpoint)
p50=9ms  p95=14ms  p99=26ms  max=62ms
```

## Compteur brut vs `rate()`

```
indusense_http_requests_total{path="/predict-tabular", status="200"} = 104   # ne redescend jamais
rate(indusense_http_requests_total{path="/predict-tabular"}[30s])           # pendant la charge
  status="200" -> 1.01   status="429" -> 0.27

# 35s après l'arrêt de Locust, même requête :
  status="200" -> 0   status="429" -> 0   status="422" -> 0   status="413" -> 0
```

Le compteur reste à sa dernière valeur ; `rate()` retombe honnêtement à
0 — comportement identique à celui documenté par le formateur.

## p50/p95/p99 — pas de collapse, contrairement au doc formateur

```
histogram_quantile(0.5|0.95|0.99, sum by (le) (rate(indusense_http_request_duration_seconds_bucket{path="/predict-tabular"}[2m])))
p50 = 6.58 ms   p95 = 9.73 ms   p99 = 54.28 ms
```

Trois valeurs distinctes, pas une ligne confondue. Le doc formateur
observe l'inverse (p50=p95=p99, toutes les requêtes sous le premier
bucket `le=0.1`) : nos buckets par défaut (`prometheus_client`, non
personnalisés) sont `.005/.01/.025/.05/.075/.1/...` — bien plus fins
sous 100 ms que leur histogramme, donc pas la même interpolation.
Vérifié, pas supposé : différence réelle entre les deux implémentations,
pas une reproduction du "piège" du guide.

## 422 et 413 — tous les deux visibles, contrairement au blind spot du guide

```
curl -X POST .../predict-tabular -d '{"features":{}}'        -> 422
curl -X POST .../predict-tabular --data-binary @big(70000o)  -> 413

indusense_http_requests_total{path="/predict-tabular", status="422"} = 1
indusense_http_requests_total{path="/predict-tabular", status="413"} = 1
```

Le doc formateur documente un angle mort réel chez eux (le 413 ne
traverse jamais leur middleware d'instrumentation). Chez nous, testé et
confirmé **absent** : `add_request_id` (qui incrémente les métriques)
est déclaré après `limit_body_size` dans `main.py`, donc il enveloppe ce
dernier et voit son retour même en cas de rejet — les deux codes
d'erreur sont comptés.

## Trouvaille non prévue par le guide

24 % d'échecs (429) pendant la charge : 3 users à `wait_time(1,3)` sur
un seul endpoint dépassent readily le plafond de 60 req/min (module 26).
Pas un bug — le rate limit fonctionne comme conçu, juste plus agressif
que le trafic bas-volume du guide (3 users, cible non précisée). À
ajuster (`wait_time` plus long, ou plusieurs endpoints) si l'objectif
est un test de charge sans 429, plutôt qu'une preuve que le rate limit
réagit sous charge réelle.

## Reproductible

```
docker compose up -d --wait
uv run --frozen locust -f perf/locustfile.py --host http://127.0.0.1:8010 --headless -u 3 -r 1 -t 10m
```
