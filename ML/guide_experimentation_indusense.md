# Guide d'expérimentation — InduSense (modules 25-39)

> **Public** : toute personne qui veut essayer la solution déployée par
> elle-même — évaluateur, formateur, nouveau contributeur. Toutes les
> commandes ci-dessous ont réellement été exécutées au cours du
> développement (voir `pipeline_proof.md`, `drift_proof.md`,
> `perf_proof.md`, `hitl_proof.md`) — rien de spéculatif.

## 0. Ce que vous allez pouvoir faire

Tout tourne en local, aucune dépendance cloud :

1. Démarrer la stack complète (API + Postgres + Prometheus + Grafana) en une commande.
2. Appeler l'API de prédiction et lire sa réponse.
3. Observer la dérive et les métriques SLO dans Grafana, en conditions réelles.
4. Déclencher volontairement une alerte de dérive et suivre le runbook.
5. Rejouer la boucle complète de retour terrain (HITL) : historique de
   prédictions → revue humaine (Streamlit) → arbitrage champion/
   challenger → feux verts Prefect → mode fantôme.
6. Générer du trafic de charge et lire les panneaux Prometheus en direct.

## 1. Prérequis

| Outil | Usage | Vérifier |
|---|---|---|
| Docker Desktop | Compose (db/api/prometheus/grafana) | `docker compose version` |
| Python 3.13 + `uv` | scripts, tests, Streamlit | `uv --version` |
| Un PostgreSQL local rempli du Gold InduSense | source de vérité (`gold_machine_hourly_feature`, 15 machines, ~1 an d'historique horaire) | voir §1.1 |

### 1.1 Alimenter Postgres

Ce dépôt ne contient pas de dump du Gold. Deux cas :

- **Vous avez déjà un Postgres de développement rempli** (bronze/silver/gold
  déjà exécuté, cf. `TP1`-`TP6`) : `indusense_db`, utilisateur
  `indusense_user`. Le `.env` par défaut (`DB_HOST=localhost`) le cible
  directement — rien à faire.
- **Environnement neuf** : lancez les notebooks `TP1.ipynb` → `TP6.ipynb`
  dans l'ordre (ingestion bronze → silver → Gold), ou restaurez un dump
  existant dans un conteneur Postgres 16 local. Le schéma est géré par
  Alembic (`alembic upgrade head`, voir `alembic/`).

Le conteneur `db` de `compose.yaml`, lui, démarre **vide** — c'est
volontaire (module 28) : c'est le Postgres de l'API/du scoring en
production, pas la source Gold. Pour les expériences qui ont besoin du
Gold à l'intérieur du conteneur (§4), on restaure un dump dedans une
fois (§4.1).

## 2. Démarrer la stack

```bash
cd ML
docker compose config -q                 # valide compose.yaml
docker compose up -d --build --wait      # db, api, prometheus, grafana
docker compose ps                        # les 4 services doivent etre "healthy"
```

Interfaces disponibles :

| Service | URL locale | Identifiants |
|---|---|---|
| API | http://127.0.0.1:8010 | `X-API-Key: dev-local-key` |
| Swagger | http://127.0.0.1:8010/docs | réseau d'administration seulement (voir `security_controls.md`) |
| Prometheus | http://127.0.0.1:9091 | aucun |
| Grafana | http://127.0.0.1:3010 | `admin`/`admin` par défaut |

## 3. Première prédiction via l'API

```bash
curl -s http://127.0.0.1:8010/health
# {"status":"ok"}

curl -s http://127.0.0.1:8010/ready
# {"status":"ready"}  (503 si le modèle n'a pas pu charger)

curl -s -X POST http://127.0.0.1:8010/predict-tabular \
  -H "X-API-Key: dev-local-key" -H "Content-Type: application/json" \
  -d '{"features": {"temp_mean_24h": 48.0, "pressure_mean_24h": 195.0}}'
# {"failure_proba_24h": ...}
```

Le contrat complet des features attendues (~78 colonnes) est documenté
dans `pipeline_proof.md` ; `scripts/backfill_predictions.py` en contient
un exemple réel complet (`perf/locustfile.py` aussi, pour un payload
prêt à rejouer).

## 4. Observer la dérive et les métriques (Grafana/Prometheus)

### 4.1 Peupler le Postgres du conteneur avec le vrai Gold

```bash
# depuis un Postgres source deja rempli (docker-db-1 ou equivalent) :
docker exec <postgres-source> pg_dump -U indusense_user -d indusense_db \
  --no-owner --no-privileges > dump.sql
docker exec -i ml-db-1 psql -U indusense_user -d indusense_db < dump.sql
```

### 4.2 Lancer l'exporteur de dérive (sur l'hôte, pas dans le conteneur)

```bash
uv run --frozen python scripts/drift_windows.py        # fige 2 references + 4 scenarios reels
uv run --frozen python scripts/evaluate_drift.py --fenetre 2
uv run --frozen python scripts/export_drift_metrics.py &   # :9110/metrics, relit les CSV /15s
```

> **Piège connu** : le port 9109 (convention DAT pour le drift-exporter)
> peut déjà être occupé sur une machine partagée par un exporteur laissé
> actif par un tiers — vérifiez avec `netstat -ano | grep 9109` avant de
> lancer le vôtre sur ce port, ou utilisez 9110 comme documenté dans
> `drift_proof.md`.

### 4.3 Regarder les dashboards

- Grafana → dossier **InduSense** → **InduSense — dérive & métriques**
  (PSI par capteur, rappel, taux d'alerte, PSI dans le temps, précision,
  ROC-AUC) et **InduSense — SLO API** (disponibilité, latence p95, taux
  d'erreur, débit par route).
- Prometheus → `http://127.0.0.1:9091/alerts` : `IndusenseDriftPSIEleve`
  passe à `firing` dès que `indusense_drift_psi > 0.25` (déjà le cas
  après l'étape 4.2, fenêtre 2 = capteur en dérive).

### 4.4 Suivre le runbook sur l'alerte

Ouvrez `docs/runbook.md` et suivez-le pas à pas sur l'alerte déclenchée
à l'étape précédente (isoler la feature, vérifier le rappel, exclure un
changement de régime, confirmer la cause physique).

## 5. Générer du trafic de charge

```bash
uv run --frozen locust -f perf/locustfile.py --host http://127.0.0.1:8010 \
  --headless -u 3 -r 1 -t 2m
```

Pendant que ça tourne, dans Prometheus :

```
rate(indusense_http_requests_total{path="/predict-tabular"}[2m])
histogram_quantile(0.95, sum by (le) (rate(indusense_http_request_duration_seconds_bucket[5m])))
```

Détail des deux résultats surprenants trouvés en le faisant : `perf_proof.md`.

## 6. La boucle HITL, pas à pas

Chaque script est un point d'entrée autonome — vous pouvez vous arrêter
après n'importe quelle étape.

### 6.1 Historique de prédictions à réviser

```bash
uv run --frozen python scripts/backfill_predictions.py
# 19944 fenetres reelles scorees (split test, jamais vues a l'entrainement)
```

### 6.2 Revue — simulée puis réelle (Streamlit)

```bash
# simulateur : rejoue la verite terrain reelle deja connue
uv run --frozen python scripts/simulate_feedback.py
uv run --frozen python scripts/simulate_feedback.py --incident-non-predit-rate 1.0

# interface reelle, pour les cas ambigus (capteur defaillant / maintenance)
uv run --frozen streamlit run scripts/streamlit_review.py
# -> ouvre http://localhost:8501
```

Dans Streamlit : la première section propose une fausse alerte à
qualifier (capteur défaillant / maintenance préventive / fausse alerte
confirmée) ; la seconde permet de déclarer un incident que le modèle
n'a pas vu venir.

### 6.3 Arbitrage champion vs challenger

```bash
uv run --frozen python scripts/arbitrate_challenger.py
```

Entraîne un challenger sur les fenêtres avril-mai revues (§6.1-6.2),
l'arbitre contre le champion en production sur juin (jamais vu par
aucun des deux), imprime la matrice gain/stabilité/régression/angle
mort et la décision. Le résultat mesuré sur nos données réelles est
`REJET_DU_MODELE_N` (7 régressions pour 9 gains) — rejouer cette
commande donne le même résultat, déterministe.

### 6.4 Feux verts Prefect avant de lancer un cycle

```bash
uv run --frozen python src/indusense/flows/retrain_flow.py
```

Vérifie 4 conditions réelles (quota de validations, quota de pannes
confirmées, qualité des capteurs, dérive mesurée) avant de relancer
l'arbitrage. Pour voir le cas « suspendu » :

```bash
PREDICTIONS_DB_URL="sqlite:///artifacts/predictions_test_vide.db" \
  uv run --frozen python src/indusense/flows/retrain_flow.py
rm artifacts/predictions_test_vide.db
```

### 6.5 Mode fantôme

```bash
uv run --frozen python scripts/run_shadow_mode.py
```

N'agit que si le dernier arbitrage a été accepté. Sur nos données
réelles, l'arbitrage (§6.3) a conclu au rejet — ce script rapporte donc
`PAS_ELIGIBLE` et ne touche jamais à `model.joblib`. C'est le
comportement attendu, pas une panne : la boucle a correctement refusé
de promouvoir un candidat marginal.

## 7. Rejouer le pipeline de scoring horaire lui-même

```bash
uv run --frozen python flows/pipeline.py
# ou, depuis le conteneur (image deja construite a l'etape 2) :
docker compose run --rm --no-deps api python -m indusense.flows.predict_flow
```

Idempotent : relancez-le, le nombre de lignes en base ne double jamais
(`pipeline_proof.md`).

## 8. Arrêter et nettoyer

```bash
docker compose down            # conserve les volumes (pgdata, prometheus_data, grafana_data)
docker compose down -v         # + supprime les volumes (repart de zero au prochain up)
```

Fichiers locaux générés à supprimer si besoin :
`artifacts/predictions.db`, `artifacts/models/challenger.joblib`,
`reports/hitl/*.csv`, `reports/drift/*.csv` (hors `drift_spec.md`).

## 9. Dépannage — pièges déjà rencontrés dans ce projet

| Symptôme | Cause | Solution |
|---|---|---|
| `predictions.db` a un schéma inattendu après une mise à jour du code | schéma étendu (module 35), base locale ancienne | `rm artifacts/predictions.db`, il est régénéré au prochain scoring |
| L'exporteur de dérive ne répond pas sur `:9109` | port déjà pris par un autre process sur une machine partagée | utiliser `:9110` (déjà la config par défaut de ce dépôt) |
| `docker compose run` avec un `-e CHEMIN=/...` échoue sous Git Bash/Windows | MSYS réécrit les chemins Unix en chemins Windows | `MSYS_NO_PATHCONV=1` devant la commande |
| `uv sync` échoue hors ligne pendant un build Docker | `--frozen` manquant, uv tente de re-résoudre le lock | toujours `uv run --frozen ...` / `uv sync --frozen` en CI et conteneur |
| `arbitrate_challenger.py` plante sur `ValueError: unknown format is not supported` | mélange de dtypes `bool`/`int` avant `pd.concat` | déjà corrigé dans ce dépôt (voir `hitl_proof.md`, module 37) |
