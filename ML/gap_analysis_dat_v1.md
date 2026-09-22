# Gap analysis — DAT_InduSense_Sprint3_v1.0.pdf vs le build réel

Analyse du DAT (38 pages, 18 sections, modules 23-34, daté 04/09/2026)
contre l'état réel du dépôt `ML/` tel que construit dans ce chat
(modules 25 à 34 + harnais Locust). Le DAT est une architecture *cible*
très complète (gouvernance de release signée, promotion staging/prod,
watermarks, edge proxy, Alertmanager...) ; notre build est un système
qui tourne réellement, mais nettement plus simple sur plusieurs axes
structurants. Ce document liste les écarts, pas les défauts de code —
rien ici n'est un bug, sauf le point §0.

## 0. Le point le plus important : port 9109/9110 probablement inversés

Le DAT (chapitre 8.2, 8.3, 10.1) est explicite : **drift-exporter → 9109**,
**prefect-exporter → 9110**. Notre exporteur de dérive (`export_drift_metrics.py`)
tourne sur **9110** — choisi parce qu'un process occupait déjà 9109 en
permanence sur cette machine, démarré le **2026-09-03**. Le DAT est
daté du **2026-09-04** (v1.0). Ce n'est presque certainement pas un
processus abandonné : c'est **l'exporteur de référence du corrigé**,
conforme à ce même DAT, laissé actif sur la machine partagée.

Conséquence concrète : notre exporteur devrait être **sur 9109**, et le
port 9110 est réservé à un **exporteur Prefect que nous n'avons pas
construit** (voir §4). Documenté dans `drift_proof.md` comme « piège
Windows », alors que c'est en réalité un conflit de convention de port
avec le corrigé officiel. À corriger si l'objectif est la conformité DAT
(nécessite de coexister avec l'exporteur de référence — probablement le
renommer/l'arrêter n'est pas de notre ressort sur une machine partagée).

## 1. Vue d'ensemble par module (traçabilité DAT §16)

| Module | Objet DAT | Statut réel |
|---|---|---|
| M23 | Paquet Python propre | ✅ Conforme — `pyproject.toml`, `src/`, `tests/` |
| M24 | Qualité et CI | ⚠️ Partiel — CI existe mais sans scan de secrets, audit de dépendances, ni build/smoke-test d'image (voir §6) |
| M25 | API REST | ⚠️ Partiel — FastAPI/Pydantic/Uvicorn/Swagger présents, mais contrat d'entrée différent (voir §2) |
| M26 | Sécurité API | ⚠️ Partiel — clé API et rate limit réels, mais en mémoire par process, pas de proxy/TLS (voir §5) |
| M27 | Conteneur | ✅ Conforme — multi-stage, non-root UID 10001, healthcheck |
| M28 | Stack locale | ⚠️ Partiel — Compose + API + Postgres + Prometheus + Grafana réels, mais pas d'edge proxy ni d'Alertmanager |
| M29 | Design d'orchestration | 🔴 Divergent — un seul flow monolithique, pas deux deployments Prefect Server (voir §4) |
| M30 | Exploitation du flow | 🔴 Divergent — idempotence prouvée, mais pas de watermark ni de work pools/workers persistants |
| M31 | Concepts de drift | ✅ Conforme sur le fond (PSI/KS, référence figée) — seuils à 2 paliers au lieu de 3 (voir §7) |
| M32 | Rapport et alerting | ⚠️ Partiel — CSV produits, alerte Prometheus déclenchée, pas de rapports JSON versionnés avec schéma/hash |
| M33 | Observabilité | ⚠️ Partiel — `/metrics`, Prometheus, SLI présents ; pas de SLO formels versionnés ni de recording rules |
| M34 | Dashboards/runbooks | ⚠️ Partiel — 2 dashboards + runbook réels ; DAT en attend 4, avec Alertmanager (voir §7) |

Rien n'est à 0% ; rien n'est à 100% non plus. Le socle fonctionnel
existe et *tourne réellement* — l'écart est presque entièrement sur la
**gouvernance** (release, promotion, watermark) plutôt que sur les
briques techniques elles-mêmes.

## 2. Contrat API — divergent

| Point DAT | Notre build |
|---|---|
| Entrée `machine_id` + ≥7 relevés `{timestamp, temperature, pressure_bar}` | Entrée générique `{"features": {...70 clés...}}` |
| En-tête `Idempotency-Key`, conflit 409 sur payload divergent | Absent — pas de rejeu idempotent côté API |
| Réponse inclut `release_id`, `model_version`, `threshold` | Réponse = `{failure_proba_24h}` seul |
| `api_predictions` séparée de `predictions` (batch) | Une seule table `predictions`, écrite par le flow uniquement — l'API ne persiste rien |
| `/ready` expose `release_id`, version, digest, seuil (réseau interne) | `/ready` renvoie `{"status":"ready"}` sans métadonnées |
| Edge proxy NGINX unique, termine TLS, écrase les en-têtes de forwarding | Aucun proxy — Compose expose l'API directement |

Le modèle RandomForest 12 features du DAT vs notre XGBoost b11-gkf
~70 features est un choix assumé plus tôt dans le projet (données
réelles Postgres bronze/silver/gold, pas le CSV/TSV démonstrateur du
DAT) — pas un gap à corriger en soi, juste à documenter comme un écart
délibéré par rapport au corrigé.

## 3. Gouvernance de release — absente

Le DAT structure tout son chapitre 9 autour d'un **manifeste de release
signé** (`release_id = sha256({image_digest, commit_sha, model_version,
model_sha256, feature_schema_version, threshold, db_expand_revision})`),
d'une table `model_promotions` (CANDIDATE→APPROVED) et
`release_activations` (bascule atomique, rollback automatique unique).

Rien de tout cela n'existe dans notre build :
- pas de `release_id` nulle part (ni en base, ni dans les métriques, ni dans l'API) ;
- un seul modèle (`model.joblib`) chargé directement, pas de registre `model_artifacts/<sha256>/` ;
- pas de statut CANDIDATE/APPROVED/ACTIVE — le modèle en place est *de facto* actif, sans promotion tracée ;
- pas de rollback automatisé (juste un `model.joblib.dvc` versionné par DVC).

C'est cohérent avec l'absence de plusieurs environnements réels
(staging/production) dans notre build — un seul environnement local +
CI. Le dispositif de release du DAT n'a de sens qu'avec cette
séparation d'environnements.

## 4. Orchestration Prefect — divergence structurelle majeure

| DAT | Notre build |
|---|---|
| Deux **deployments** Prefect Server distincts (`indusense-train`, `indusense-score-hourly`), work pools et workers dédiés | Un seul flow (`indusense_pipeline`) exécuté à la demande (`python -m indusense.flows.predict_flow`) |
| `run_mode` fermé (`scheduled` / `rescore_release` / `manual_replay`) | Aucun — un seul mode d'exécution |
| Table `scoring_watermarks`, avance uniquement après persistance complète | Aucun watermark — chaque exécution retraite tout le Gold disponible |
| État `SKIPPED_NO_NEW_SOURCE` si rien de neuf | Absent — le flow rejoue systématiquement |
| Exporteur Prefect dédié sur :9110 (runs, retries, retard de démarrage) | Aucun exporteur Prefect — le port 9110 est occupé par notre exporteur de dérive (voir §0) |
| Serveur Prefect durable + UI (`prefect-server:4200`) | Prefect utilisé en mode éphémère (le DAT réserve explicitement ce mode « aux tests », §6.3) |

C'est l'écart le plus structurant : le DAT attend une plateforme
d'orchestration persistante avec deux pipelines métier séparés ;
`predict_flow.py` fait tout en un seul flow ponctuel. Fonctionnellement
prouvé (idempotence, reprise via modèle déjà entraîné), mais
architecturalement à l'opposé du modèle cible.

## 5. Sécurité — rate limit non partagé, pas de proxy

- Rate limit (module 26) : `_rate_limit_state` est un `dict` en mémoire
  du process Python — correct pour un seul worker Uvicorn, mais le DAT
  exige un **quota partagé entre workers** (zone mémoire NGINX). Avec
  plusieurs workers Uvicorn, notre limite serait multipliée par le
  nombre de workers, pas partagée.
- Pas d'edge proxy → pas de TLS, pas de séparation entre en-têtes de
  forwarding reçus du client et reconstruits par un proxy de confiance
  (`X-Forwarded-For` n'est pas traité chez nous).
- Pas de scan de secrets automatisé en CI (le DAT l'exige en gate #5,
  chapitre 9.2) — confirmé absent de `.github/workflows/ci.yml`.
- Pas de filesystem en lecture seule sur le conteneur API (`read_only`
  absent de `compose.yaml`) — le DAT l'exige en 5.7 et ADR-007.

## 6. CI/CD — gates partielles

`ci.yml` réel : `uv sync --frozen` → `ruff check` → `black --check` →
`pytest -m "not requires_local_infra"` → (job séparé) `uv build`.

Écarts vs chapitre 9.2/9.3 du DAT :
- **`--frozen` au lieu de `--locked`** dans le job qualité : le DAT est
  explicite (principe #3, §3.3) — `--locked` fait échouer la CI si le
  verrou diverge de `pyproject.toml` ; `--frozen` l'accepte tel quel
  sans vérifier la cohérence. `--frozen` est correct dans le Dockerfile
  (le workspace n'y est pas encore copié), pas dans la CI qualité.
- **Aucune construction ni smoke-test d'image Docker en CI** — le
  Dockerfile/Compose existent et ont été prouvés manuellement (modules
  27-28), mais la CI ne construit jamais l'image ; c'est une gate
  explicite du DAT (`9.2`, point 6).
- Pas de scan de secrets ni d'audit de dépendances en CI.
- Pas de signature (`cosign`), pas de SBOM, pas de `images.lock.yml`.
- Un seul environnement (pas de staging/production, donc pas de
  promotion ni de comparaison de digests entre environnements).

## 7. Observabilité et drift — notre point le plus abouti, encore partiel

Ce qu'on a de solidement construit et prouvé (modules 31-34) :
PSI/KS réels, 2 références figées, 4 scénarios rejoués, `drift_spec.md`
+ `docs/runbook.md`, exporteur Prometheus, alerte réellement
déclenchée et observée `firing`, 2 dashboards Grafana provisionnés.

Ce qui manque pour coller au DAT (chapitre 10) :
- **Seuil PSI à 3 paliers** (`<0,10` stable, `0,10-0,25` à surveiller,
  `≥0,25` dérive forte) — nous n'avons qu'un seuil binaire à 0,25.
- **Fenêtre glissante de 7 jours** avec cutoff à **26h** (horizon 24h +
  délai d'arrivée des incidents 2h) — nos fenêtres sont des mois
  calendaires entiers, pas des fenêtres glissantes avec cutoff.
- `indusense_labels_coverage_ratio`, `indusense_model_metrics_available`,
  `indusense_drift_report_age_seconds` / fraîcheur — absents ; nos
  métriques n'expriment pas la fraîcheur ni la couverture des labels.
- `indusense_pressure_reuse_ratio` — n'a pas de sens chez nous (pas de
  `merge_asof` température/pression avec tolérance 90 min ; notre Gold
  est déjà fusionné en amont).
- **Alertmanager absent** — nos règles Prometheus passent bien à
  `firing`, mais rien ne route vers un destinataire (pas de receiver,
  pas d'inhibition/silence testés).
- `monitoring/slo.yaml` versionné avec recording rules formelles
  (error budget, états `NO_DATA`/`INSUFFICIENT_WINDOW`/`VALID`) —
  absent ; notre dashboard SLO calcule les mêmes idées en PromQL direct
  dans les panneaux, pas en recording rules versionnées.
- **4 dashboards attendus** (`indusense-overview`, `indusense-prefect`,
  `indusense-drift`, `indusense-freshness`) vs **2 construits**
  (dérive, SLO) — pas de vue Prefect (cohérent avec l'absence
  d'exporteur Prefect, §4), pas de vue fraîcheur dédiée.

## 8. Persistance — modèle simplifié

- Une seule table `predictions` (clé `machine_id, window_start`), pas
  de séparation batch/API (`predictions` / `api_predictions` du DAT),
  pas de `prediction_attempts` (journal append-only des tentatives),
  pas de vues `predictions_current` / `predictions_history`.
- Pas de `release_id` en colonne — impossible de tracer quelle version
  du modèle a produit quelle ligne au-delà du fichier `model.joblib`
  courant.
- Une seule base logique (`indusense_db`, un seul rôle
  `indusense_user`) — le DAT sépare `indusense` et `prefect` avec des
  rôles distincts sans droits croisés (moot ici puisque Prefect tourne
  en mode éphémère, §4).
- Pas de politique de rétention/partitionnement, pas de legal hold.

## 9. Ce que notre build fait *mieux* que le scénario du DAT

Pour rester équilibré — tout n'est pas un manque :
- **Modèle** : XGBoost tuné (Optuna, `b11-gkf`), ~70 features
  d'ingénierie temporelle sur données réelles (Postgres, 15 machines,
  1 an d'historique horaire) — contre un RandomForest 12 features sur
  un CSV/TSV démonstrateur dans le DAT. PR-AUC test 0,89, ROC-AUC
  0,996, rappel 0,91.
- **Drift** : nos 4 scénarios sont construits sur ces vraies données
  (avec perturbations synthétiques documentées), pas sur un jeu
  pédagogique fixe — F3 reproduit l'angle mort PSI/rappel du DAT
  (§10.2-10.3) sans avoir eu besoin de le forcer.
- **Preuves d'exécution réelle systématiques** : chaque module de ce
  chat a été exécuté pour de vrai (Docker Compose, Postgres restauré
  depuis un dump réel, Prometheus interrogé en direct, Locust en
  charge réelle) — la méthodologie de preuve du DAT (§13, matrice de
  recette exécutable) est largement respectée en esprit, juste sur un
  périmètre plus restreint que ce qu'il décrit.

## 10. Priorisation (si l'objectif est la conformité DAT)

Reprend l'esprit P0/P1/P2 du DAT (§14), appliqué à nos écarts :

| Priorité | Écart | Pourquoi |
|---|---|---|
| P0 | Aucun release_id / manifeste de release | Sans ça, aucune preuve de quelle version du modèle a produit quelle prédiction — risque direct sur la traçabilité (ADR-006, ADR-012 du DAT) |
| P0 | CI n'exécute jamais le build/smoke-test Docker | Une régression du Dockerfile ne serait détectée qu'en local, jamais en CI |
| P1 | Deux deployments Prefect séparés (train/score) + watermark | Nécessaire pour un vrai mode "production" horaire sans réentraînement implicite (ADR-013) |
| P1 | Alertmanager + routage d'alertes | Une alerte Prometheus qui ne route nulle part n'est qu'une valeur affichée, pas une astreinte fonctionnelle |
| P2 | 3 paliers PSI, fenêtres glissantes 7j + cutoff 26h | Affine la sensibilité, pas bloquant pour la démonstration du mécanisme |
| P2 | Séparation predictions/api_predictions + prediction_attempts | Utile seulement si l'API se met à persister elle-même (aujourd'hui elle ne le fait pas) |

## Corriger en premier

Le point §0 (port 9109/9110) est le seul élément de ce document qui
ressemble à une **erreur** plutôt qu'à un écart de périmètre assumé —
tout le reste est une différence d'ambition entre un système qui tourne
réellement et une architecture cible de référence beaucoup plus large.
