# Cahier de tests — InduSense (modules 25-39)

> **Public** : évaluateurs, testeurs, formateur — toute personne qui
> doit vérifier que la solution fonctionne réellement, pas juste lire
> qu'elle fonctionne. Chaque cas de test référence la commande exacte
> déjà exécutée en développement et le document de preuve correspondant
> (`*_proof.md`) ; les résultats attendus sont les résultats réellement
> mesurés, pas des cibles théoriques.

## Convention

| Colonne | Signification |
|---|---|
| **ID** | `TC-<domaine>-<numéro>` |
| **Résultat attendu** | ce qui doit être observé — un chiffre, un code retour, une valeur en base |
| **Preuve associée** | fichier `*_proof.md` où ce cas a déjà été exécuté et documenté |

Statut global de chaque domaine à la dernière exécution complète :
✅ tous les cas passent · ⚠️ partiel · ❌ échec non résolu.

## Prérequis communs

Stack démarrée (`docker compose up -d --build --wait`), Postgres source
rempli du Gold InduSense — voir `guide_experimentation_indusense.md`
§1-2 pour la mise en place complète avant de dérouler ce cahier.

---

## 1. Paquet et qualité (M23-24) — ✅

| ID | Objectif | Étapes | Résultat attendu | Preuve |
|---|---|---|---|---|
| TC-PKG-01 | Le paquet s'installe et se construit | `uv sync --frozen` puis `uv build` | build réussi, `dist/*.whl` produit | `pipeline_proof.md` |
| TC-PKG-02 | Lint propre | `uv run --frozen ruff check src tests` | `All checks passed!` | tous les `*_proof.md`, exécuté avant chaque commit |
| TC-PKG-03 | Format propre | `uv run --frozen black --check src tests` | `All done!`, 0 fichier à reformater | idem |
| TC-PKG-04 | Suite de tests locale complète | `uv run --frozen pytest -q` | 119 passed, 5 skipped, 1 xfailed (dernier relevé) | `hitl_proof.md` |
| TC-PKG-05 | CI GitHub Actions verte | pousser sur `ci/quality-and-build` | jobs `quality` et `build` en `success` | historique des runs sur la branche |

## 2. API (M25) — ✅

| ID | Objectif | Étapes | Résultat attendu | Preuve |
|---|---|---|---|---|
| TC-API-01 | Liveness | `curl http://127.0.0.1:8010/health` | `200 {"status":"ok"}` | `guide_experimentation_indusense.md` §3 |
| TC-API-02 | Readiness avec modèle chargé | `curl http://127.0.0.1:8010/ready` | `200 {"status":"ready"}` | idem |
| TC-API-03 | Prédiction valide | `POST /predict-tabular` avec features connues | `200 {"failure_proba_24h": <float 0-1>}` | idem |
| TC-API-04 | Payload sans feature reconnue | `POST /predict-tabular` avec `{"features":{}}` | `422` | `perf_proof.md` |
| TC-API-05 | Métriques Prometheus exposées | `curl http://127.0.0.1:8010/metrics` | texte OpenMetrics, contient `indusense_http_requests_total` | `drift_proof.md` |

## 3. Sécurité (M26) — ✅

| ID | Objectif | Étapes | Résultat attendu | Preuve |
|---|---|---|---|---|
| TC-SEC-01 | Clé API absente rejetée | `POST /predict-tabular` sans `X-API-Key` | `401` | `security_controls.md` |
| TC-SEC-02 | Corps trop gros rejeté | payload > 64 Kio | `413` dès le 65 537e octet | `security_controls.md`, `perf_proof.md` |
| TC-SEC-03 | Rate limit déclenché sous charge réelle | Locust 3 users, 100 s, sur `/predict-tabular` | des `429` apparaissent (mesuré : 24 % d'échecs sur 150 req) | `perf_proof.md` |
| TC-SEC-04 | Le 413/422 restent comptés dans les métriques | injecter un payload >64 Kio puis un payload invalide, relire `/metrics` | `indusense_http_requests_total{status="413"}` et `{status="422"}` > 0 | `perf_proof.md` |

## 4. Conteneur Docker (M27) — ✅

| ID | Objectif | Étapes | Résultat attendu | Preuve |
|---|---|---|---|---|
| TC-DOCKER-01 | Image se construit | `docker build -t indusense-api .` (depuis `ML/`) | build réussi | `image_proof.md` |
| TC-DOCKER-02 | Utilisateur non-root | `docker compose exec -T api id -u` | `10001` | `image_proof.md` |
| TC-DOCKER-03 | Healthcheck fonctionnel | `docker compose ps` après démarrage | `api` = `healthy` | `image_proof.md`, `compose_proof.md` |
| TC-DOCKER-04 | Taille sous le seuil | `python scripts/check_image.py <image>` | ≤ 750 Mo (mesuré : 690,9 Mo) | `image_proof.md` |

## 5. Stack Compose (M28) — ✅

| ID | Objectif | Étapes | Résultat attendu | Preuve |
|---|---|---|---|---|
| TC-COMPOSE-01 | Tous les services démarrent sains | `docker compose up -d --build --wait` | 4/4 services `healthy` | `compose_proof.md` |
| TC-COMPOSE-02 | Ordre de dépendance respecté | observer les logs de démarrage | `db` healthy avant `api`, `api` avant `prometheus`/`grafana` | `compose_proof.md` |
| TC-COMPOSE-03 | `docker compose config` valide | `docker compose config -q` | code retour 0, aucune erreur | `compose_proof.md` |

## 6. Orchestration Prefect — flow de scoring (M29-30) — ✅

| ID | Objectif | Étapes | Résultat attendu | Preuve |
|---|---|---|---|---|
| TC-PREFECT-01 | Flow exécutable localement | `uv run --frozen python flows/pipeline.py` | `Pipeline terminé : {'rows_scored': 15, 'rows_in_db': 15, ...}` | `pipeline_proof.md` |
| TC-PREFECT-02 | Flow exécutable dans le conteneur | `docker compose run --rm --no-deps api python -m indusense.flows.predict_flow` | même résultat, contre Postgres réel | `pipeline_proof.md` |
| TC-PREFECT-03 | Idempotence | exécuter TC-PREFECT-01 deux fois de suite | `rows_in_db` identique aux deux passages (15), vérifié indépendamment via `psql`/`sqlite3` | `pipeline_proof.md` |
| TC-PREFECT-04 | Reprise sans réentraînement | vérifier `mtime` de `model.joblib` avant/après le flow | inchangé (`ensure_model` réutilise le modèle existant) | `pipeline_proof.md` |

## 7. Dérive — PSI/KS (M31-34) — ✅

| ID | Objectif | Étapes | Résultat attendu | Preuve |
|---|---|---|---|---|
| TC-DRIFT-01 | Références et scénarios se construisent | `uv run --frozen python scripts/drift_windows.py` | 6 fichiers CSV écrits, tailles cohérentes (11k-19k lignes chacun) | `drift_proof.md` |
| TC-DRIFT-02 | Témoin : pas de fausse alerte | `evaluate_drift.py --fenetre 1` | PSI max < 0,25, alerte = non | `drift_proof.md` |
| TC-DRIFT-03 | Capteur en dérive détecté | `evaluate_drift.py --fenetre 2` | PSI temp_mean_24h ≈ 7,38 > 0,25, alerte = OUI, rappel modèle stable (~0,89) | `drift_proof.md` |
| TC-DRIFT-04 | Angle mort du PSI (concept drift) | `evaluate_drift.py --fenetre 3` | PSI < 0,25 (muet) mais rappel effondré (~0,04) | `drift_proof.md` |
| TC-DRIFT-05 | Contre-épreuve de référence | `evaluate_drift.py --fenetre 4 --reference haute_charge` | PSI retombe sous 0,25 (contrairement à la référence `normale`) | `drift_proof.md` |
| TC-DRIFT-06 | Persistance sur 2 fenêtres | rejouer TC-DRIFT-03 une 2e fois | `suivi_fenetres.csv` marque `persistant=True` | `drift_proof.md` |
| TC-DRIFT-07 | Alerte Prometheus réellement déclenchée | démarrer l'exporteur (`export_drift_metrics.py`) après TC-DRIFT-03, consulter `/alerts` | `IndusenseDriftPSIEleve` = `firing` | `drift_proof.md` |

## 8. Observabilité et charge (M33-34 + Locust) — ✅

| ID | Objectif | Étapes | Résultat attendu | Preuve |
|---|---|---|---|---|
| TC-OBS-01 | 2 cibles Prometheus UP | `curl http://127.0.0.1:9091/api/v1/targets` | `indusense-api` et `indusense-drift` = `up` | `drift_proof.md` |
| TC-OBS-02 | Dashboards Grafana provisionnés | `curl -u admin:admin .../api/search?query=InduSense` | 2 dashboards listés (dérive, SLO), dossier `InduSense` | `drift_proof.md` |
| TC-OBS-03 | Panneau dérive lit des données réelles | requête PromQL `indusense_drift_psi{fenetre="2"}` | 8 séries (une par feature surveillée), valeurs cohérentes avec TC-DRIFT-03 | `drift_proof.md` |
| TC-OBS-04 | `rate()` redescend à l'arrêt du trafic | Locust actif puis arrêté, requêter `rate(...)` 35 s après | valeur = 0 alors que le compteur brut reste à son dernier total | `perf_proof.md` |
| TC-OBS-05 | Quantiles de latence distincts | `histogram_quantile(0.5\|0.95\|0.99, ...)` pendant charge | p50/p95/p99 mesurés distincts (6,58/9,73/54,28 ms au dernier relevé) | `perf_proof.md` |

## 9. Boucle HITL (M35-39) — ✅ (chaîne complète honnête : rien promu, comme attendu sur ces données)

| ID | Objectif | Étapes | Résultat attendu | Preuve |
|---|---|---|---|---|
| TC-HITL-01 | Journal versionné peuplé | `backfill_predictions.py` | 19 944 lignes, dont 917 avec `failure_proba_24h ≥ 0.5` (= tp+fp du holdout) | `hitl_proof.md` |
| TC-HITL-02 | Revue ne perd jamais un verdict humain | marquer une ligne revue puis rejouer le flow (TC-PREFECT-01) | `review_status`/`ground_truth` inchangés après le replay | `hitl_proof.md` |
| TC-HITL-03 | Biais de sélection mesuré | `simulate_feedback.py` sans puis avec `--incident-non-predit-rate 1.0` | couverture des vraies pannes dans les retours : 662/727 (91,1 %) → 727/727 (100 %) | `hitl_proof.md` |
| TC-HITL-04 | Streamlit — affiner une fausse alerte | `AppTest`, radio + submit sur une ligne `FAUSSE_ALERTE` | `review_status` devient `CAPTEUR_DEFAILLANT` (ou équivalent choisi), commentaire écrit en base | `hitl_proof.md`, `tests/test_streamlit_review.py` |
| TC-HITL-05 | Streamlit — déclarer un incident non prédit | `AppTest`, sélection + submit | `review_status=INCIDENT_NON_PREDIT`, `ground_truth=True` écrits | idem |
| TC-HITL-06 | Arbitrage champion/challenger | `arbitrate_challenger.py` | matrice gain/régression imprimée, décision journalisée dans `reports/hitl/arbitration_log.csv` | `hitl_proof.md` |
| TC-HITL-07 | Règle d'or respectée | lire la dernière ligne de `arbitration_log.csv` | `decision=REJET_DU_MODELE_N` malgré un bilan net positif (gains=9, régressions=7) | `hitl_proof.md` |
| TC-HITL-08 | Feux verts Prefect — cycle déclenché | `retrain_flow.py` sur la base peuplée | 4/4 feux `OK`, arbitrage relancé, même décision qu'à TC-HITL-07 | `hitl_proof.md` |
| TC-HITL-09 | Feux verts Prefect — cycle suspendu | `retrain_flow.py` avec `PREDICTIONS_DB_URL` pointant une base vide | `status=SUSPENDU`, feux 1 et 2 rouges, aucun entraînement lancé | `hitl_proof.md` |
| TC-HITL-10 | Mode fantôme respecte l'arbitrage réel | `run_shadow_mode.py` après TC-HITL-06 | `status=PAS_ELIGIBLE`, `model.joblib` inchangé (pas de `.backup-*` créé) | `hitl_proof.md` |
| TC-HITL-11 | Chemin mécanique de promotion (isolé) | `pytest tests/test_shadow.py -m requires_local_infra` | test `..._promotes_a_genuinely_accepted_challenger` passe : sauvegarde créée, `model.joblib` remplacé | `hitl_proof.md` |

---

## Matrice de synthèse

| Domaine | Cas de test | Statut à la dernière exécution |
|---|---|---|
| Paquet & qualité | 5 | ✅ |
| API | 5 | ✅ |
| Sécurité | 4 | ✅ |
| Conteneur | 4 | ✅ |
| Stack Compose | 3 | ✅ |
| Prefect (scoring) | 4 | ✅ |
| Dérive | 7 | ✅ |
| Observabilité + charge | 5 | ✅ |
| Boucle HITL | 11 | ✅ (chaîne honnête : `REJET`/`PAS_ELIGIBLE` sont les résultats corrects sur ces données, pas des échecs) |
| **Total** | **48** | — |

## Note de lecture — TC-HITL-06 à 10

Ces cas ne « réussissent » pas en promouvant un nouveau modèle : ils
réussissent parce que le mécanisme **refuse correctement** de le faire
sur les données réelles disponibles (7 régressions pour 9 gains ne
suffisent pas). Un testeur qui s'attend à voir `model.joblib` changer à
la fin de ce cahier se trompe d'objectif — la preuve attendue est que
la chaîne de décision a fonctionné, pas qu'elle ait accepté un
candidat. TC-HITL-11 est le seul cas qui construit délibérément un
arbitrage accepté (fabriqué pour l'occasion) afin de prouver que le
mécanisme de bascule fonctionnerait s'il devait un jour être sollicité.
