# Preuve du flow Prefect — cache, reprise, idempotence

Règle du module : une task peut être *mise en cache* (rejouée depuis un
résultat déjà connu) ou *reprise* (sautée si son effet existe déjà) ; la
dernière étape doit en plus être *idempotente*. Chaque ligne ci-dessous a
été observée en exécutant réellement `flows/pipeline.py`, pas déduite du
code.

| Contrôle | Statut | Preuve |
|---|---|---|
| Décomposition tasks/flow | Implémenté | 5 tasks Prefect assemblées dans `@flow indusense_pipeline` (`flows/pipeline.py`) |
| Run nommé | Implémenté | `indusense-pipeline-gold_dataset-20260922-090902` |
| Cache sur l'ingestion | Implémenté | 3ᵉ exécution → `Cached(type=COMPLETED)` sur `build-gold-dataset`, dans un process neuf |
| Reprise sur l'entraînement | Implémenté | `ensure-model` réutilise `artifacts/models/model.joblib` à chaque appel — jamais réentraîné (mtime inchangé, `tests/test_flow_pipeline.py`) |
| Idempotence du stockage | Implémenté | `rows_in_db=15` puis `15` après un 2ᵉ passage, sur une base neuve |
| Non-régression | Implémenté | `uv run pytest -q` → 82 passed, 5 skipped, 1 xfailed |

## Les 5 tasks

```
build-gold-dataset  ->  ensure-model  ->  load-latest-features  ->  predict-latest  ->  store-predictions
```

Adapté au projet réel plutôt que copié du modèle du module : notre Gold
vient de Postgres (`gold_machine_hourly_feature`), pas de 3 fichiers plats
joints à la main, et le modèle est `b11-gkf` (XGBoost, `indusense.modeling`),
pas un `RandomForestClassifier`. `predict-latest` score la dernière mesure
de **chaque** machine (15 lignes, une par machine), pas une seule ligne
globale.

## TP1 — Premier lancement, run nommé

```
uv run --frozen python flows/pipeline.py
Flow run 'indusense-pipeline-gold_dataset-20260922-090835' - Beginning flow run
Task run 'build-gold-dataset-f5a'    - Finished in state Completed()
Task run 'ensure-model-de9'          - Reprise : modele existant reutilise -> ...\model.joblib
Task run 'load-latest-features-2cb'  - Finished in state Completed()
Task run 'predict-latest-3db'        - Finished in state Completed()
Task run 'store-predictions-3d8'     - 15 prédictions upsertées -> artifacts\predictions.db (15 lignes en base)
Flow run 'indusense-pipeline-gold_dataset-20260922-090835' - Finished in state Completed()
Pipeline terminé : {'rows_scored': 15, 'rows_in_db': 15, 'db_path': 'artifacts\\predictions.db'}
```

**Bug réel rencontré et corrigé sur ce tout premier lancement** :
`_gold_freshness_cache_key()` retournait la clé de cache brute
(`"132940:2026-06-08 23:00:00+00:00:data\\gold\\gold_dataset.csv"`) —
Prefect l'utilise comme nom de fichier pour persister le résultat, et
les `:` ainsi que les espaces ne sont pas un nom de fichier valide sous
Windows (`OSError: [WinError 123]`). Corrigé en hachant la clé
(`hashlib.sha256(...).hexdigest()`) avant de la retourner — la task avait
fini par se terminer (`Completed()`), mais son résultat n'était jamais
mis en cache tant que ce n'était pas corrigé.

## TP2 — Cache et reprise, process neuf

```
uv run --frozen python flows/pipeline.py     # 3e lancement, meme process shell mais process Python neuf
Task run 'build-gold-dataset-709' - Finished in state Cached(type=COMPLETED)
Task run 'ensure-model-729'       - Reprise : modele existant reutilise -> ...\model.joblib
```

`Cached(type=COMPLETED)` confirme que l'ingestion Gold n'a pas été
rejouée du tout — la preuve survit à la fermeture du terminal entre deux
lancements de `uv run`, chacun un nouveau process Python. `ensure-model`
saute systématiquement l'entraînement (`retrain=False` par défaut) tant
que `artifacts/models/model.joblib` existe déjà — confirmé aussi par
`tests/test_flow_pipeline.py::test_pipeline_does_not_retrain_when_model_exists`
(mtime du fichier inchangé après un appel du flow).

## TP3 — Idempotence du stockage

```
uv run --frozen python scripts/demo_prefect_idempotence.py
1er passage : rows_scored=15 rows_in_db=15
2e  passage : rows_scored=15 rows_in_db=15
OK idempotence : 15 lignes stables apres 2 passages (...\indusense-preuve-bb931s32\predictions.db)
```

Base neuve (`tempfile.TemporaryDirectory`) à chaque lancement du script —
jamais une base déjà remplie, sinon la preuve serait truquée. Comptage
indépendant (`SELECT COUNT(*)`, `count_predictions()`), pas seulement la
valeur renvoyée par le flow. Le script sort en code 1 si les comptes
divergent (jamais observé).

**Piège du module appliqué par anticipation** : `predictions_store.py`
utilise `with closing(sqlite3.connect(...)) as conn` dès sa première
version — `closing()` garantit la fermeture réelle de la connexion
(`with conn:` seul ne fait que committer/annuler, sans fermer), évitant
le `PermissionError` Windows au nettoyage du dossier temporaire que le
module signale comme piège rencontré. Confirmé : aucune erreur de
nettoyage sur les lancements répétés de ce script.

Mécanisme : clé composite `(machine_id, window_start)` +
`INSERT ... ON CONFLICT(machine_id, window_start) DO UPDATE` — un 2ᵉ
passage sur les mêmes données met à jour les lignes existantes au lieu
d'en ajouter (`tests/test_predictions_store.py`, 4 tests dédiés).

---

# Module 30 — Flow empaqueté, Postgres réel

Objectif : sortir le flow de `flows/pipeline.py` (racine, absent de
l'image Docker — seul `src/` y est copié, module 27) vers le paquet
`indusense`, et prouver son idempotence contre un **vrai** PostgreSQL,
dans le **vrai** conteneur `api`, pas seulement en local contre SQLite.

| Contrôle | Statut | Preuve |
|---|---|---|
| Flow empaqueté | Implémenté | `src/indusense/flows/predict_flow.py` ; `flows/pipeline.py` n'est plus qu'une façade à 3 lignes |
| Point d'entrée conteneur | Implémenté | `docker compose run --rm --no-deps api python -m indusense.flows.predict_flow` |
| Stockage portable | Implémenté | `predictions_store.py` réécrit en SQLAlchemy — même SQL upsert sur SQLite (local) et PostgreSQL (conteneur) |
| Idempotence sur Postgres réel | Implémenté | `SELECT COUNT(*) FROM predictions` → 15 après 2 exécutions, vérifié via `docker compose exec db psql` (pas via la valeur renvoyée par le flow) |
| Modèle embarqué réutilisé | Implémenté | `Reprise : modele existant reutilise -> /app/artifacts/models/model.joblib` |
| Non-régression | Implémenté | `uv run pytest -q` → 84 passed, 3 skipped, 1 xfailed ; `ruff check src tests` + `black --check src tests` → OK ; `uv build` → OK |

## TP1 — Le paquet est la source de vérité

`src/indusense/flows/predict_flow.py` contient les 5 `@task` et le
`@flow indusense_pipeline` ; `flows/pipeline.py` se réduit à un import
(`from indusense.flows.predict_flow import indusense_pipeline, main`) et
un appel à `main()`. Installé dans `.venv` par `uv sync` comme tout autre
module du paquet — donc présent dans l'image (`COPY --from=build
/app/.venv /app/.venv`).

## TP2 — Stockage portable, une seule URL, un seul SQL

`predictions_store.py` : `create_engine(url)` + `INSERT ... ON CONFLICT
... DO UPDATE`, valide nativement sur SQLite (≥3.24) et PostgreSQL —
aucune branche `if backend == "postgres"`. `config.get_predictions_engine()`
lit `PREDICTIONS_DB_URL` (Postgres sous Compose) et retombe sur
`sqlite:///artifacts/predictions.db` en local si la variable est absente.

**Deux bugs réels rencontrés et corrigés en le faisant tourner pour de
vrai (pas déduits en lisant le code) :**

1. `store-predictions` reçoit un `Engine` SQLAlchemy (verrou de thread
   interne, non picklable) — la politique de cache par défaut de Prefect
   tentait de le hacher et échouait à chaque appel (`HashError`, non
   fatal mais loggé en erreur). Corrigé par `cache_policy=NO_CACHE` sur
   cette task, qui doit de toute façon toujours s'exécuter.
2. Le pool de connexions SQLAlchemy n'était jamais fermé dans
   `scripts/demo_prefect_idempotence.py` — le fichier SQLite restait
   verrouillé, et `tempfile.TemporaryDirectory` échouait à se nettoyer
   sous Windows (`PermissionError [WinError 32]`). Corrigé par
   `engine.dispose()` dans un `finally`.

## TP3 — Idempotence contre un vrai Postgres, dans le vrai conteneur

```
docker compose up -d --build --wait db api
 Container ml-db-1  Healthy
 Container ml-api-1 Healthy
```

Le Postgres de Compose est neuf (vide) — restauré depuis un `pg_dump` du
Postgres de développement réel (`docker-db-1`, celui rempli par le
pipeline ETL Bronze→Silver→Gold des modules précédents), pour disposer
de vraies données (`gold_machine_hourly_feature`, 132 940 lignes) plutôt
que de rejouer l'ETL dans ce conteneur jetable.

**Deux bugs réels supplémentaires trouvés à ce stade, corrigés pour de
vrai :**

3. `appuser` (non-root, `--no-create-home`, module 27) ne peut pas créer
   `data/gold/` ni le répertoire d'état Prefect dans `/app` (possédé par
   `root`) : `PermissionError: [Errno 13] Permission denied: 'data'` et
   `Failed to create the Prefect home directory`. Corrigé dans le
   `Dockerfile` : `RUN mkdir -p /app/data /app/artifacts && chown -R
   appuser:appgroup /app/data /app/artifacts`, plus `ENV
   PREFECT_HOME=/app/data/.prefect` — seuls ces sous-répertoires passent
   à `appuser`, pas `/app` entier (`.venv` et le modèle restent en
   lecture seule).
4. `PREDICTIONS_DB_URL` codé en dur dans `compose.yaml` avec le mot de
   passe `ThEP@ssW0rd` non encodé : le `@` littéral était lu comme
   séparateur host par `create_engine()`
   (`could not translate host name "ssW0rd@db" to address`). Corrigé en
   encodant le mot de passe (`%40`).

**Piège Windows/Git Bash rencontré, comme annoncé par le module** : un
premier essai avec `-e PREFECT_HOME=/tmp/prefect` sur la ligne de
commande a été réécrit par la conversion MSYS de Git Bash en
`C:/Users/.../Temp/prefect` avant d'atteindre Docker
(`UserWarning: Failed to create the Prefect home directory at
C:/Users/Aelion/AppData/Local/Temp/prefect`) — résolu en fixant
`PREFECT_HOME` directement dans le `Dockerfile` plutôt qu'en argument de
commande, ce qui supprime le besoin de `MSYS_NO_PATHCONV=1` ici.

```
docker compose run --rm --no-deps api python -m indusense.flows.predict_flow
Task run 'build-gold-dataset-c2e'   - Gold exporté : data/gold/gold_dataset.csv
Task run 'ensure-model-0bd'         - Reprise : modele existant reutilise -> /app/artifacts/models/model.joblib
Task run 'store-predictions-905'    - 15 prédictions upsertées -> postgresql+psycopg2://indusense_user:***@db:5432/indusense_db (15 lignes en base)
Pipeline terminé : {'rows_scored': 15, 'rows_in_db': 15, 'db_url': 'postgresql+psycopg2://indusense_user:***@db:5432/indusense_db'}

docker compose run --rm --no-deps api python -m indusense.flows.predict_flow   # 2e passage
Task run 'store-predictions-ecc'    - 15 prédictions upsertées -> postgresql+psycopg2://indusense_user:***@db:5432/indusense_db (15 lignes en base)
Pipeline terminé : {'rows_scored': 15, 'rows_in_db': 15, ...}
```

Vérification indépendante, hors du flow (pas la valeur qu'il prétend
avoir écrite) :

```
docker compose exec db psql -U indusense_user -d indusense_db -c "SELECT COUNT(*) FROM predictions;"
 count
-------
    15
(1 row)
```

15 lignes stables après 2 exécutions dans le conteneur réel — nos 15
machines réelles (pas les 4 lignes du jeu de données jouet du guide).
