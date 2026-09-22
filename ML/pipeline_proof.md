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
