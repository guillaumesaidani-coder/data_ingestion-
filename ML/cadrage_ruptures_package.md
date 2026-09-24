# Cadrage : les trois ruptures du package `indusense`

> **Objet** : cadrer trois pistes d'amélioration, à traiter dans un chat dédié.
> **Base** : dépôt au commit `6d7b38f` (branche `main`), package `ML/src/indusense/`.
> **Origine** : ces ruptures sont apparues en déduisant le schéma fonctionnel à
> partir du code (docstrings, imports internes, points d'entrée, flux entre
> domaines). La méthode est détaillée dans `package_indusense_pedagogique.pptx`.

---

## 1. Contexte : le schéma fonctionnel attendu

Le package s'organise en cinq domaines, posés sur un socle (`config.py`) :

| # | Domaine | Modules |
|---|---|---|
| ① | Données | `ingest`, `processing/*`, `data`, `data_quality` |
| ② | Modélisation | `modeling/*`, `cli` |
| ③ | Service | `flows/predict_flow`, `scoring`, `api/main` |
| ④ | Retour humain (HITL) | `predictions_store` (+ `scripts/streamlit_review.py`) |
| ⑤ | Cycle de vie | `flows/retrain_flow`, `drift`, `arbitration`, `shadow` |

La boucle cible est la suivante : ① produit le Gold → ② entraîne → ③ prédit
→ ④ un technicien valide → ⑤ décide de réentraîner puis de basculer → retour en ③.

Les ruptures sont numérotées d'après le domaine où elles se trouvent
(①, ③, ⑤).

**Constat** : cette boucle n'est pas fermée dans le code. Trois maillons
reposent sur une action manuelle (un notebook ou un script lancé à la main)
ou sont absents.

---

## 2. Contraintes communes aux trois chantiers

- **Ne pas casser les notebooks** : TP2 et TP6 importent déjà
  `indusense.processing.*`. Toute modification d'une signature doit rester
  compatible avec eux, ou les notebooks doivent être mis à jour dans le même
  commit.
- **Tests** : environ 135 tests (`uv run --frozen pytest tests/`). La CI exécute
  `-m "not requires_local_infra"`. Un test qui a besoin de Postgres, des
  fichiers DVC ou de `mlflow_tp7.db` doit porter le marqueur
  `requires_local_infra`.
- **Qualité** : `ruff check src tests` et `black --check src tests`
  (longueur de ligne 100) doivent passer, sinon la CI échoue.
- **Idempotence** : les écritures existantes (`upsert_predictions`,
  `ingest_terrain_batch`) sont idempotentes. Les nouveaux branchements
  doivent l'être aussi, car un rejeu ne doit jamais dupliquer de lignes.
- **Portabilité SQLite / PostgreSQL** : `predictions_store` n'a qu'un seul
  jeu de requêtes pour les deux moteurs. Ne pas introduire de branche
  `if backend == ...`.
- **Image Docker** : le Dockerfile ne copie que `src/` et `model.joblib`. Du
  code placé dans `scripts/` ou `flows/` (à la racine de `ML/`) n'est donc
  **pas** dans l'image.
- **Infra locale** : Postgres dans le conteneur `docker-db-1` (port 5432),
  base peuplée par TP4 à TP6, données DVC sur DagsHub.

---

## 3. Rupture ① : la chaîne Bronze → Silver → Gold n'est pas automatisée

### Constat
- `processing/silver_telemetry.build_silver_telemetry` et
  `processing/gold_features.build_gold_features` ne sont appelés **par aucun
  module du package** ni par aucun flow. Leurs seuls appelants sont :
  - les notebooks `TP2.ipynb` (Silver télémétrie) et `TP6.ipynb` (Gold) ;
  - les tests `tests/test_processing_*.py`.
- Le Silver des **incidents** et de la **maintenance** (`silver_incident`,
  `silver_maintenance`) n'a **aucun équivalent dans le package** : sa logique
  n'existe que dans `TP5.ipynb`, qui n'importe rien de `indusense`.
- Le flow de prédiction ne recalcule pas le Gold.
  `predict_flow.build_gold_dataset` appelle `data.export_gold_dataset`, qui
  fait un simple `SELECT * FROM gold_machine_hourly_feature` : il exporte le
  Gold déjà calculé par TP6.
- `ingest.ingest_terrain_batch` ajoute bien un lot en Bronze, mais n'est
  appelé que par `tests/test_ingest.py` et n'accepte que les sources
  `telemetry` et `incidents` (`BRONZE_TABLES`), pas `maintenance`.

### Impact
Un nouveau lot terrain ne se propage jusqu'au Gold que si quelqu'un relance
TP4, TP5 et TP6 à la main. La reproductibilité dépend donc de l'exécution de
notebooks.

### Pistes
1. Extraire la logique Silver incidents/maintenance de TP5 vers
   `processing/` (par exemple `silver_events.py`), selon le modèle de
   `silver_telemetry.py` (fonction pure DataFrame → DataFrame + tests).
2. Créer un flow Prefect `flows/etl_flow.py` (dans le package) :
   `ingest` (Bronze) → Silver (télémétrie, incidents, maintenance) → Gold
   (`build_gold_features`) → écriture en base, avec ouverture et clôture d'un
   `ingestion_batch` à chaque étage, comme le font TP4 à TP6.
3. Faire des notebooks de simples **clients** du package : ils appellent les
   fonctions au lieu de dupliquer la logique.

### Critères d'acceptation
- [ ] Une seule commande (flow ou CLI) reconstruit Silver et Gold à partir du
      Bronze, sans notebook.
- [ ] Le Gold produit est **identique** à celui de TP6 : même nombre de
      lignes (132 940 aujourd'hui) et mêmes colonnes. La preuve peut
      s'appuyer sur l'empreinte md5 de `gold_dataset.csv` exporté
      (`47f3185b…`).
- [ ] Relancer deux fois la chaîne ne duplique rien.
- [ ] Les tests unitaires passent sans Postgres ; le test de bout en bout est
      marqué `requires_local_infra`.

### Questions ouvertes
- Rebuild complet (TRUNCATE, comme TP4) ou incrémental (append, comme
  `ingest_terrain_batch`) ?
- Faut-il ajouter `maintenance` à `ingest_terrain_batch` ?

---

## 4. Rupture ③ : l'API en ligne est isolée de la boucle

### Constat
- `api/main.py` n'importe que `indusense.config` (`get_api_key`,
  `get_model_path`).
- **Logique dupliquée** : `predict_tabular` recalcule lui-même les colonnes
  attendues (`imputer.feature_names_in_` puis `reindex`), ce que fait déjà
  `scoring.model_feature_cols`. Les deux implémentations peuvent diverger.
- **Prédictions non stockées** : l'API ne fait aucun appel à
  `predictions_store`. Une prédiction servie par l'API ne peut donc jamais
  être revue par un technicien (④), ni compter dans les feux verts de
  réentraînement (⑤).
- **Obstacle de contrat** : `PredictRequest` ne contient que
  `features: dict[str, float | None]`. Or la table `predictions` a pour clé
  primaire `(machine_id, window_start)`. En l'état, une requête API ne peut
  donc pas devenir une ligne de `predictions`.
- La réponse ne renvoie pas `model_version`, alors que
  `config.get_model_version()` existe.

### Pistes
1. **Sans changer le contrat** : faire utiliser `scoring.model_feature_cols`
   (ou `score_features`) par l'API pour supprimer la duplication. C'est un
   refactor pur, couvert par `tests/test_api.py`.
2. **Avec évolution du contrat** : ajouter `machine_id` et `window_start` à
   la requête (en optionnel, pour la rétrocompatibilité). Quand ils sont
   fournis, enregistrer via `upsert_predictions` avec `model_version` et
   `features_payload`. Renvoyer `model_version` dans la réponse.
3. Documenter la décision (API « consultative » ou « source de vérité »)
   dans `architecture_deploiement_indusense.md`.

### Critères d'acceptation
- [ ] Une seule implémentation de « quelles colonnes le modèle attend ».
- [ ] Si le contrat évolue : une requête avec identifiants crée ou met à jour
      une ligne de `predictions` ; rejouer la même requête ne crée pas de
      doublon et **n'efface pas** un verdict technicien existant
      (`UPSERT_SQL` ne touche pas les colonnes de revue ; à conserver).
- [ ] Les garde-fous de sécurité restent prouvés par `tests/test_security.py` :
      401, 422, 413, 429.
- [ ] L'image Docker se construit, `/ready` et `/predict-tabular` répondent
      (`make build`, puis test manuel ou `make check-health`).

### Questions ouvertes
- Dans l'image Docker, `PREDICTIONS_DB_URL` pointe vers Postgres
  (`compose.yaml`). Hors compose, le défaut est une SQLite locale au
  conteneur, donc perdue au redémarrage. Faut-il rendre l'écriture
  optionnelle (activée seulement si l'URL est configurée) ?
- L'écriture en base doit-elle bloquer la réponse, ou passer en tâche de fond
  (`BackgroundTasks` de FastAPI) ?

---

## 5. Rupture ⑤ : la bascule du modèle est détachée du cycle

### Constat
- `flows/retrain_flow.retrain_cycle` s'arrête à `run_arbitration` et
  `persist_arbitration`, puis renvoie `{"status": "ARBITRE", ...}`. Aucune
  suite n'est déclenchée.
- `shadow.run_shadow_cycle`, qui fait le mode fantôme puis appelle
  `promote_challenger()` pour remplacer `model.joblib`, n'est appelé que par
  `scripts/run_shadow_mode.py` (manuel) et par `tests/test_shadow.py`. Il
  relit le journal d'arbitrage (`reports/hitl/arbitration_log.csv`).
- **Feu vert n°5 dépendant d'un notebook** : `check_new_terrain_batch` cherche
  un `ingestion_batch` avec `source_name = 'gold'`. Seul TP6 en écrit ;
  `ingest_terrain_batch` n'accepte pas cette source. Le feu ne peut donc
  passer au vert que si TP6 est relancé (lien avec la rupture ①).
- `promote_challenger` remplace le fichier `model.joblib` **local**. Or ce
  fichier est suivi par DVC et embarqué dans l'image Docker publiée sur
  ghcr.io. Une bascule ne met donc à jour ni DVC, ni l'image, ni l'API en
  cours d'exécution, qui charge le modèle une seule fois au démarrage.

### Pistes
1. Chaîner dans `retrain_cycle` : quand la décision fait partie de
   `ACCEPTED_DECISIONS`, lancer `run_shadow_cycle` en sous-flow ou en tâche,
   et journaliser le résultat (`BLOQUE` / bascule).
2. Définir ce qu'est une « bascule » en production : `dvc add` + commit du
   nouveau `model.joblib.dvc`, puis la CI republie l'image, puis
   redéploiement. Et prévoir le retour arrière (la sauvegarde
   `model.backup-*.joblib` existe déjà).
3. Alimenter le feu n°5 depuis le flow ETL de la rupture ① (un batch `gold`
   écrit par le flow, pas seulement par TP6).

### Critères d'acceptation
- [ ] Un seul flow va de « feux verts » à « bascule ou blocage », sans script
      manuel.
- [ ] Une bascule ne se produit jamais sans confirmation par le mode
      fantôme ; une sauvegarde du champion est toujours créée.
- [ ] Le feu n°5 peut passer au vert sans exécuter de notebook.
- [ ] Les tests existants (`test_retrain_flow.py`, `test_shadow.py`,
      `test_arbitration.py`) passent, et un test couvre l'enchaînement.

### Questions ouvertes
- La bascule doit-elle rester soumise à une **validation humaine** (le
  runbook réserve la décision de réentraîner au data scientist) ? Si oui, le
  flow s'arrête sur une demande d'approbation plutôt que de basculer seul.
- Le mode fantôme rejoue aujourd'hui la fenêtre d'arbitrage elle-même (le
  dataset s'arrête au 2026-06-08). Faut-il le garder tel quel ou attendre des
  données plus fraîches ?

---

## 6. Dépendances et ordre suggéré

```
Rupture ③ piste 1 (refactor scoring)   ← indépendant, rapide, sans risque
Rupture ①  (flow ETL)                  ← débloque le feu n°5 de la rupture ⑤
Rupture ③ piste 2 (contrat + stockage) ← décision d'architecture à prendre
Rupture ⑤  (chaînage + bascule)        ← dépend de ① et d'une règle de gouvernance
```

Chaque chantier doit faire l'objet d'un commit (ou d'une PR) séparé, avec ses
tests, et la CI doit rester au vert.

## 7. Hors périmètre

- Réduction de la taille de l'image (xgboost-cpu, dépendances notebooks dans
  un groupe séparé) : chantier distinct.
- Migration de MLflow vers DagsHub.
- Réintégration des tests `test_model_b11` en CI.

## 8. Pour démarrer le chat dédié

1. Lire ce fichier, puis `ML/src/indusense/` (surtout `flows/`, `api/main.py`,
   `scoring.py`, `shadow.py`, `processing/`).
2. Vérifier l'état : `git log --oneline -3`, puis
   `uv run --frozen pytest tests/ -q -m "not requires_local_infra"`.
3. Pour les tests locaux : Docker Desktop lancé, `docker compose start` dans
   `Docker/`, et le service Windows `postgresql-x64-18` arrêté (conflit sur
   le port 5432).
4. Choisir **une** rupture, trancher ses questions ouvertes, puis
   implémenter.
