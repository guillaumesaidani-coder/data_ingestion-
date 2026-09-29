# Cadrage : les nouveaux opérateurs d'un lot d'incidents

> **Objet** : cadrer un petit chantier, à traiter dans un chat dédié.
> **Base** : dépôt au commit `ca34bf1` (branche `main`), package `ML/src/indusense/`.
> **Origine** : écart repéré en extrayant la validation de TP4 vers
> `processing/bronze_validation.py` (voir [cadrage_ruptures_package.md](cadrage_ruptures_package.md),
> rupture ①, point 2).

---

## 1. Contexte

Chaque incident est signalé par un opérateur (`operator_name`,
`operator_badge` dans `releves_incidents.csv`). Le modèle de données sépare
les deux :

| Table | Rôle | Alimentée par |
|---|---|---|
| `operator` | Référentiel : 1 ligne par opérateur. `operator_id` (clé), `operator_key` (nom normalisé, **UNIQUE**), `badge_hash` (badge pseudonymisé), `is_active` | TP4, section 4 |
| `bronze_incidents` | Miroir du fichier source, nom et badge en clair | TP4 ou `ingest_terrain_batch` |
| `silver_incident` | `operator_id` (clé étrangère vers `operator`) | TP5 ou `flows/etl_flow.py` |

Le lien se fait dans `processing/silver_events.build_silver_incidents` :
`operator_key = operator_name.strip().lower()`, puis une jointure **gauche**
sur `operator`. Un nom absent du référentiel donne donc `operator_id = NULL`,
**sans erreur ni trace**.

---

## 2. Constat

- **TP4 (section 4)** calcule `operator_key` et `badge_hash` pour toutes les
  lignes du fichier, puis fait un upsert dans `operator` :
  ```sql
  INSERT INTO operator (operator_key, badge_hash) VALUES (:key, :hash)
  ON CONFLICT (operator_key) DO UPDATE SET badge_hash = EXCLUDED.badge_hash
  ```
  avec `badge_hash = sha256(f"{nom.strip().lower()}|{badge.strip().lower()}")[:16]`.
- **`ingest.ingest_terrain_batch`** (source `incidents`) valide et écrit le lot
  en Bronze, mais **ne touche pas à `operator`**.
- **Conséquence** : un opérateur qui apparaît pour la première fois dans un
  lot terrain obtient `operator_id = NULL` dans `silver_incident` après
  `indusense etl`. Rien ne le signale : pas de log, pas de
  `data_quality_issue`, pas de compteur dans `ingestion_batch`.

### Impact

- **Aujourd'hui, sur le Gold et le modèle** : aucun. `operator_id` n'est pas
  une feature du Gold (TP6 ne le lit pas).
- **Sur la traçabilité** : la question « qui a signalé cet incident ? » n'a
  plus de réponse pour les nouveaux lots. C'est exactement ce qu'un audit ou un
  retour terrain demande.
- **Leçon générale** : une jointure gauche vers un référentiel incomplet est
  une perte de données silencieuse. C'est l'objet pédagogique du chantier.

---

## 3. Contraintes (reprises de [cadrage_ruptures_package.md](cadrage_ruptures_package.md) §2)

- **Portabilité SQLite / PostgreSQL** : `ingest.py` tourne contre SQLite dans
  les tests. `INSERT ... ON CONFLICT (col) DO UPDATE` fonctionne sur les deux
  moteurs (SQLite ≥ 3.24). Aucune branche `if backend == ...`.
- **Idempotence** : rejouer le même lot ne doit créer ni doublon dans
  `operator` ni nouvelle ligne Bronze (déjà garanti par `content_hash`).
- **Non-régression** : le contenu de `operator` produit à partir de
  `releves_incidents.csv` doit être identique à celui de TP4 (15 opérateurs
  aujourd'hui, mêmes `badge_hash`).
- **Qualité** : `ruff check src tests`, `black --check src tests` (longueur de
  ligne 100) ; tests unitaires sans Postgres ; le test sur la vraie base est
  marqué `requires_local_infra`.
- **Ne pas confondre les deux pseudonymisations** : `badge_hash` (TP4, sans
  sel, 16 caractères) n'est **pas** `processing/anonymization.anon()` (salé,
  `OP_ANON_…`, utilisé pour produire `releves_incidents_anonymised.csv`). Il
  faut reprendre l'algorithme de TP4 à l'identique, sinon les hash des
  opérateurs existants changent.

---

## 4. Pistes

### Piste A : upsert à l'ingestion (recommandée)

1. Ajouter à `processing/bronze_validation.py` (ou à un petit module
   `processing/operators.py`) une fonction pure :
   `build_operators(bronze_incidents) -> DataFrame[operator_key, badge_hash]`,
   qui reprend TP4 (nom normalisé, `badge_hash`, `drop_duplicates` sur
   `operator_key`).
2. Dans `ingest_terrain_batch`, pour la source `incidents` uniquement :
   upsert des opérateurs, dans le même esprit que TP4.
3. Tests : fonction pure (normalisation, hash identique à TP4, dédoublonnage)
   et ingestion SQLite (nouvel opérateur créé, rejeu sans doublon).

**Pourquoi** : c'est là que TP4 le fait. Le référentiel est à jour dès que le
lot est entré, avant toute reconstruction Silver.

### Piste B : dériver `operator` au moment du Silver

Le flow ETL reconstruirait les opérateurs à partir de tout
`bronze_incidents`, juste avant `rebuild_silver_incident`.

**Avantage** : se « répare » tout seul, y compris pour d'éventuels lots déjà
ingérés sans opérateurs.
**Inconvénient** : un étage Silver qui écrit dans un référentiel, et un
couplage entre deux tables qu'on voulait indépendantes.

### Complément commun aux deux pistes : rendre la perte visible

Quel que soit le choix, `build_silver_incidents` (ou la task du flow) doit
**compter** les incidents sans `operator_id` et le dire : log Prefect au
minimum, idéalement `rows_rejected` ou une ligne `data_quality_issue` de
niveau WARNING. Un `operator_id` NULL n'empêche pas l'incident d'aller au
Gold : c'est un avertissement, pas un rejet.

---

## 5. Décisions à prendre (avec recommandation)

| Question | Recommandation | Raison |
|---|---|---|
| Piste A ou B ? | **A** | Même endroit que TP4 ; le Silver reste une pure transformation |
| Upsert à partir de toutes les lignes ou seulement de `parse_ok=True` ? | **`parse_ok=True` seulement** | TP4 prend toutes les lignes ; mais un opérateur issu d'une ligne rejetée n'est pas fiable (nom mal saisi). Écart assumé, à documenter. Vérifier qu'il ne change rien sur `releves_incidents.csv` (0 rejet aujourd'hui) |
| Badge différent pour un opérateur existant ? | **Garder `DO UPDATE` comme TP4** | Le badge peut être réattribué ; l'historique des badges est hors périmètre |
| `operator_name` vide ? | **Ne pas créer d'opérateur** | Avec `operator_key = NULL`, `ON CONFLICT` ne joue pas (NULL ≠ NULL) : chaque rejeu insérerait une nouvelle ligne. À vérifier au passage : la table actuelle contient-elle déjà une ligne à `operator_key` NULL ? |
| Rendre la perte visible ? | **Oui : log + WARNING** | Voir §4, complément |

---

## 6. Critères d'acceptation

- [ ] Un lot d'incidents avec un opérateur inconnu crée une ligne dans
      `operator` ; après `indusense etl`, l'incident a un `operator_id`.
- [ ] Rejouer le même lot ne crée aucune ligne supplémentaire dans `operator`.
- [ ] Les `badge_hash` calculés sur `releves_incidents.csv` sont identiques à
      ceux de la table `operator` actuelle (test `requires_local_infra`).
- [ ] Un incident sans opérateur connu est signalé (log, et WARNING si retenu).
- [ ] CI au vert : tests unitaires sans Postgres, `ruff`, `black`.

---

## 7. Hors périmètre

- **Données personnelles en clair dans `bronze_incidents`** : `operator_name`
  et `operator_badge` y sont stockés tels quels (miroir du fichier). La
  question RGPD (pseudonymiser dès le Bronze ? purger ?) mérite son propre
  cadrage.
- Historique des badges d'un opérateur, gestion de `is_active`.
- Transformer TP4 en client du package (point 3 de la suite de la rupture ①).

---

## 8. Pour démarrer le chat dédié

1. Lire ce fichier, puis `ML/src/indusense/ingest.py`,
   `processing/bronze_validation.py`, `processing/silver_events.py`,
   `flows/etl_flow.py`, et la section 4 de `ML/TP4.ipynb`.
2. Vérifier l'état : `git log --oneline -3`, puis
   `uv run --frozen pytest tests/ -q -m "not requires_local_infra"`.
3. Regarder la table actuelle :
   `docker exec docker-db-1 psql -U indusense_user -d indusense_db -c "SELECT * FROM operator"`.
4. Pour les tests locaux : Docker Desktop lancé, conteneur `docker-db-1`
   démarré, service Windows `postgresql-x64-18` arrêté (conflit sur le
   port 5432).
5. Trancher les décisions du §5, puis implémenter. Un seul commit, avec ses
   tests.
