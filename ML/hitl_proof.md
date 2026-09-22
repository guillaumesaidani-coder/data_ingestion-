# Preuve — boucle HITL champion/challenger (plan_action_hitl_champion_challenger.md)

## M35 — Journal de prédiction versionné

Objectif : tracer quelle version du modèle a produit quelle prédiction,
avec un instantané des features utilisées, sans jamais perdre un
verdict humain déjà enregistré lors d'un replay du flow.

| Contrôle | Statut | Preuve |
|---|---|---|
| `model_version` sur chaque ligne | Implémenté | `indusense.config.get_model_version()` — SHA-256 tronqué de `model.joblib` |
| `features_payload` (photo JSON) | Implémenté | `predict_latest` sérialise les ~78 colonnes reindexées, NaN → null |
| Colonnes de revue (`review_status`, `ground_truth`, `reviewer_comment`, `reviewed_at`) | Implémenté | `predictions_store.py`, défaut `A_VALIDER` |
| Revue jamais écrasée par un replay | Implémenté | `ON CONFLICT DO UPDATE` ne mentionne que les colonnes de prédiction ; testé (`test_replay_does_not_erase_an_existing_review`) ET vérifié en conditions réelles ci-dessous |
| Non-régression | Implémenté | `uv run pytest -q` → 91 passed, 5 skipped, 1 xfailed (+2 tests vs avant M35) ; `ruff` + `black` OK |

### Décision prise en cours de route : pas de migration Alembic

Le plan prévoyait une migration Alembic (comme les 8 déjà présentes
pour bronze/silver/gold). Vérification faite : `predictions` n'est **pas**
une table gérée par `models.py`/Alembic (`MANAGED_PREFIXES` dans
`alembic/env.py` ne couvre que `bronze_`/`silver_`/`gold_` +
`ingestion_batch`/`operator`/`data_quality_issue`) — elle vit entièrement
dans `predictions_store.py` en SQL portable (SQLite local / Postgres
conteneur, module 30). Faire passer `predictions` sous Alembic aurait
cassé cette portabilité pour un gain nul : la table est intégralement
régénérable (`CREATE TABLE IF NOT EXISTS`), pas une source de vérité à
migrer avec soin. Schéma étendu directement dans `CREATE_TABLE_SQL` ;
`artifacts/predictions.db` (local, jetable) supprimé et régénéré avec le
nouveau schéma plutôt que migré en place.

### Preuve en conditions réelles

```
uv run --frozen python flows/pipeline.py
Task run 'store-predictions' - 15 prédictions upsertées -> sqlite:///.../predictions.db (15 lignes en base)

# verification directe en base
MACH-09 2026-06-08T23:00:00+00:00 0.0002 f550cff24814 A_VALIDER n_features=78
MACH-06 2026-06-08T23:00:00+00:00 0.0004 f550cff24814 A_VALIDER n_features=78
```

Revue marquée sur une vraie ligne (`review_prediction(..., "PANNE_CONFIRMEE", True, "test-reel", ...)`),
puis flow rejoué une 2e fois pour de vrai :

```
uv run --frozen python flows/pipeline.py   # 2e passage
Pipeline terminé : {'rows_scored': 15, 'rows_in_db': 15, ...}

# apres le replay
model_version= f550cff24814   review_status= PANNE_CONFIRMEE   ground_truth= 1   comment= test-reel
```

`model_version` reste identique (même modèle, pas de réentraînement) ;
la revue a survécu au replay, comme le mécanisme le garantit.

## M36 — Simulateur de retour terrain + Streamlit réel

Objectif : construire le mécanisme de revue (simulé + humain), et
mesurer — pas affirmer — le biais de sélection que la feuille de route
signale (§5) : un technicien n'inspecte que les alertes.

| Contrôle | Statut | Preuve |
|---|---|---|
| Historique réel à réviser | Implémenté | `scripts/backfill_predictions.py` — 19 944 fenêtres du split test (avril-juin 2026, jamais vues à l'entraînement), scorées avec le vrai modèle |
| Simulateur de retour terrain | Implémenté | `scripts/simulate_feedback.py` — vérité terrain réelle (`label_failure_next_24h`), pas inventée |
| Biais de sélection mesuré | Implémenté | 65 vraies pannes invisibles dans les retours sans le bouton, 0 avec |
| Streamlit réel (2 flux) | Implémenté | `scripts/streamlit_review.py` — affiner une fausse alerte, déclarer un incident non prédit |
| Streamlit prouvé pour de vrai | Implémenté | `tests/test_streamlit_review.py`, `streamlit.testing.v1.AppTest` — clics réels, relecture en base après coup |
| Non-régression | Implémenté | `uv run pytest -q` → 94 passed, 5 skipped, 1 xfailed (+3 tests) ; `ruff`/`black` OK |

### `score_features()` mutualisé (module 35→36)

`predict_flow.predict_latest` (scoring horaire) et
`backfill_predictions.py` (historique) partagent maintenant
`indusense.scoring.score_features()` — même colonnes attendues, même
sérialisation NaN→null, une seule implémentation plutôt que deux qui
auraient pu diverger (leçon du module 30 appliquée par anticipation).

### Preuve en conditions réelles — backfill

```
uv run --frozen python scripts/backfill_predictions.py
19944 fenêtres scorées (modèle f550cff24814)
  dont 917 alertes (proba >= 0.5) sur 19944 (4.6%)
```

917 alertes = exactement `tp + fp` (662 + 255) du holdout mesuré au
module 30 (`artifacts/models/metrics.json`) — cohérence croisée, pas une
coïncidence : même modèle, même split test.

### Preuve en conditions réelles — biais de sélection

```
uv run --frozen python scripts/simulate_feedback.py
Alertes revues (biais de sélection, comme un vrai technicien) : 917
  -> 662 PANNE_CONFIRMEE, 255 FAUSSE_ALERTE
Pannes manquées par le modèle (faux négatifs) dans ce lot : 65
Couverture des vraies pannes dans les retours : 662/727 (91.1%) sans le bouton
BIAIS DE SÉLECTION MESURÉ : 65 vraies pannes restent invisibles dans les
retours terrain tant que rien ne force leur remontée.

uv run --frozen python scripts/simulate_feedback.py --incident-non-predit-rate 1.0
Déclarées via 'incident non prédit' (taux 100%) : 65
Couverture des vraies pannes dans les retours : 0/65 (0.0%) sans le bouton -> 65/65 (100.0%) avec
```

État final vérifié directement en base (`GROUP BY review_status`) :
`PANNE_CONFIRMEE=662`, `FAUSSE_ALERTE=255`, `INCIDENT_NON_PREDIT=65`,
`A_VALIDER=18962` — ce dernier chiffre est exactement `tn` (18962) du
holdout : les vraies machines saines, jamais inspectées par un
technicien, restent à juste titre non revues.

### Preuve en conditions réelles — Streamlit (AppTest, pas un survol)

`streamlit.testing.v1.AppTest` charge réellement `streamlit_review.py`,
clique les widgets et relit la base après coup :
- affiner une fausse alerte → `CAPTEUR_DEFAILLANT` + commentaire écrits ;
- déclarer un incident non prédit → `INCIDENT_NON_PREDIT`,
  `ground_truth=True` + commentaire écrits ;
- les métriques affichées (`Température`, `Vibration`) viennent bien de
  `features_payload` (module 35), pas de valeurs codées en dur dans la page.

## M37 — Arbitrage champion vs challenger

Objectif : entraîner un challenger sur des données que le champion n'a
jamais vues (pas un réentraînement à l'identique — ça ne prouverait
rien sur un modèle déjà fort, rappel 0,9106), puis arbitrer sur la
matrice gain/stabilité/régression/angle mort de la feuille de route
(§5), jamais sur un score global seul.

| Contrôle | Statut | Preuve |
|---|---|---|
| Challenger entraîné sur données jamais vues | Implémenté | `trainval` (champion) + avril-mai 2026 revu par la boucle HITL (module 36) — juin 2026 réservé à l'arbitrage |
| Matrice gain/stabilité/régression/angle mort | Implémenté | `scripts/arbitrate_challenger.py`, testée isolément (`tests/test_arbitration.py`, 5 tests) |
| Règle d'or (jamais un score global seul) | Implémenté | `arbitration_decision()` — refuse un candidat en net positif si les régressions sont trop nombreuses |
| Décision journalisée | Implémenté | `reports/hitl/arbitration_log.csv`, append à chaque run — équivalent léger de `model_promotions` (gap analysis §3) |
| Non-régression | Implémenté | `uv run pytest -q` → 99 passed (+5) ; `ruff`/`black` OK |

### Bug réel rencontré et corrigé

`pd.concat([gold.y_tv, y_new])` — `y_tv` est `bool` (colonne Postgres
`label_failure_next_24h`), `y_new` (vérité HITL) est `int` : la
concaténation sans caster les deux au même type produit un `Series`
`object`, que `sklearn` refuse (`ValueError: unknown format is not
supported` sur `average_precision_score`). Corrigé en castant les deux
en `int` avant `pd.concat`.

### Preuve en conditions réelles — résultat mesuré, pas arrangé

```
uv run --frozen python scripts/arbitrate_challenger.py
Entraînement augmenté : 112996 lignes (champion) + 17090 lignes avril-mai
  (609 pannes confirmées/déclarées) = 130086 lignes
Arbitrage sur juin 2026 : 2854 fenêtres, jamais vues par aucun des deux modèles
Challenger (mesuré sur juin) : rappel=0.8898 précision=0.7554 ROC-AUC=0.9971

Gain pur (champion faux, challenger juste)   : +9
Stabilité (les deux justes)                  : 2798
Régression (champion juste, challenger faux) : -7
Angle mort (les deux faux)                   : 40
Bilan net : +2
Décision : REJET_DU_MODELE_N
```

Résultat honnête, pas retouché pour "réussir la démo" : bilan net
positif (+2) mais **rejeté** — 7 régressions sur des pannes réelles est
trop pour un gain de 9, la règle d'or de la feuille de route
("un score global supérieur ne suffit pas") fonctionne exactement comme
prévu. `challenger.joblib` sauvegardé mais **non activé** — la bascule
reste un acte séparé (module 39).

## M38 — Prefect : quotas avant réentraînement

Objectif : `ensure_model(..., retrain=True)` existe depuis les modules
29-30 mais n'était jusqu'ici jamais piloté par un vrai critère. Quatre
feux verts (feuille de route §6) doivent être réunis avant de dépenser
un cycle d'arbitrage — jamais un déclenchement calendaire aveugle.

| Contrôle | Statut | Preuve |
|---|---|---|
| Refactor `src/indusense/arbitration.py` | Implémenté | Logique du module 37 déplacée du script vers le paquet — `scripts/arbitrate_challenger.py` et `retrain_flow.py` partagent `run_arbitration()`/`persist_arbitration()` |
| Feu vert 1 — quota validations | Implémenté | `>= 50` validations humaines |
| Feu vert 2 — quota pannes confirmées | Implémenté | `>= 10` pannes confirmées/déclarées |
| Feu vert 3 — qualité des données | Implémenté | `indusense.data_quality` (Pandera), bornes reprises du contrat DAT §5.4 |
| Feu vert 4 — dérive constatée | Implémenté | PSI (`indusense.drift`, module 31) `trainval` vs période proposée, seuil 0.25 |
| Cycle suspendu si feux non verts | Implémenté | Prouvé en conditions réelles (base vide) |
| Cycle déclenché si 4 feux verts | Implémenté | Prouvé en conditions réelles (données du module 36) |
| Non-régression | Implémenté | `uv run pytest -q` → 110 passed (+11) ; `ruff`/`black` OK |

### 2 bugs réels rencontrés et corrigés

1. `check_drift_signal` sur des fixtures de test à variance nulle :
   PSI dégénère (tous les quantiles de bin s'effondrent en un seul
   point) et ne détecte aucun écart, même avec une moyenne très
   différente — pas un bug du code, un piège de conception de test
   (fixture corrigée avec un vrai étalement, `rng.normal`).
2. `_load_reviews` supposait que la table `predictions` existait déjà —
   plante (`OperationalError: no such table`) sur un environnement neuf
   (avant le premier backfill). Corrigé en exécutant `CREATE_TABLE_SQL`
   avant la lecture, même garde que `predictions_store.py`.

### Preuve en conditions réelles — les 4 feux verts, avec les vraies données

```
uv run --frozen python src/indusense/flows/retrain_flow.py
Feu vert '1_quota_validations' : OK -- 982 validations humaines (seuil 50)
Feu vert '2_quota_pannes_confirmees' : OK -- 727 pannes confirmées/déclarées (seuil 10)
Feu vert '3_qualite_donnees' : OK -- aucune valeur hors bornes
Feu vert '4_derive' : OK -- PSI max temp_mean_24h=0.3841 (seuil 0.25)
4 feux verts réunis -- lancement de l'arbitrage champion/challenger
Arbitrage terminé : décision=REJET_DU_MODELE_N (gains=9, régressions=7)
```

PSI réel de 0,38 entre `trainval` (juin 2025-mars 2026) et avril-mai
2026 sur `temp_mean_24h` — dérive saisonnière authentique, pas
fabriquée pour la démo. Décision d'arbitrage identique au module 37
(même logique, appelée derrière les feux verts plutôt qu'à la demande).

### Preuve en conditions réelles — cycle suspendu

```
PREDICTIONS_DB_URL="sqlite:///predictions_empty_test.db" \
  uv run --frozen python src/indusense/flows/retrain_flow.py
Feu vert '1_quota_validations' : SUSPENDU -- 0 validations humaines (seuil 50)
Feu vert '2_quota_pannes_confirmees' : SUSPENDU -- 0 pannes confirmées/déclarées (seuil 10)
Feu vert '3_qualite_donnees' : OK -- aucune valeur hors bornes
Feu vert '4_derive' : OK -- PSI max temp_mean_24h=0.3841 (seuil 0.25)
Cycle suspendu : feux non verts -> ['1_quota_validations', '2_quota_pannes_confirmees']
```

Base réellement vide (pas simulée en mémoire) — le cycle s'arrête avant
de dépenser un entraînement, exactement comme prévu.

## M39 — Mode fantôme avant bascule

Objectif : dernier filet avant la vraie production — le challenger
accepté score en silence à côté du champion, et la bascule
(`model.joblib`) ne s'exécute que si l'observation confirme, zéro
tolérance à la régression.

| Contrôle | Statut | Preuve |
|---|---|---|
| Journal séparé du fantôme | Implémenté | `reports/hitl/shadow_log.csv` — jamais mélangé à `predictions` (contrat : ce qui a été réellement servi) |
| Zéro tolérance à la régression | Implémenté | `shadow_confirms()` — plus strict que l'arbitrage (qui tolère 1 régression sous dérogation) |
| Bascule réelle et réversible | Implémenté | `promote_challenger()` — sauvegarde horodatée avant tout écrasement |
| Cycle honnête sur l'arbitrage réel | Implémenté | Notre seul arbitrage réel (modules 37-38) a conclu `REJET_DU_MODELE_N` → le mode fantôme le respecte, `PAS_ELIGIBLE`, rien n'est promu |
| Chemin mécanique de promotion prouvé séparément | Implémenté | `test_shadow_cycle_promotes_a_genuinely_accepted_challenger` — arbitrage accepté fabriqué pour l'occasion, bascule de fichier réelle vérifiée |
| Non-régression | Implémenté | `uv run pytest -q` → 119 passed (+9) ; `ruff`/`black` OK |

### Bug réel trouvé et corrigé

`promote_challenger(challenger_path: Path = DEFAULT_CHALLENGER_PATH, ...)`
— une valeur par défaut de signature est figée à la **définition** de
la fonction, pas à son appel. Un test qui `monkeypatch` le module
`DEFAULT_CHALLENGER_PATH` n'aurait donc aucun effet sur cette valeur par
défaut déjà capturée. Corrigé en résolvant `challenger_path` à
l'intérieur du corps de la fonction (même pattern déjà utilisé pour
`model_path`), pas dans la signature.

### Décision de conception : pourquoi pas la même table `predictions`

Le mode fantôme score deux modèles sur la même fenêtre, mais un seul
(le champion) est réellement vu par un technicien. Écrire les deux dans
`predictions` casserait le contrat implicite de cette table (ce qui a
été réellement servi, réellement examinable en Streamlit) — d'où un
journal séparé (`shadow_log.csv`), pas une extension de la clé primaire
comme envisagé dans le plan initial.

### Preuve en conditions réelles — le vrai pipeline, honnête jusqu'au bout

```
uv run --frozen python scripts/run_shadow_mode.py
Statut : PAS_ELIGIBLE
dernière décision d'arbitrage = REJET_DU_MODELE_N, le mode fantôme
n'observe que les challengers acceptés
```

`model.joblib` vérifié inchangé après coup (aucun `.backup-*` créé) —
le pipeline M35→M39 a correctement refusé de toucher au modèle de
production à chaque étape où c'était justifié : arbitrage (7
régressions pour 9 gains) puis, par construction, mode fantôme. C'est
le résultat honnête de ce projet, pas un aboutissement retouché pour la
démo.

### Preuve du chemin mécanique — arbitrage accepté fabriqué pour l'occasion

```
uv run --frozen python -m pytest tests/test_shadow.py -m requires_local_infra -v
test_shadow_cycle_promotes_a_genuinely_accepted_challenger PASSED
```

Arbitrage `ACCEPTATION_DIRECTE` écrit dans un journal temporaire,
`run_shadow_window` remplacé par une matrice fixe (gains=5,
régressions=0) pour isoler le câblage de la performance ML — déjà
prouvée ailleurs. Vérifié après coup : le fichier de sauvegarde contient
l'ancien modèle, `model.joblib` contient le nouveau, `shadow_log.csv`
écrit. Le mécanisme de bascule fonctionne réellement ; c'est
l'arbitrage réel de nos données qui a décidé de ne pas l'utiliser.

## Bilan M35-M39

Boucle HITL complète, chaque étape prouvée en conditions réelles sur
les vraies données InduSense (pas le jeu pédagogique de la feuille de
route) : journal versionné → simulateur + Streamlit réel → arbitrage
champion/challenger → 4 feux verts Prefect → mode fantôme. Résultat de
bout en bout honnête : le modèle de production n'a jamais été touché,
parce que rien dans les vraies données ne le justifiait encore — la
boucle a fonctionné comme un garde-fou, pas comme une chaîne de
validation automatique.
