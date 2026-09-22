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

## M36-M39 — pas commencés
