# InduSense 4.0 — maintenance prédictive

InduSense prédit le risque de panne d'une presse industrielle dans les 24 h,
à partir de la télémétrie machine et de l'historique d'incidents. Le modèle
(XGBoost) est servi par une API FastAPI, surveillé par Prometheus et Grafana,
et amélioré par une boucle de revue humaine (HITL) : un technicien confirme
ou infirme chaque alerte, et ces verdicts nourrissent le modèle suivant.

Ce qu'il faut savoir du modèle avant d'utiliser son score : il s'appuie
surtout sur les incidents déjà déclarés (un incident de sévérité 3 dans les
24 h précède 93 % des pannes), bien plus que sur les capteurs. Détails dans
la [model card](artifacts/model_card.md).

## Démarrer

```bash
make stack-up        # base, API, Prometheus, Grafana (base initialisée au 1er démarrage)
make stack-etl       # reconstruit Silver + Gold
make model-card      # certifie le modèle servi, puis régénère sa model card
```

Adresses et identifiants des interfaces : [acces_local.md](docs/05_exploitation/acces_local.md).

## Carte de la documentation

| Thème | À lire en premier | Tout le thème |
|---|---|---|
| **Cadrage** — pourquoi le projet, ce qu'il doit livrer | [PROJECT.md](docs/01_cadrage/PROJECT.md) | [cadrage_roi_indusense.md](docs/01_cadrage/cadrage_roi_indusense.md), [feuille_de_route_mlops_indusense.md](docs/01_cadrage/feuille_de_route_mlops_indusense.md), [gap_analysis_dat_v1.md](docs/01_cadrage/gap_analysis_dat_v1.md), [suivi_projet_ia.md](docs/01_cadrage/suivi_projet_ia.md) |
| **Architecture** — comment c'est construit | [architecture_deploiement_indusense.md](docs/02_architecture/architecture_deploiement_indusense.md) | [cadrage_ruptures_package.md](docs/02_architecture/cadrage_ruptures_package.md), [cadrage_operateurs_ingest.md](docs/02_architecture/cadrage_operateurs_ingest.md) |
| **Données** — Bronze, Silver, Gold | [modele_relationnel_ingestion.md](docs/03_donnees/modele_relationnel_ingestion.md) | [BRONZE.md](docs/03_donnees/BRONZE.md), [gold_roadmap.md](docs/03_donnees/gold_roadmap.md), [ANALYSE_ARTIFACTS.md](pipeline_artifacts/ANALYSE_ARTIFACTS.md) |
| **Modèle** — choix et méthodes | [model_card.md](artifacts/model_card.md) | [b5_maintenance_ml.md](docs/04_modele/b5_maintenance_ml.md), [groupkfold_ml.md](docs/04_modele/groupkfold_ml.md), [optuna_choix_methode.md](docs/04_modele/optuna_choix_methode.md), [optuna_groupkfold_ml.md](docs/04_modele/optuna_groupkfold_ml.md), [XGBOOST_HYPERPARAMETRES.md](docs/04_modele/XGBOOST_HYPERPARAMETRES.md), [early_stopping_ml.md](docs/04_modele/early_stopping_ml.md), [normalization_machine_ml.md](docs/04_modele/normalization_machine_ml.md), [zscore_machine_ml.md](docs/04_modele/zscore_machine_ml.md), [vigilance_ml.md](docs/04_modele/vigilance_ml.md) |
| **Exploitation** — faire tourner et surveiller | [runbook.md](docs/05_exploitation/runbook.md) | [guide_experimentation_indusense.md](docs/05_exploitation/guide_experimentation_indusense.md), [plan_action_hitl_champion_challenger.md](docs/05_exploitation/plan_action_hitl_champion_challenger.md), [drift_spec.md](reports/drift/drift_spec.md), [acces_local.md](docs/05_exploitation/acces_local.md) |
| **Sécurité** — menaces et contrôles | [threat_model.md](docs/06_securite/threat_model.md) | [security_controls.md](docs/06_securite/security_controls.md) |
| **Preuves** — ce qui a été démontré | [cahier_de_tests_indusense.md](docs/07_preuves/cahier_de_tests_indusense.md) | [pipeline_proof.md](docs/07_preuves/pipeline_proof.md), [image_proof.md](docs/07_preuves/image_proof.md), [compose_proof.md](docs/07_preuves/compose_proof.md), [drift_proof.md](docs/07_preuves/drift_proof.md), [perf_proof.md](docs/07_preuves/perf_proof.md), [hitl_proof.md](docs/07_preuves/hitl_proof.md) |

## Documents de référence

- [Model card](artifacts/model_card.md) — ce que fait le modèle servi, ses limites, sa certification. Générée par `generate_model_card.py`, ne pas la modifier à la main.
- [Base de connaissance](src/indusense/knowledge/base_connaissance.yaml) — ce que le métier attend du modèle ; sert à expliquer les scores et à certifier le modèle.
- [Runbook](docs/05_exploitation/runbook.md) — que faire sur une alerte de dérive, et après tout changement de modèle.
- [Cahier de tests](docs/07_preuves/cahier_de_tests_indusense.md) — chaque exigence, sa commande de vérification, sa preuve.

## Livrables pédagogiques

Les présentations (`.pptx`) sont produites localement dans le dossier
`Livrables/`, volontairement non versionné (fichiers binaires régénérés
souvent) : elles ne sont donc pas dans le dépôt.
