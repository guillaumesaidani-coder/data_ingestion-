---
library_name: xgboost
license: other
tags:
- tabular-classification
- predictive-maintenance
- xgboost
- manufacturing
- time-series-features
model-index:
- name: indusense-xgb-maintenance-b11-gkf
  results:
  - task:
      type: binary-classification
    dataset:
      name: Gold machine hourly features — indusense_db
      type: indusense-gold-machine-hourly
    metrics:
    - type: pr_auc
      value: 0.8799
      name: PR-AUC (average precision, test chronologique)
---

# Model Card for indusense-xgb-maintenance-b11-gkf

<!-- Provide a quick summary of what the model is/does. -->

Classifieur XGBoost prédisant le risque de panne d'une machine industrielle dans les 24h à venir, à partir de features télémétrie (température, pression, tension, vitesse de rotation) et d'historique d'incidents agrégés sur des fenêtres glissantes 6h/12h/24h.

## Model Details

### Model Description

<!-- Provide a longer summary of what this model is. -->

Entraîné sur 112,996 observations horaires (15 machines, 69 features), évalué en validation croisée GroupKFold (une machine exclue par fold, pour ne jamais évaluer sur une machine vue à l'entraînement) et sur un jeu de test chronologiquement postérieur. Hyperparamètres optimisés par Optuna (TPE, 30 essais) sur l'objectif GroupKFold — voir `TP11.ipynb`.

- **Developed by:** Guillaume Saïdani
- **Funded by [optional]:** [More Information Needed]
- **Shared by [optional]:** [More Information Needed]
- **Model type:** XGBoost (gradient boosting), classification binaire, pipeline scikit-learn (imputation médiane + XGBClassifier)
- **Language(s) (NLP):** n/a (données tabulaires, pas de NLP)
- **License:** Usage interne — données propriétaires (télémétrie machine, non publiques)
- **Finetuned from model [optional]:** Aucun — entraîné from scratch

### Model Sources [optional]

<!-- Provide the basic links for the model. -->

- **Repository:** ML/ (ce dépôt) — TP11.ipynb (recherche HP) ; artefact persisté : MLflow `runs:/fa336bc0880b48bc849a4a6b1e6f412d/xgboost_b11_gkf` (store `ML/mlflow_tp7.db`, voir DL/TP9.ipynb)
- **Paper [optional]:** [More Information Needed]
- **Demo [optional]:** [More Information Needed]

## Uses

<!-- Address questions around how the model is intended to be used, including the foreseeable users of the model and those affected by the model. -->

### Direct Use

<!-- This section is for the model use without fine-tuning or plugging into a larger ecosystem/app. -->

Scorer un enregistrement horaire machine (features télémétrie + historique incidents agrégé) et obtenir une probabilité de panne à 24h. Seuil de décision par défaut 0.5 (celui évalué ici) — à ajuster selon l'arbitrage rappel/précision métier avant tout déploiement (cf. TP8 §8 pour une démarche de calibration de seuil sur une version antérieure).

### Downstream Use [optional]

<!-- This section is for the model use when fine-tuned for a task, or when plugged into a larger ecosystem/app -->

Alimentation d'un tableau de bord de maintenance prédictive classant les machines par risque décroissant, à l'usage d'un planificateur de maintenance humain — pas d'arrêt automatique de machine.

### Out-of-Scope Use

<!-- This section addresses misuse, malicious use, and uses that the model will not work well for. -->

- Arrêt automatique ou décision de maintenance sans validation humaine.
- Toute machine hors des 15 machines couvertes par l'entraînement, sans ré-entraînement (les performances varient déjà fortement d'une machine connue à l'autre, cf. limites).
- Interprétation de `predict_proba` comme une probabilité calibrée — aucune calibration (Platt/isotonic) n'a été appliquée.
- Usage réglementaire ou de certification sécurité — aucune validation de ce type.

## Bias, Risks, and Limitations

<!-- This section is meant to convey both technical and sociotechnical limitations. -->

- **Performance très hétérogène par machine** : PR-AUC en validation croisée GroupKFold va de 0.39 (MACH-07) à 1.00 (MACH-11, MACH-15) — écart-type ±0.19 autour d'une moyenne de 0.78. Un score agrégé unique masque des machines où le modèle est nettement moins fiable.
- **Sur-ajustement structurel** : PR-AUC train = 1.000 contre ~0.78 en CV — piloté à 63% par une seule feature (`incident_max_severity_prev_24h`, diagnostic TP9/TP10). Persiste malgré une régularisation poussée (`reg_lambda`, `min_child_weight` élevés) ; qualifié de structurel, pas résolu par les hyperparamètres seuls.
- **Historique de fuite de données** : une version antérieure (B7, TP8) incluait `feature_row_id`, un identifiant séquentiel corrélé à l'ordre temporel, qui gonflait le PR-AUC de +0.111. Corrigé depuis TP8b — retiré explicitement des colonnes de fuite — mais signale la fragilité du pipeline de features aux fuites indirectes.
- **Rappel modéré au seuil par défaut** : 87.8% de rappel, 89 pannes non détectées sur 727 au seuil 0.5 — un seuil plus bas augmenterait le rappel au prix de plus de fausses alertes (arbitrage non refait ici pour B11).
- **Tentative de normalisation par machine infructueuse** : une normalisation z-score par machine a dégradé la généralisation en GroupKFold (TP10) — confirme que XGBoost est déjà insensible à l'échelle des features, ne pas réintroduire cette étape.

### Recommendations

<!-- This section is meant to convey recommendations with respect to the bias, risk, and technical limitations. -->

Ne jamais utiliser en décision automatique. Suivre la performance par machine individuellement, pas seulement l'agrégat — une machine comme MACH-07 justifie une vigilance humaine renforcée plutôt qu'une confiance dans le score. Calibrer les probabilités (Platt/isotonic) avant tout usage nécessitant un score interprétable comme une probabilité réelle. Recalibrer le seuil de décision selon le coût métier faux négatif vs faux positif avant déploiement.

## How to Get Started with the Model

Use the code below to get started with the model.

```python
# Artefact persisté dans MLflow (voir DL/TP9.ipynb) — rechargement direct
import mlflow.xgboost
mlflow.set_tracking_uri('sqlite:///mlflow_tp7.db')
model = mlflow.xgboost.load_model('runs:/fa336bc0880b48bc849a4a6b1e6f412d/xgboost_b11_gkf')

from sklearn.impute import SimpleImputer
X_imputed = SimpleImputer(strategy='median').fit(X_train_val).transform(X_new)
proba = model.predict_proba(X_imputed)[:, 1]  # risque de panne à 24h
```

## Training Details

### Training Data

<!-- This should link to a Dataset Card, perhaps with a short stub of information on what the training data is all about as well as documentation related to data pre-processing or additional filtering. -->

Table `gold_machine_hourly_feature` (PostgreSQL, base indusense_db) — 112,996 observations horaires (entraînement + validation), 15 machines, 69 features (rolling 6h/12h/24h télémétrie + historique incidents agrégé 24h/7j). Split chronologique (quantiles 0.70/0.85 sur `window_start`), pas de KFold aléatoire (fuite temporelle sinon).

### Training Procedure

<!-- This relates heavily to the Technical Specifications. Content here should link to that section when it is relevant to the training procedure. -->

#### Preprocessing [optional]

Imputation des valeurs manquantes par médiane (`SimpleImputer`), aucune normalisation (XGBoost invariant à l'échelle — confirmé empiriquement, TP12).


#### Training Hyperparameters

- **Training regime:** fp32, XGBoost gradient boosting, hyperparamètres Optuna (TPE, 30 essais, objectif GroupKFold(5)), scale_pos_weight=27.12 (déséquilibre de classes) <!--fp32, fp16 mixed precision, bf16 mixed precision, bf16 non-mixed precision, fp16 non-mixed precision, fp8 mixed precision -->

#### Speeds, Sizes, Times [optional]

<!-- This section provides information about throughput, start/end time, checkpoint size if relevant, etc. -->

4.7s pour un ré-entraînement complet sur 112,996 lignes (poste de travail local, CPU)

## Evaluation

<!-- This section describes the evaluation protocols and provides the results. -->

### Testing Data, Factors & Metrics

#### Testing Data

<!-- This should link to a Dataset Card if possible. -->

Même table, partition test chronologiquement postérieure — 19,944 lignes, 727 pannes positives.

#### Factors

<!-- These are the things the evaluation is disaggregating by, e.g., subpopulations or domains. -->

Évaluation agrégée toutes machines confondues pour la métrique principale ; performance par machine disponible via validation croisée GroupKFold (hétérogénéité 0.39-1.00).

#### Metrics

<!-- These are the evaluation metrics being used, ideally with a description of why. -->

PR-AUC (average precision — préférée à l'accuracy vu le déséquilibre de classe), ROC-AUC, F1, matrice de confusion au seuil 0.5.

### Results

PR-AUC train=0.9997 · PR-AUC test=0.8799 · ROC-AUC test=0.9949 · F1 test=0.7586 · TP=638 TN=18900 FP=317 FN=89 · Précision=66.8% · Rappel=87.8%

#### Summary

PR-AUC test 0.88 sur un problème fortement déséquilibré (scale_pos_weight=27) — signal réel mais hétérogène selon les machines (voir limites). Écart train/CV important : le modèle généralise moins bien qu'il ne le suggère sur ses propres données d'entraînement.

## Model Examination [optional]

<!-- Relevant interpretability work for the model goes here -->

Non réalisé — SHAP (TreeExplainer) recommandé en prochaine étape pour identifier les features dominantes par machine (cf. b7_optimisation_explicabilite.md, hors périmètre ici).

## Environmental Impact

<!-- Total emissions (in grams of CO2eq) and additional considerations, such as electricity usage, go here. Edit the suggested text below accordingly -->

Carbon emissions can be estimated using the [Machine Learning Impact calculator](https://mlco2.github.io/impact#compute) presented in [Lacoste et al. (2019)](https://arxiv.org/abs/1910.09700).

- **Hardware Type:** Intel Core i7-12700H (CPU) — entraînement XGBoost, pas de GPU requis
- **Hours used:** 0.0013 h (ré-entraînement complet mesuré ci-dessus)
- **Cloud Provider:** Aucun — poste de travail local
- **Compute Region:** France (FRA)
- **Carbon Emitted:** 0.0025 gCO2eq (0.045 Wh) — mesure CodeCarbon de ce ré-entraînement

## Technical Specifications [optional]

### Model Architecture and Objective

XGBoost : n_estimators=287, max_depth=9, learning_rate=0.0289, objectif binaire pondéré (scale_pos_weight=27.12)

### Compute Infrastructure

Poste de travail local, connexion PostgreSQL directe. Artefact tracké et versionné dans MLflow (voir DL/TP9.ipynb) : runs:/fa336bc0880b48bc849a4a6b1e6f412d/xgboost_b11_gkf.

#### Hardware

CPU suffisant — pas de dépendance GPU

#### Software

xgboost 3.3.0, scikit-learn 1.9.0, optuna 4.9.0, mlflow 3.14.0, Python 3.13 (ML/.venv)

## Citation [optional]

<!-- If there is a paper or blog post introducing the model, the APA and Bibtex information for that should go in this section. -->

**BibTeX:**

[More Information Needed]

**APA:**

[More Information Needed]

## Glossary [optional]

<!-- If relevant, include terms and calculations in this section that can help readers understand the model or model card. -->

[More Information Needed]

## More Information [optional]

[More Information Needed]

## Model Card Authors [optional]

Guillaume Saïdani

## Model Card Contact

guillaume.saidani@ext.aelion.fr