---
library_name: xgboost
license: other
tags:
- tabular-classification
- predictive-maintenance
- xgboost
- manufacturing
- time-series-features
model_version: ba0728eb82ac
certification: conforme
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
      value: 0.8945
      name: PR-AUC (average precision, test chronologique)
---

# Model Card for indusense-xgb-maintenance-b11-gkf

<!-- Provide a quick summary of what the model is/does. -->

Classifieur XGBoost qui estime le risque de panne (incident de sévérité ≥ 4) d'une machine dans les 24 h à venir. En pratique, il s'appuie surtout sur un précurseur : un incident de sévérité 3 déclaré dans les 24 h précédentes, qui précède 93% des pannes de l'entraînement. L'historique d'incidents porte 77% de sa décision ; la télémétrie pèse peu. Modèle servi : `ba0728eb82ac`.

## Model Details

### Model Description

<!-- Provide a longer summary of what this model is. -->

Fichier servi : `artifacts/models/model.joblib` (versionné par DVC), `model_version` `ba0728eb82ac` = SHA-256 tronqué du fichier, le même que l'API, les prédictions et la certification. Métriques mesurées en scorant ce fichier, sans ré-entraînement.

Entraîné sur 112,996 observations horaires (15 machines, 78 features). Cible : `label_failure_next_24h`, vraie si un incident de sévérité ≥ 4 survient sur la machine dans les 24 h suivantes. Taux de panne observé à l'entraînement : 35.6% après un incident de sévérité 3 dans les 24 h, 0.0% sans aucun incident. Recette évaluée en validation croisée GroupKFold (une machine exclue par fold) ; hyperparamètres optimisés par Optuna (TPE, 30 essais) — voir `TP11.ipynb`.

- **Developed by:** Guillaume Saïdani
- **Funded by [optional]:** [More Information Needed]
- **Shared by [optional]:** [More Information Needed]
- **Model type:** XGBoost (gradient boosting), classification binaire, pipeline scikit-learn (imputation médiane + XGBClassifier)
- **Language(s) (NLP):** n/a (données tabulaires, pas de NLP)
- **License:** Usage interne — données propriétaires (télémétrie machine, non publiques)
- **Finetuned from model [optional]:** Aucun — entraîné from scratch

### Model Sources [optional]

<!-- Provide the basic links for the model. -->

- **Repository:** ML/ (ce dépôt) — `indusense train -o artifacts/models/model.joblib` ; recherche d'hyperparamètres : TP11.ipynb ; certification : `scripts/certify_model.py`.
- **Paper [optional]:** [More Information Needed]
- **Demo [optional]:** [More Information Needed]

## Uses

<!-- Address questions around how the model is intended to be used, including the foreseeable users of the model and those affected by the model. -->

### Direct Use

<!-- This section is for the model use without fine-tuning or plugging into a larger ecosystem/app. -->

Scorer un enregistrement horaire machine et obtenir un risque de panne à 24 h. Le score monte surtout après un incident de sévérité 3 déclaré : une panne sans ce précurseur est rarement détectée. Seuil de décision par défaut 0.301 (probabilité calibrée, équivalent exact du score brut 0.5) (celui évalué ici) — à ajuster selon l'arbitrage rappel/précision métier.

### Downstream Use [optional]

<!-- This section is for the model use when fine-tuned for a task, or when plugged into a larger ecosystem/app -->

Alimentation d'un tableau de bord de maintenance prédictive classant les machines par risque décroissant, à l'usage d'un planificateur de maintenance humain — pas d'arrêt automatique de machine.

### Out-of-Scope Use

<!-- This section addresses misuse, malicious use, and uses that the model will not work well for. -->

- Arrêt automatique ou décision de maintenance sans validation humaine.
- Toute machine d'un type de presse absent de l'entraînement, sans ré-entraînement : PR-AUC moyenne 0.87 quand seule la machine est inconnue, 0.81 quand tout son type l'est (cf. limites). Une nouvelle machine d'un type déjà couvert reste dans le périmètre.
- Détection d'une dégradation visible seulement dans les capteurs : le modèle réagit surtout aux incidents déclarés.
- Usage réglementaire ou de certification sécurité — aucune validation de ce type.

## Bias, Risks, and Limitations

<!-- This section is meant to convey both technical and sociotechnical limitations. -->

- **Dépendance à la saisie des incidents** : 93% des pannes de l'entraînement suivent un incident de sévérité 3 dans les 24 h. Un incident non saisi, ou saisi avec une autre sévérité, fait chuter le risque ; les pannes sans ce précurseur passent le plus souvent inaperçues.
- **Performance hétérogène par machine** : PR-AUC en validation croisée GroupKFold (une machine par fold) de 0.60 (MACH-05) à 1.00 (MACH-15) — écart-type ±0.11 autour d'une moyenne de 0.87. Un score agrégé unique masque des machines où le modèle est nettement moins fiable.
- **Types de presse différents, invisibles pour le modèle** : ni `machine_id` ni le type (`machine.model`) ne sont des entrées. Moyenne par type (une machine cachée) : InduPress-X1 0.85 · InduPress-X2 0.88 · InduPress-X3 0.87 · InduPress-Z1 0.84. Quand tout un type est caché à l'entraînement, la PR-AUC moyenne passe de 0.87 à 0.81 (pire cas MACH-11 : −0.23).
- **Sur-ajustement structurel** : PR-AUC train = 1.000 contre 0.87 en CV — `incident_max_severity_prev_24h` porte à elle seule 32% de l'explication (SHAP, certification). Persiste malgré une régularisation poussée (`reg_lambda`, `min_child_weight` élevés) ; qualifié de structurel, pas résolu par les hyperparamètres seuls.
- **Historique de fuite de données** : une version antérieure (B7, TP8) incluait `feature_row_id`, un identifiant séquentiel corrélé à l'ordre temporel, qui gonflait le PR-AUC de +0.111. Corrigé depuis TP8b — retiré explicitement des colonnes de fuite — mais signale la fragilité du pipeline de features aux fuites indirectes.
- **Rappel au seuil par défaut** : 91.1% de rappel, 65 pannes non détectées sur 727 au seuil 0.301 (probabilité calibrée, équivalent exact du score brut 0.5) — un seuil plus bas augmenterait le rappel au prix de plus de fausses alertes.
- **Tentative de normalisation par machine infructueuse** : une normalisation z-score par machine a dégradé la généralisation en GroupKFold (TP10) — confirme que XGBoost est déjà insensible à l'échelle des features, ne pas réintroduire cette étape.

### Recommendations

<!-- This section is meant to convey recommendations with respect to the bias, risk, and technical limitations. -->

Ne jamais utiliser en décision automatique. Veiller à la saisie des incidents et de leur sévérité : c'est la première entrée du modèle. Suivre la performance par machine et par type de presse, pas seulement l'agrégat — une machine comme MACH-05 justifie une vigilance humaine renforcée plutôt qu'une confiance dans le score. Ré-entraîner avant de scorer un nouveau type de presse. Probabilités calibrées (Platt sur la marge, apprise en GroupKFold par machine) : revérifier la calibration sur le test après chaque réentraînement. Comparer le modèle à la règle « sévérité max 24 h = 3 » : il ne vaut que s'il fait mieux qu'elle.

## How to Get Started with the Model

Use the code below to get started with the model.

```python
# Le fichier servi par l'API (dvc pull pour le récupérer)
import joblib
model = joblib.load('artifacts/models/model.joblib')  # imputer + XGBClassifier
proba = model.predict_proba(X_new)[:, 1]  # risque de panne à 24h
alerte = proba >= model.threshold_  # seuil calibré porté par le modèle
```

## Training Details

### Training Data

<!-- This should link to a Dataset Card, perhaps with a short stub of information on what the training data is all about as well as documentation related to data pre-processing or additional filtering. -->

Table `gold_machine_hourly_feature` (PostgreSQL, base indusense_db) — 112,996 observations horaires (entraînement + validation), 15 machines, 78 features (rolling 6h/12h/24h télémétrie + historique incidents agrégé 24h/7j). Split chronologique (quantiles 0.70/0.85 sur `window_start`), pas de KFold aléatoire (fuite temporelle sinon).

### Training Procedure

<!-- This relates heavily to the Technical Specifications. Content here should link to that section when it is relevant to the training procedure. -->

#### Preprocessing [optional]

Imputation des valeurs manquantes par médiane (`SimpleImputer`), aucune normalisation (XGBoost invariant à l'échelle — confirmé empiriquement, TP12).


#### Training Hyperparameters

- **Training regime:** fp32, XGBoost gradient boosting, hyperparamètres Optuna (TPE, 30 essais, objectif GroupKFold(5)), scale_pos_weight=27.12 (déséquilibre de classes) <!--fp32, fp16 mixed precision, bf16 mixed precision, bf16 non-mixed precision, fp16 non-mixed precision, fp8 mixed precision -->

#### Speeds, Sizes, Times [optional]

<!-- This section provides information about throughput, start/end time, checkpoint size if relevant, etc. -->

Non mesuré pour ce modèle (voir Carbon Emitted)

## Evaluation

<!-- This section describes the evaluation protocols and provides the results. -->

### Testing Data, Factors & Metrics

#### Testing Data

<!-- This should link to a Dataset Card if possible. -->

Même table, partition test chronologiquement postérieure — 19,944 lignes, 727 pannes positives.

#### Factors

<!-- These are the things the evaluation is disaggregating by, e.g., subpopulations or domains. -->

Évaluation agrégée toutes machines confondues pour la métrique principale ; performance par machine disponible via validation croisée GroupKFold (hétérogénéité 0.60-1.00) et par type de presse (voir limites).

#### Metrics

<!-- These are the evaluation metrics being used, ideally with a description of why. -->

PR-AUC (average precision — préférée à l'accuracy vu le déséquilibre de classe), ROC-AUC, F1, Brier score, matrice de confusion au seuil 0.301 (probabilité calibrée, équivalent exact du score brut 0.5).

### Results

Modèle `ba0728eb82ac` — PR-AUC train=0.9998 · PR-AUC test=0.8945 · ROC-AUC test=0.9957 · F1 test=0.8054 · Brier test=0.0099 · TP=662 TN=18962 FP=255 FN=65 · Précision=72.2% · Rappel=91.1%

#### Summary

PR-AUC test 0.89 sur un problème fortement déséquilibré (scale_pos_weight=27) — signal réel, porté surtout par l'historique d'incidents, et hétérogène selon les machines (voir limites). Écart train/CV important : le modèle généralise moins bien qu'il ne le suggère sur ses propres données d'entraînement.

## Model Examination [optional]

<!-- Relevant interpretability work for the model goes here -->

Certification du modèle `ba0728eb82ac` (`scripts/certify_model.py`, 50,000 lignes d'entraînement) : **conforme**. Part de l'explication = moyenne des |contributions SHAP| (TreeSHAP exact d'XGBoost), rapportée au total.

- Par famille : Historique d'incidents 77% · Maintenance 7% · Rotation 5% · Tension 4% · Température 3% · Production 2% · Pression 2%.
- Variables les plus lourdes :
  - `incident_max_severity_prev_24h` : 32.2%
  - `incident_count_prev_7d` : 23.6%
  - `incident_count_prev_24h` : 12.3%
  - `hours_since_last_incident` : 7.6%
  - `maintenance_count_prev_30d` : 3.3%
  - `days_since_last_maintenance` : 3.3%
- Constats de la certification :
  - avertissement (informativite, `temp_std_1h`) : Variable vide dans tout l'entraînement : écartée par l'imputer, construite pour rien
  - avertissement (informativite, `pressure_std_1h`) : Variable vide dans tout l'entraînement : écartée par l'imputer, construite pour rien
  - avertissement (concentration_famille) : La famille « incidents » porte 77% de l'explication (max 60%)
  - avertissement (sens, `temp_max_12h`) : Sens attendu : hausse du risque ; appris : baisse (corrélation valeur / contribution -0.60)
  - avertissement (sens, `type_baisse_pression_count_prev_24h`) : Sens attendu : hausse du risque ; appris : baisse (corrélation valeur / contribution -0.14)
  - avertissement (sens, `type_vibration_count_prev_24h`) : Sens attendu : hausse du risque ; appris : baisse (corrélation valeur / contribution -0.17)
  - avertissement (sens, `type_surconsommation_count_prev_24h`) : Sens attendu : hausse du risque ; appris : baisse (corrélation valeur / contribution -0.12)
  - avertissement (sens, `type_alarme_capteur_count_prev_24h`) : Sens attendu : hausse du risque ; appris : baisse (corrélation valeur / contribution -0.11)
  - avertissement (sens, `type_defaut_qualite_count_prev_24h`) : Sens attendu : hausse du risque ; appris : baisse (corrélation valeur / contribution -0.22)

Une contribution décrit ce que le modèle a appris, pas une cause physique.

## Environmental Impact

<!-- Total emissions (in grams of CO2eq) and additional considerations, such as electricity usage, go here. Edit the suggested text below accordingly -->

Carbon emissions can be estimated using the [Machine Learning Impact calculator](https://mlco2.github.io/impact#compute) presented in [Lacoste et al. (2019)](https://arxiv.org/abs/1910.09700).

- **Hardware Type:** Intel Core i7-12700H (CPU) — entraînement XGBoost, pas de GPU requis
- **Hours used:** Non mesuré pour ce modèle (voir Carbon Emitted)
- **Cloud Provider:** Aucun — poste de travail local
- **Compute Region:** Non mesuré pour ce modèle (voir Carbon Emitted)
- **Carbon Emitted:** Non mesuré pour `ba0728eb82ac` : la mesure CodeCarbon est prise par `indusense train` (fichier `training_emissions.json`), absente pour ce modèle entraîné avant cet ajout.

## Technical Specifications [optional]

### Model Architecture and Objective

XGBoost : n_estimators=287, max_depth=9, learning_rate=0.0289, objectif binaire pondéré (scale_pos_weight=27.12)

### Compute Infrastructure

Poste de travail local, connexion PostgreSQL directe. Modèle versionné par DVC (`artifacts/models/model.joblib.dvc`), identifié par `model_version` `ba0728eb82ac`.

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