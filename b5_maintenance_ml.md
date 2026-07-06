# B5 — Machine Learning tabulaire : prédire la panne (Maintenance prédictive)

> Objectif : construire un **premier modèle** de prédiction de panne à partir du **Gold dataset**
> indusense (`data/raw/indusense_gold.parquet`). Modules suivants : optimisation (B7), documentation (B8).

## Scénario

Le Gold dataset donne, pour chaque machine et chaque heure, des indicateurs capteurs et un label
« panne dans les prochaines heures ? ». Question : **peut-on prédire une panne à l'avance ?**
→ Problème de **classification binaire supervisée** sur données tabulaires, avec deux pièges à éviter :
la **fuite temporelle** et le **déséquilibre** (pannes rares).

## Grandes étapes

### 1. Préparer les données
- Charger le parquet, trié par machine puis par temps.
- **Exclure les colonnes de fuite** , les identifiants et les labels des features.
- Choisir un **horizon** (ex. `label_failure_next_24h`) et construire la cible `y` (0/1).
- Utiliser le **split temporel fourni** (`split_set` : train < validation < test) — **pas** de split aléatoire.
- Imputer les `NaN` (médiane) ; standardiser pour les modèles linéaires.

### 2. Gérer le déséquilibre
- Vérifier le taux de panne (rare) → **ne pas utiliser l'accuracy**.
- Compenser : `class_weight="balanced"` (scikit-learn) et `scale_pos_weight` (XGBoost, calculé sur le train).

### 3. Entraîner trois modèles (du plus simple au plus expressif)
- **Régression logistique** — baseline linéaire (Pipeline : imputation → standardisation → modèle).
- **Random Forest** — non linéaire, robuste.
- **XGBoost** — gradient boosting, candidat principal (HP sobres, à optimiser en B7).

### 4. Évaluer et comparer
- Métriques : Choisir une métrique adaptée au problème
- **Validation croisée temporelle** (`TimeSeriesSplit`) pour une estimation robuste.
- Choisir un **seuil de décision** et lire la **matrice de confusion** (faux négatifs vs faux positifs).

### 5. Suivre et sélectionner
- Journaliser paramètres et métriques dans **MLflow** (backend SQLite local).
- Produire un **tableau comparatif** trié par PR-AUC et retenir le **meilleur modèle** → passe en B7.

## Livrables

- Notebook exécuté (`01_maintenance_ml.ipynb`) couvrant les étapes 1 à 5.
- Trois modèles entraînés + tableau comparatif (PR-AUC / ROC-AUC).
- Matrice de confusion commentée + runs MLflow.
- Conclusion : quel modèle passe au module B7, et pourquoi.

```

## Ressources

- [scikit-learn — Pipeline & modèles](https://scikit-learn.org/stable/modules/classes.html)
- [XGBoost — paramètres](https://xgboost.readthedocs.io/en/stable/parameter.html)
- [PR-AUC (average_precision_score)](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.average_precision_score.html)
- [TimeSeriesSplit](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.TimeSeriesSplit.html)
- [MLflow — tracking](https://mlflow.org/docs/latest/tracking.html)
