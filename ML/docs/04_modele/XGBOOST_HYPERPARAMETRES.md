# Hyperparamètres XGBoost — TP7 (baseline)

| Paramètre | Valeur | Rôle |
|---|---|---|
| `n_estimators` | 300 | Nombre d'arbres séquentiels |
| `max_depth` | 6 | Profondeur maximale de chaque arbre |
| `learning_rate` | 0.1 | Pas d'apprentissage (shrinkage) |
| `subsample` | 0.8 | Fraction des observations utilisées par arbre |
| `colsample_bytree` | 0.8 | Fraction des features utilisées par arbre |
| `scale_pos_weight` | neg / pos | Compensation du déséquilibre de classes |

## Résultats obtenus

| Métrique | Train | Validation |
|---|---|---|
| PR-AUC | 1.0000 | **0.8174** |
| ROC-AUC | 1.0000 | 0.9923 |
| F1 | 0.9996 | 0.7351 |

- **Recall** : 87.9 % (639 / 727 pannes détectées)
- **Précision** : 63.6 % (366 fausses alertes)

## Axes d'optimisation — B7

Le fort overfitting (PR-AUC train = 1.0, Δ = 0.18) s'explique par la sur-dépendance
à `incident_max_severity_prev_24h` (63 % de l'importance du modèle).

Leviers prioritaires :

```python
max_depth        : 6  →  3 ou 4         # réduire la complexité des arbres
min_child_weight : 1  →  5 ou 10        # noeuds nécessitent plus d'exemples
reg_alpha        : 0  →  0.1 à 1.0      # régularisation L1
reg_lambda       : 1  →  5 à 10         # régularisation L2
n_estimators     : 300 → early stopping # arrêt optimal automatique
```
