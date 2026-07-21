# Optuna GroupKFold — Tuning ciblant la généralisation machine

> Re-tuner les hyperparamètres avec GroupKFold comme objectif Optuna corrige le biais introduit par B8 : les params B8 ont été optimisés sur un critère biaisé (split temporel, mêmes machines en train et val).

---

## 1. Le problème : hyperparamètres biaisés par le critère d'optimisation

Dans TP8b, Optuna optimisait le PR-AUC sur le **split temporel** val — les mêmes machines étaient dans train et val. Les params trouvés (max_depth=8, reg_lambda≈0) favorisent la mémorisation des signatures machine plutôt que la généralisation.

```
B8 Optuna objectif : PR-AUC val (split temporel)
  → optimise pour "bien prédire sur les mêmes machines qu'en train"
  → favorise la complexité (max_depth=8, reg_lambda très faible)

B11 Optuna objectif : PR-AUC moyen GroupKFold(5)
  → optimise pour "bien prédire sur une machine jamais vue"
  → doit favoriser la régularisation et la généralisation
```

**Hypothèse** : les params B11 devraient montrer plus de régularisation (max_depth plus faible, reg_lambda plus fort) que B8.

---

## 2. Mécanisme — GroupKFold dans l'objectif Optuna

```python
def objective(trial):
    params = { ... hyperparamètres suggérés ... }

    scores = []
    for tr_idx, vl_idx in GroupKFold(n_splits=5).split(X, y, groups):
        pipe = Pipeline([imputer, XGBClassifier(**params)])
        pipe.fit(X[tr_idx], y[tr_idx])
        scores.append(average_precision_score(y[vl_idx], pipe.predict_proba(X[vl_idx])[:, 1]))

    return np.mean(scores)   # ← ce que Optuna maximise

study.optimize(objective, n_trials=30)
```

**Coût** : N_TRIALS × N_FOLDS_OPT fits = 30 × 5 = 150 fits (vs 60 pour B8 Optuna).  
**Évaluation finale** : 15 folds (une machine par fold) pour comparer honnêtement avec B9-GKF.

---

## 3. Fingerprint hyperparamétrique — B8 vs B11

| HP | B8 (split temporel) | B11 (GroupKFold) | Δ | Interprétation |
|----|--------------------|-----------------:|---|----------------|
| n_estimators | 382 | 287 | ↓ | Moins d'arbres — cohérent avec lr plus faible |
| max_depth | 8 | 9 | ↑ | Légèrement plus profond (compensé par forte régularisation) |
| learning_rate | 0.1937 | **0.0289** | ↓↓ | Apprentissage 7× plus lent |
| subsample | 0.987 | 0.825 | ↓ | Plus de stochasticité |
| colsample_bytree | 0.804 | **0.574** | ↓↓ | Beaucoup moins de features par arbre |
| min_child_weight | 6 | **11** | ↑↑ | Feuilles plus contraintes |
| reg_alpha (L1) | 0.00144 | 0.0192 | ↑ | ×13 |
| reg_lambda (L2) | 0.000142 | **0.623** | ↑↑↑ | **×4 380** — signal fort du tuner |

**Hypothèse confirmée** : GroupKFold a poussé le tuner vers un modèle beaucoup plus régularisé. `reg_lambda` quasi nul en B8 (mémorisation tolérée) vs 0.623 en B11 (généralisation forcée). `learning_rate` divisé par 7.

---

## 4. Résultats observés (TP11)

| Modèle | Optuna objectif | PR-AUC CV (15 folds) | PR-AUC test | F1 test | Overfitting Δ |
|--------|----------------|----------------------|-------------|---------|---------------|
| B8/B9-GKF | Split temporel | 0.7792 ± 0.1772 | 0.8595 | 0.7908 | 0.1503 |
| **B11-GKF** | **GroupKFold(5)** | **0.7839 ± 0.1891** | **0.8799** | 0.7586 | 0.2158 |

**Observations :**
- **PR-AUC test 0.8799 = meilleur de toute la série** — le tuning honnête généralise mieux au test.
- **CV quasi-stable** (+0.005) — l'amélioration vient du volume de données (modèle final entraîné sur train+val).
- **F1 recul (0.7586 vs 0.7908)** — le modèle plus régularisé est moins agressif sur le seuil de décision ; acceptable si PR-AUC est la métrique de référence.
- **Overfitting Δ 0.2158** — train=0.9997, CV=0.7839. Reste élevé malgré la régularisation forte. L'overfitting est structurel (hétérogénéité entre machines), pas uniquement lié aux HP.

---

## 5. Règle générale

> Le critère d'optimisation Optuna doit refléter le vrai scénario de déploiement. Optimiser sur un split temporel pour un modèle qui sera déployé sur des machines inconnues introduit un biais systématique dans les hyperparamètres. GroupKFold comme objectif garantit que le tuner cherche des params qui généralisent à de nouvelles entités.

**Limite** : GroupKFold dans l'objectif est coûteux (N_FOLDS × N_TRIALS fits). Pour des datasets larges, réduire N_FOLDS_OPT à 3–5 tout en gardant une évaluation finale complète (N_MACHINES folds).
