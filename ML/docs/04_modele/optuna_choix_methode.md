# Optuna — Pourquoi et comment choisir cette méthode d'optimisation ?

> **Contexte** : après TP7, le modèle XGBoost affiche un PR-AUC val de 0.8174 mais un PR-AUC train de ~1.0 (overfitting Δ = 0.183). L'objectif de TP8 est de réduire cet écart tout en maintenant ou améliorant la performance en validation.

---

## 1. Le problème de départ — pourquoi optimiser ?

Un modèle de ML a deux types de paramètres :

- **Paramètres appris** : les poids, les seuils de décision des arbres → ajustés automatiquement pendant `fit()`
- **Hyperparamètres** : la profondeur des arbres, le taux d'apprentissage, la régularisation → **fixés avant l'entraînement**, non appris

Le choix des hyperparamètres impacte directement :
- la capacité du modèle (trop simple → underfitting, trop complexe → overfitting)
- la vitesse d'apprentissage
- la robustesse en généralisation

### Diagnostic TP7

| Symptôme | Valeur | Interprétation |
|---|---|---|
| PR-AUC train | ~1.000 | Le modèle mémorise le jeu d'entraînement |
| PR-AUC val | 0.817 | Performance réelle bien inférieure |
| Δ overfitting | 0.183 | Écart trop grand → risque en production |
| `incident_max_severity_prev_24h` | 63 % importance | Sur-dépendance à une seule feature |

**Conclusion** : les hyperparamètres par défaut (max_depth=6, learning_rate=0.1) donnent un modèle trop complexe pour ce jeu de données.

---

## 2. Les trois familles de méthodes d'optimisation

### 2.1 Grid Search — recherche exhaustive

```
Idée : tester toutes les combinaisons possibles d'un tableau de valeurs.

max_depth     = [3, 4, 5, 6]
learning_rate = [0.01, 0.05, 0.1, 0.3]
→ 4 × 4 = 16 combinaisons à tester
```

| ✅ Avantages | ❌ Inconvénients |
|---|---|
| Simple à comprendre | Explose exponentiellement (curse of dimensionality) |
| Reproductible | Aucune intelligence — teste des valeurs inutiles |
| Garantit de couvrir la grille | 8 hyperparamètres × 5 valeurs = 5⁸ = 390 625 runs |

**Verdict** : utilisable pour 2–3 hyperparamètres sur plage discrète. Impossible à l'échelle de TP8 (8 HP).

---

### 2.2 Random Search — recherche aléatoire

```
Idée : tirer aléatoirement N combinaisons dans l'espace continu.

max_depth     ~ Uniform(2, 8)
learning_rate ~ LogUniform(0.01, 0.3)
→ N=60 tirages aléatoires indépendants
```

| ✅ Avantages | ❌ Inconvénients |
|---|---|
| Couvre l'espace continu | Aucune exploitation des résultats précédents |
| Meilleur que Grid Search à budget égal | Chaque trial est indépendant — pas d'apprentissage |
| Simple | Gaspille des évaluations sur des zones peu prometteuses |

**Intuition géométrique** : en haute dimension, les points aléatoires couvrent mieux l'espace que les grilles régulières (les grilles ont des "couloirs vides").

---

### 2.3 Optimisation bayésienne — recherche intelligente

```
Idée : construire un modèle probabiliste de la fonction objectif
       et utiliser ce modèle pour choisir le prochain point à évaluer.
```

**Principe en 4 étapes :**

```
1. Évaluer quelques points aléatoires (phase exploratoire)
   ↓
2. Construire un modèle de substitution (surrogate model)
   qui prédit f(hyperparamètres) → PR-AUC val
   ↓
3. Choisir le prochain point selon une fonction d'acquisition
   qui équilibre exploration (zones inconnues) et exploitation (zones prometteuses)
   ↓
4. Évaluer, mettre à jour le surrogate, répéter
```

**Analogie** : comme un chercheur d'or qui creuse là où la probabilité de trouver de l'or est la plus haute d'après ses observations précédentes — plutôt que de creuser au hasard.

| ✅ Avantages | ❌ Inconvénients |
|---|---|
| Chaque trial informe les suivants | Plus complexe à implémenter |
| Converge vers l'optimum en peu d'évaluations | Overhead de calcul du surrogate |
| Efficace sur des fonctions coûteuses | Peut se piéger dans un optimum local |

---

## 3. Optuna — implémentation du TPE

### 3.1 Pourquoi Optuna plutôt que Hyperopt ou Scikit-Optimize ?

| Critère | Optuna | Hyperopt | Scikit-Optimize |
|---|---|---|---|
| Algorithme | TPE (+ CMA-ES, NSGa-II...) | TPE | GP, RF, GBRT |
| API | `suggest_float`, `suggest_int` | dictionnaire de domaines | `@use_named_args` |
| Pruning (early stopping trials) | ✅ natif | ❌ | ❌ |
| Visualisations intégrées | ✅ | ❌ | partiel |
| Maintenance active | ✅ (Preferred Networks) | ralentie | ralentie |
| Intégration MLflow | ✅ | manuelle | manuelle |

**Choix Optuna** : API la plus lisible, pruning natif (arrêt anticipé des mauvais trials), et visualisations intégrées pour comprendre quels hyperparamètres comptent vraiment.

---

### 3.2 Le TPE — Tree-structured Parzen Estimator

C'est l'algorithme de sampling utilisé par défaut dans Optuna. Son nom vient de sa structure interne :

**Principe mathématique simplifié :**

```
Pour chaque hyperparamètre h, TPE modélise deux distributions :

  l(h) = distribution des valeurs de h qui ont donné de BONS résultats
           (trials dont f > percentile γ)

  g(h) = distribution des valeurs de h qui ont donné de MAUVAIS résultats
           (trials dont f ≤ percentile γ)

Le prochain point est choisi là où le ratio l(h) / g(h) est maximal
→ valeurs probables parmi les bons, improbables parmi les mauvais
```

**En pratique :**

```
Trials 1–10  : exploration aléatoire (pas assez de données pour le surrogate)
Trials 11–60 : exploitation — TPE concentre les tirages dans les zones
               où les meilleurs trials précédents se trouvent
```

**Pourquoi "Tree-structured" ?** Parce que l'espace des hyperparamètres peut être conditionnel (ex. : si `booster=dart`, alors `drop_rate` n'existe pas). TPE modélise ces dépendances comme un arbre.

---

## 4. Conception de l'espace de recherche TP8

L'espace n'est pas choisi arbitrairement — chaque paramètre cible un symptôme diagnostiqué en TP7.

```python
def objective(trial):
    params = {
        # Anti-overfitting : réduire la complexité des arbres
        "max_depth":        trial.suggest_int("max_depth", 2, 6),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 20),
        "reg_alpha":        trial.suggest_float("reg_alpha", 0.0, 2.0),
        "reg_lambda":       trial.suggest_float("reg_lambda", 1.0, 10.0),

        # Capacité : équilibre précision / généralisation
        "n_estimators":     trial.suggest_int("n_estimators", 100, 500),
        "learning_rate":    trial.suggest_float("learning_rate", 0.01, 0.3, log=True),

        # Diversification des arbres (réduction de corrélation)
        "subsample":        trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
    }
    return eval_pr_auc_val(params)   # objectif = maximiser PR-AUC val
```

### Justification de chaque plage

| HP | Plage | Raison |
|---|---|---|
| `max_depth` | [2, 6] | Baseline=6 → réduire vers 3–4 simplifie les règles |
| `min_child_weight` | [1, 20] | Augmenter de 1 → évite les feuilles sur-spécialisées |
| `reg_alpha` (L1) | [0, 2] | Baseline=0 → pousse les poids inutiles vers zéro |
| `reg_lambda` (L2) | [1, 10] | Baseline=1 → lisse tous les poids |
| `learning_rate` | log[0.01, 0.3] | Log-scale car l'impact est multiplicatif, pas linéaire |
| `subsample` | [0.6, 1.0] | Sous-échantillonner réduit la corrélation entre arbres |
| `colsample_bytree` | [0.5, 1.0] | Moins de features par arbre = plus de diversité |

---

## 5. Lecture de la courbe de convergence

```
PR-AUC val
  0.94 │
  0.90 │                              ●────────●
  0.86 │              ●───────────────          
  0.82 ├ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─  Baseline 0.8174
  0.78 │     ●
  0.74 │●
  0.70 │
       └──────────────────────────────────────── trial
        1   5   10   20   30   40   50   60
```

**Phase 1 (trials 1–12)** : exploration — TPE n'a pas encore assez d'observations. Les valeurs sont dispersées, certaines mauvaises.

**Phase 2 (trials 12–35)** : concentration — TPE identifie que `max_depth` faible et `reg_lambda` élevé sont prometteurs. Les trials s'améliorent progressivement.

**Phase 3 (trials 35–60)** : exploitation fine — petits ajustements autour du meilleur point connu. La courbe "best so far" stagne → convergence atteinte.

> **Signal de bonne convergence** : si le best ne progresse plus sur les 15–20 derniers trials, 60 trials étaient suffisants. Sinon, augmenter `n_trials`.

---

## 6. Résultats TP8 — ce qu'Optuna a trouvé

### Hyperparamètres retenus

```
max_depth        = 4        (vs 6 baseline)   → moins profond ✅
learning_rate    = 0.061    (vs 0.1 baseline) → plus prudent  ✅
n_estimators     = 444      (vs 300 baseline) → plus d'arbres
min_child_weight = 9        (vs 1 baseline)   → feuilles plus denses ✅
reg_lambda       = 7.88     (vs 1 baseline)   → forte régularisation L2 ✅
subsample        = 0.993
colsample_bytree = 0.820
```

### Impact sur les métriques

| Métrique | Baseline TP7 | Optuna B7 | Δ |
|---|---|---|---|
| PR-AUC val | 0.817 | **0.862** | **+0.045** ✅ |
| Overfitting Δ | 0.183 | **0.136** | **−0.047** ✅ |
| PR-AUC test | — | 0.735 | écart val/test normal |
| Recall test | — | 0.836 | avec seuil = 0.84 |
| ROC-AUC test | — | 0.994 | excellent |

### Lecture du résultat

Optuna a trouvé que la combinaison **arbres moins profonds + régularisation L2 forte + learning rate modéré** réduit efficacement le surapprentissage tout en améliorant la généralisation. La réduction de `max_depth` de 6 à 4 est la décision la plus impactante : elle empêche le modèle de créer des règles trop spécifiques à l'ensemble d'entraînement.

---

## 7. Quand utiliser Optuna en pratique ?

| Situation | Recommandation |
|---|---|
| ≤ 3 HP, plages discrètes connues | Grid Search suffit |
| Budget limité, exploration large | Random Search (n=50–100) |
| ≥ 4 HP, budget ≤ 200 trials | **Optuna TPE** ← notre cas |
| Objectif multi-critères (ex. PR-AUC + empreinte carbone) | Optuna NSGa-II |
| HP très corrélés, espace continu | Optuna CMA-ES |
| Budget > 500 trials, infrastructure cloud | Ray Tune + Optuna backend |

---

## Résumé en une phrase

> Optuna avec le sampler TPE est la méthode d'optimisation d'hyperparamètres qui offre le meilleur ratio **qualité de recherche / nombre d'évaluations** pour des espaces continus de 4 à 20 dimensions — ce qui correspond exactement au problème de fine-tuning XGBoost en maintenance prédictive.
