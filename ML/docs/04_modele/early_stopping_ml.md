# Early Stopping — Régularisation par arrêt anticipé

> Technique de régularisation qui arrête l'entraînement dès que la performance sur le jeu de validation cesse de s'améliorer, évitant la mémorisation du train set.

---

## 1. Le problème : mémorisation progressive

XGBoost construit des arbres **séquentiellement** — chaque arbre corrige les erreurs des précédents. Avec suffisamment d'arbres, le modèle finit par mémoriser le train set ligne par ligne.

```
n_estimators = 50   → PR-AUC train ~0.93  |  val ~0.82   ← généralisation
n_estimators = 150  → PR-AUC train ~0.97  |  val ~0.845  ← optimum
n_estimators = 382  → PR-AUC train ~1.00  |  val ~0.849  ← mémoire + bruit
```

Le passage de train=0.93 à train=1.00 ne correspond pas à "apprendre mieux le signal" — il correspond à **mémoriser le bruit** spécifique au train set. Ce bruit ne se généralisera pas.

**Coût** : l'overfitting Δ = train − val mesure ce que le modèle a "inventé" sur le train.
Dans TP8b : Δ = 1.000 − 0.850 = **0.150** — 15 points qui disparaîtront en production.

---

## 2. Mécanisme de l'early stopping

Au lieu de fixer `n_estimators` à l'avance, on surveille la performance sur un **eval set** à chaque arbre ajouté :

```
Arbre 001 → val PR-AUC = 0.710   ← new best
Arbre 050 → val PR-AUC = 0.830   ← new best
Arbre 120 → val PR-AUC = 0.852   ← new best
Arbre 125 → val PR-AUC = 0.851
Arbre 140 → val PR-AUC = 0.849
Arbre 150 → val PR-AUC = 0.851
           ↑ 30 arbres sans amélioration → STOP
           → modèle conservé à l'arbre 120
```

```python
model = XGBClassifier(
    n_estimators=1000,          # borne haute — ne sera pas atteinte
    early_stopping_rounds=30,   # stop si pas de progrès sur 30 arbres consécutifs
    eval_metric="aucpr",
)
model.fit(
    X_train, y_train,
    eval_set=[(X_val, y_val)],  # surveille ce set à chaque arbre
    verbose=False,
)

print(model.best_iteration)   # arbre optimal trouvé
print(model.best_score)       # PR-AUC à cet arbre
```

---

## 3. Implications

### 3a. Val set comme signal d'arrêt

Le val set influence indirectement l'entraînement (il détermine quand s'arrêter). Ce n'est pas une fuite de données — les labels val ne sont pas utilisés comme features — mais c'est à mentionner si on reporte les métriques val comme "jamais vues".

**Règle** : quand on utilise l'early stopping, les métriques **test** sont la référence propre. Les métriques val sont biaisées par leur rôle dans le critère d'arrêt.

### 3b. `n_estimators` devient une borne haute

On le fixe à une valeur élevée (500, 1000) — le modèle ne l'atteindra généralement pas. Si `best_iteration` est proche de `n_estimators`, il faut augmenter la borne.

### 3c. `early_stopping_rounds` — sensibilité

| Valeur | Comportement |
|--------|-------------|
| Trop petit (5–10) | S'arrête trop tôt, sous-apprend |
| Raisonnable (20–50) | Équilibre généralisation / capacité |
| Trop grand (200+) | Laisse trop longtemps sur le plateau, risque de surapprentissage |

Valeur typique : **20–50** selon la volatilité du score de validation.

### 3d. Compatibilité avec sklearn Pipeline

L'early stopping nécessite de passer `eval_set` au `fit()`. Dans un Pipeline sklearn, cela se fait via `model__eval_set` dans les paramètres du fit — mais le val set doit déjà être transformé par l'imputer. La solution propre est de séparer l'imputer du modèle :

```python
imputer = SimpleImputer(strategy="median").fit(X_train)
X_train_imp = imputer.transform(X_train)
X_val_imp   = imputer.transform(X_val)

model_es = XGBClassifier(n_estimators=1000, early_stopping_rounds=30, ...)
model_es.fit(X_train_imp, y_train,
             eval_set=[(X_val_imp, y_val)],
             verbose=False)
```

---

## 4. Quand utiliser l'early stopping

| Situation | Recommandation |
|-----------|---------------|
| train ≈ val (Δ < 0.05) | Early stopping peu utile — le modèle ne surapprend pas |
| train >> val (Δ > 0.10) | **Early stopping recommandé** — la mémorisation est avérée |
| Données bruitées | Early stopping protège contre l'apprentissage du bruit |
| Features peu discriminantes | Le modèle compensera par plus d'arbres → early stopping utile |

---

## 5. Résultats observés (TP8c)

| Modèle | n_estimators utilisés | PR-AUC val | PR-AUC test | Overfitting Δ |
|--------|----------------------|------------|-------------|---------------|
| B8 (sans early stopping) | 382 (fixe) | 0.8497 | **0.8232** | 0.1503 |
| B8-ES (early stopping) | **57** | 0.8471 | 0.7968 | 0.1529 |

**Observation** : B8-ES s'est arrêté à l'arbre 57 (val plateau atteint), mais le modèle B8 avec 382 arbres généralise **mieux** en test (0.8232 vs 0.7968, soit −0.026). Le Δ overfitting reste quasi identique (0.1529 vs 0.1503).

**Explication** : `learning_rate=0.194` et `max_depth=8` (params B8) font converger le modèle très rapidement — val atteint son plateau en 57 arbres, mais les arbres 57–382 apportaient encore de l'information utile pour le test set. L'overfitting est **structurel** (distribution des features, écart temporel train/test) et non dû à l'excès d'arbres.

**Conséquence** : dans ce cas précis, l'early stopping dégrade les performances en test sans réduire l'overfitting. B8 reste le meilleur modèle de la série.

---

## 6. Règle générale

> L'early stopping ne supprime pas l'overfitting — il le **limite** en trouvant le point optimal sur la courbe val. Si Δ reste élevé après early stopping, le problème est dans les features (signal trop faible ou trop réactif), pas dans le nombre d'arbres.

**Cas TP8c** : l'early stopping avec `early_stopping_rounds=30` et `learning_rate=0.194` a trouvé un point trop conservateur. Si val ≠ test en distribution (biais temporel), s'arrêter sur val peut dégrader test. Quand early stopping ne réduit pas Δ, la prochaine étape est **GroupKFold** (validation par machine) ou **enrichissement des features**.
