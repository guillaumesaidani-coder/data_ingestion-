# GroupKFold — Validation par machine (entité)

> Stratégie de validation croisée où chaque fold contient toutes les observations d'une machine entière. Teste la capacité du modèle à généraliser à une **machine jamais vue** plutôt qu'à une période temporelle future.

---

## 1. Le problème : biais de mémorisation machine

Dans le split temporel (TP8b), le train et le val contiennent **les mêmes machines**. Le modèle peut apprendre des signatures spécifiques à chaque machine (patterns vibratoires, niveaux de pression typiques) qui ne se généralisent pas.

```
Split temporel (TP8b) :
  Train  → MACH-01, MACH-02, ..., MACH-15  (période jan–jun)
  Val    → MACH-01, MACH-02, ..., MACH-15  (période jul–sep)
  → La "fuite" machine est structurelle : le modèle sait à quoi ressemble chaque machine.

GroupKFold (TP9) :
  Fold 1 → Train : MACH-02..15  |  Val : MACH-01
  Fold 2 → Train : MACH-01,03..15 | Val : MACH-02
  ...
  → Chaque fold simule le déploiement sur une machine inconnue.
```

**Δ attendu** : le PR-AUC GroupKFold sera inférieur au PR-AUC du split temporel — c'est normal et honnête. Il mesure la généralisation réelle.

---

## 2. Mécanisme

```python
from sklearn.model_selection import GroupKFold

gkf = GroupKFold(n_splits=N_MACHINES)  # un fold = une machine
groups = df_trainval["machine_id"]

for fold, (train_idx, val_idx) in enumerate(gkf.split(X, y, groups)):
    X_tr, y_tr = X.iloc[train_idx], y.iloc[train_idx]
    X_vl, y_vl = X.iloc[val_idx],   y.iloc[val_idx]
    # train + eval sur ce fold
```

**Contrainte** : chaque machine doit avoir suffisamment de positifs pour que le PR-AUC soit calculable (règle : > 50 positifs recommandé).

---

## 3. Implications

### 3a. `n_splits` = nombre de machines

Avec N machines, on fait N folds. Chaque fold évalue sur une machine entière — on obtient N scores indépendants et leur distribution (min/max/std) est informative.

### 3b. Interprétation des résultats

| Score | Interprétation |
|-------|----------------|
| PR-AUC fold élevé (>0.80) | Cette machine est "facile" — ses pannes ressemblent à celles des autres |
| PR-AUC fold faible (<0.60) | Cette machine a des patterns atypiques — potentielle machine outlier |
| Variance inter-fold élevée | Hétérogénéité forte entre machines — modèle global peu fiable |
| Variance inter-fold faible | Signal transférable entre machines — bon signe |

### 3c. `scale_pos_weight` par fold

Le ratio positifs/négatifs varie par machine. Recalculer `scale_pos_weight` à chaque fold évite le biais.

### 3d. Données test

Le test set (split temporel) reste le holdout final — non utilisé pendant le GroupKFold CV. Le modèle final est réentraîné sur tout le train+val avec les meilleurs params.

---

## 4. Quand utiliser GroupKFold

| Situation | Recommandation |
|-----------|---------------|
| Déploiement sur nouvelles machines | **GroupKFold obligatoire** — c'est le vrai scénario de prod |
| Machines très hétérogènes | GroupKFold + analyse par fold pour détecter les outliers |
| Peu de machines (<5) | Attention : chaque fold a peu de données train, variance élevée |
| Prédiction intra-machine uniquement | Split temporel suffit |

---

## 5. Résultats observés (TP9)

| Modèle | Val strategy | PR-AUC val/CV | PR-AUC test | F1 test |
|--------|-------------|---------------|-------------|---------|
| B8 (TP8b) | Split temporel | 0.8497 | 0.8232 | 0.7672 |
| **B9-GKF** | **GroupKFold** | **0.7792 ± 0.1772** | **0.8595** | **0.7908** |

### 5a. PR-AUC par machine (fold val)

| Machine | PR-AUC | Zone |
|---------|--------|------|
| MACH-11 | 1.000 | ✅ Fiable |
| MACH-15 | 1.000 | ✅ Fiable |
| MACH-08 | 0.955 | ✅ Fiable |
| MACH-06 | 0.917 | ✅ Fiable |
| MACH-01 | 0.878 | ✅ Fiable |
| MACH-04 | 0.874 | ✅ Fiable |
| MACH-14 | 0.793 | ⚠️ Fragile |
| MACH-09 | 0.784 | ⚠️ Fragile |
| MACH-12 | 0.761 | ⚠️ Fragile |
| MACH-10 | 0.760 | ⚠️ Fragile |
| MACH-03 | 0.707 | ⚠️ Fragile |
| MACH-02 | 0.664 | 🔴 Problématique |
| MACH-13 | 0.639 | 🔴 Problématique |
| MACH-05 | 0.568 | 🔴 Problématique |
| MACH-07 | 0.329 | 🔴 Outlier |

### 5b. Observations

- **Distribution bimodale** : 6 machines ≥ 0.87 (signal transférable), 4 machines < 0.70 (patterns atypiques).
- **MACH-07 = artefact temporel (0.329)** — diagnostic H1/H2/H3 (§6 TP9) : taux de panne train = 0.8% vs 3.6% flotte (dégradation tardive, burst en test). Features non atypiques (H2 rejetée). Le modèle détecte 93% des pannes quand évalué sur toutes les données MACH-07 (H3 rejetée). Le PR-AUC=0.329 est un artefact : 24 positifs en val, pas un signal que la machine est unlearnable.
- **MACH-13 (0.639)** — cohérent avec le test OOE TP8b (val Δ = −0.120), machine structurellement atypique, confirmée sur deux méthodes.
- **CV moyen = 0.7792** — estimation honnête de la généralisation à machine inconnue. B8 val = 0.8497 était surestimé (mêmes machines en train et val).
- **std = 0.1772** — n'est pas du bruit, c'est de l'hétérogénéité réelle entre machines.
- **Test = 0.8595 > B8 test = 0.8232** — le modèle final entraîné sur train+val (plus de données) améliore le test. C'est un effet volume : chaque fold CV ne voit que N−1 machines.

### 5c. Conséquences opérationnelles

| Segment | Machines | Recommandation |
|---------|---------|----------------|
| Fiables | MACH-01,04,06,08,11,15 | Modèle global suffisant |
| Fragiles | MACH-03,09,10,12,14 | Monitoring renforcé, seuil d'alerte abaissé |
| Problématiques | MACH-02,05,13 | Modèle dédié ou features supplémentaires nécessaires |
| Artefact temporel | MACH-07 | Plus de données historiques — dégradation tardive non représentée en train |

---

## 6. Règle générale

> GroupKFold est la validation de référence pour les systèmes déployés sur des entités nouvelles (machines, clients, sites). Un modèle bon en split temporel mais mauvais en GroupKFold mémorise les identités — et échouera en production sur une machine inconnue.

---

## 7. Prochaine étape — Normalisation intra-machine (TP10)

**Problème identifié** : le modèle mémorise les baselines absolues de chaque machine (pression, température, vibration). Deux machines peuvent avoir des valeurs très différentes à l'état normal — le modèle apprend ces valeurs plutôt que les écarts à la normale.

**Solution** : calculer un z-score par machine sur le train set, puis appliquer la même transformation à val et test.

```python
# Calcul des stats sur le train uniquement (évite la fuite)
machine_stats = train_df.groupby("machine_id")[FEATURE_COLS].agg(["mean", "std"])

# Transformation : (valeur - µ_machine) / σ_machine
def normalize_by_machine(df, stats):
    result = df.copy()
    for col in FEATURE_COLS:
        mu  = df["machine_id"].map(stats[col]["mean"])
        sig = df["machine_id"].map(stats[col]["std"]).replace(0, 1)
        result[col] = (df[col] - mu) / sig
    return result
```

**Impact attendu** :
- CV GroupKFold : 0.7792 → ↑ (signal plus transférable entre machines)
- Variance inter-machines ±0.1772 → ↓ (les machines "fragiles" bénéficient le plus)
- Overfitting Δ : potentiellement ↓ (moins de mémorisation de signatures machine)
