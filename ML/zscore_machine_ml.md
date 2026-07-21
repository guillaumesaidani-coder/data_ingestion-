# Z-Score Machine Pré-Calculé — Feature Engineering (TP12)

> Pré-calculer les z-scores machine sur l'historique complet corrige conceptuellement le domain shift de TP10 — mais empiler 67 nouvelles features sur les 67 raw existantes sans re-tuner les HP dégrade les performances.

---

## 1. Contexte — Correction conceptuelle de TP10

TP10 calculait les z-scores machine **à l'entraînement** (sur le fold train) → la machine val n'avait pas de stats → fallback flotte → domain shift → CV = 0.5580.

TP12 calcule les z-scores machine sur l'**historique complet** du dataset, avant tout split :

```
TP10 (mauvais) : stats = fold_train.groupby("machine_id").agg(["mean","std"])
                 → machine val : NaN → fallback flotte → z_flotte ≠ z_machine → domain shift

TP12 (correct) : stats = df_full.groupby("machine_id").agg(["mean","std"])
                 → machine val : ses propres µ/σ toujours disponibles → pas de shift
```

**Résultat** : le concept est juste, mais l'implémentation a créé un autre problème.

---

## 2. Constat — État des features zscore_machine dans la base

Sur les 69 features disponibles, seulement **2 features zscore_machine** existaient déjà :

| Feature existante | Type |
|-------------------|------|
| `temp_zscore_machine` | Z-score température par machine |
| `pressure_zscore_machine` | Z-score pression par machine |

**67 features raw** sans équivalent zscore_machine : agrégats 1h/6h/24h (temp, pression, voltage, rotation, pièces produites), indicateurs d'incidents (count, sévérité), types d'anomalies.

---

## 3. Résultats B12-GKF

| Modèle | Features | PR-AUC CV | Std | PR-AUC test | Δ CV vs B11 |
|--------|----------|-----------|-----|-------------|-------------|
| B9-GKF (TP9)  | 69  | 0.7792 | 0.1772 | 0.8595 | ref |
| B11-GKF (TP11)| 69  | 0.7839 | 0.1891 | 0.8799 | +0.0047 |
| **B12-GKF**   | **136** | **0.7480** | **0.2300** | **0.8487** | **−0.0312** |

B12 est le **moins bon** modèle des trois. CV et test dégradés, variance inter-machine en hausse.

---

## 4. Diagnostic — Pourquoi ça a dégradé

### H1 : Redondance feature × HP (confirmée)

En empilant 67 zscore_machine sur les 67 raw, on obtient 136 features avec des **paires hautement corrélées** (ex. `pressure_mean_1h` et `pressure_mean_1h_zmachine` mesurent la même grandeur sous deux référentiels).

L'hyperparamètre `colsample_bytree = 0.574` a été optimisé pour 69 features :

```
B11 (69 features)  : 69 × 0.574 ≈ 40 features vues par arbre
B12 (136 features) : 136 × 0.574 ≈ 78 features vues par arbre
```

Chaque arbre B12 voit ~2× plus de features — dont beaucoup redondantes — sans que le tuner ait cherché la valeur optimale pour cet espace. Le signal est dilué.

### H2 : Pas de re-tuning Optuna (confirmée)

Les HP B11 ont été optimisés pour un espace de 69 features. Appliqués à 136, ils sont sous-optimaux :
- `colsample_bytree` trop élevé (trop de features corrélées par arbre)
- `reg_lambda` peut-être insuffisant pour absorber le bruit des nouvelles features
- `n_estimators` potentiellement insuffisant pour un espace plus large

### H3 : Stratégie Stack vs Replace

Deux stratégies distinctes :

| Stratégie | Description | Impact attendu |
|-----------|-------------|----------------|
| **Stack** (TP12) | Raw + zscore_machine — 136 features | Redondance → HP désalignés |
| **Replace** | Remplacer raw par zscore_machine | Même N features, moins de redondance |
| **Select** | Choisir via SHAP les features les plus utiles | Espace réduit, signal concentré |

---

## 5. Évaluation B12b — Stratégie Replace (non retenu)

**Verdict : abandonné.**

Trois raisons rendent la stratégie Replace inutile :

### Raison 1 : XGBoost est invariant à l'échelle

Les arbres de décision apprennent des **seuils de split**, pas des distances. `pressure=205` pour MACH-07 et `pressure=195` pour MACH-01 → le modèle peut apprendre des seuils discriminants par machine sans normalisation. La z-score ne lui apporte pas d'information qu'il ne peut pas extraire lui-même.

### Raison 2 : Les deux zmachine critiques existent déjà dans B11

`temp_zscore_machine` et `pressure_zscore_machine` sont déjà dans les 69 features de B11. Le top SHAP plaçait `pressure_zscore_machine` en #2 (après `incident_max_severity`). Remplacer les z-scores des features de moindre importance (`rotation_mean_1h_zmachine`, `type_blocage_mecanique_count_zmachine`…) n'apportera probablement rien.

### Raison 3 : Les features count z-scorées perdent leur sens absolu

`incident_count=0` vs `1` est très informatif en absolu. Après z-score par machine, la valeur dépend de l'historique d'incidents de chaque machine — potentiellement du bruit.

---

## 6. Règle générale et modèle retenu

> **B11-GKF (test = 0,8799) est le modèle à retenir.** Pré-calculer les z-scores machine sur l'historique complet corrige conceptuellement le domain shift de TP10, mais XGBoost étant invariant à l'échelle, le bénéfice des zmachine supplémentaires est marginal pour un algorithme basé sur des seuils. La prochaine amélioration réelle est soit la **calibration des probabilités** (Platt scaling, pour le réglage de seuil opérationnel), soit une **analyse SHAP** ciblée sur les machines Problématiques.
