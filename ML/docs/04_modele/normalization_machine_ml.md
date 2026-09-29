# Normalisation intra-machine — Z-score par machine

> Technique qui centre et réduit chaque feature par machine individuellement, transformant les valeurs absolues en écarts par rapport à la normale de chaque machine. Le modèle apprend des déviations plutôt que des niveaux absolus.

---

## 1. Le problème : baselines machine hétérogènes

Deux machines peuvent avoir des valeurs de pression très différentes à l'état normal — l'une à 195 bar, l'autre à 205 bar. Sans normalisation, le modèle associe "205 bar = normal pour MACH-X" au lieu d'apprendre "un écart de +10 bar en 1h est suspect quelle que soit la machine".

```
Sans normalisation :
  MACH-01 : pressure_mean = 195 (normal) → label = 0
  MACH-07 : pressure_mean = 205 (normal) → label = 0
  MACH-01 : pressure_mean = 205 (anomalie) → label = 1  ← confus pour le modèle

Avec z-score par machine :
  MACH-01 : z = (195 - µ_01) / σ_01 = 0.0 → label = 0
  MACH-07 : z = (205 - µ_07) / σ_07 = 0.0 → label = 0
  MACH-01 : z = (205 - µ_01) / σ_01 = 3.2 → label = 1  ← signal clair
```

**Conséquence attendue** : signal plus transférable entre machines → meilleur CV GroupKFold, variance inter-machines réduite.

---

## 2. Mécanisme

```python
# 1. Calculer µ et σ sur le train set uniquement
machine_stats = train_df.groupby("machine_id")[FEATURE_COLS].agg(["mean", "std"])

# 2. Appliquer la transformation
def normalize_by_machine(df, machine_stats, fleet_stats, feature_cols):
    result = df.copy()
    for col in feature_cols:
        mu  = df["machine_id"].map(machine_stats[col]["mean"])
        sig = df["machine_id"].map(machine_stats[col]["std"])
        # Fallback flotte pour machines inconnues (µ et σ NaN)
        mu.fillna(fleet_stats[col]["mean"],  inplace=True)
        sig.fillna(fleet_stats[col]["std"],   inplace=True)
        sig.replace(0, 1, inplace=True)
        result[col] = (df[col] - mu) / sig
    return result
```

**Règle anti-fuite** : les stats (µ, σ) sont toujours calculées sur le train set du fold courant, jamais sur val ou test.

---

## 3. Cas particulier — machine inconnue dans GroupKFold

Dans un fold GroupKFold, la machine val n'a **aucune ligne dans le train** → pas de stats disponibles. Deux stratégies :

| Stratégie | Description | Scénario réel |
|-----------|------------|---------------|
| **Fleet fallback** | Utiliser µ/σ de la flotte pour la machine val | Machine toute neuve, zéro historique |
| **Burn-in** | La machine collecte des données pendant N semaines avant les prédictions | Déploiement progressif |

TP10 implémente le **fleet fallback** (scénario conservateur). En production, le burn-in est préférable dès que possible.

---

## 4. Limites

- **Machines avec très peu de données** : σ instable si <100 observations par machine
- **Dérive de machine** : si les baselines changent dans le temps (usure), les stats calculées en début de vie deviennent fausses
- **Features déjà normalisées** : certaines features (zscore_machine) sont déjà des z-scores → doublement inutile, à exclure

---

## 5. Résultats observés (TP10)

| Modèle | Val strategy | PR-AUC CV | Std | PR-AUC test |
|--------|-------------|-----------|-----|-------------|
| B9-GKF (TP9) | GroupKFold sans normalisation | **0.7792** | 0.1772 | **0.8595** |
| B10-NRM (TP10) | GroupKFold + z-score machine à l'entraînement | 0.5580 | 0.2279 | — |

**La normalisation a dégradé tous les modèles (−0.22 de CV moyen, std plus élevée).**

Seules MACH-13 (+0.063) et MACH-01 (+0.049) ont progressé. 13 machines sur 15 ont régressé, certaines massivement (MACH-06 : −0.483, MACH-14 : −0.330).

### 5a. Diagnostic — pourquoi ça a échoué

Dans GroupKFold, la machine val n'a aucune ligne dans le train → ses µ/σ sont inconnues → fallback flotte.

```
Train  : z_machine = (valeur - µ_machine) / σ_machine  ← centré sur chaque machine
Val    : z_fleet   = (valeur - µ_flotte)  / σ_flotte   ← centré sur la flotte

→ Le modèle évalue des z-scores "flotte" en ayant appris sur des z-scores "machine"
→ Décalage de distribution train/val → dégradation forte
```

C'est l'effet inverse de ce qu'on voulait : la normalisation crée une **domain shift** entre train et val.

### 5b. La bonne approche — burn-in ou pré-calcul

La normalisation intra-machine ne fonctionne que si la machine est **connue au moment de l'inférence**. Deux options valides :

| Option | Description | Disponible ? |
|--------|------------|-------------|
| **Burn-in** | Collecter N semaines de données avant de prédire | En production |
| **Pré-calculé** | Features `zscore_machine` déjà dans la BDD (calculées avec historique propre à la machine) | ✅ Déjà dans le dataset |

**Observation** : les features `zscore_machine` ont été exclues en TP10 pour "éviter le double". C'était une erreur — elles représentent exactement la normalisation correcte, pré-calculée à la feature engineering avec le bon historique machine.

---

## 6. Règle générale

> La normalisation intra-machine appliquée **à l'entraînement** dans un contexte GroupKFold crée une domain shift pour les machines inconnues (val/test). Elle n'est valide que si les stats machine sont disponibles au moment de l'inférence (burn-in ou pré-calcul en feature engineering). Les features `zscore_machine` déjà présentes dans le dataset sont la bonne implémentation — les garder, ne pas les recalculer.

**Prochaine étape** : B9-GKF reste le meilleur modèle (CV=0.7792, test=0.8595). Le levier suivant est l'enrichissement des features signal (nouvelles features temporelles, inter-machines) ou l'optimisation Optuna avec GroupKFold comme fonction d'évaluation.
