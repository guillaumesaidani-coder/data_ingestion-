# Signaux d'alerte en Machine Learning supervisé

> Deux patterns récurrents trahissent un modèle qui triche plutôt qu'il n'apprend.

---

## 1. Feature importance anormalement élevée

### Le signal

Une feature concentre une part disproportionnée de l'importance du modèle — typiquement > 40–50 % à elle seule alors que le dataset en contient des dizaines.

### Pourquoi c'est suspect

Un modèle bien entraîné sur un problème réel répartit généralement son attention sur plusieurs features complémentaires. Une feature ultra-dominante suggère que le modèle a trouvé un **raccourci** — une information qui lui donne la réponse sans avoir à comprendre le phénomène.

Il existe deux types de raccourcis :

---

#### Type A — Fuite temporelle

**Définition** : la feature contient des informations du futur au moment de la prédiction.

**Comment ça arrive :**

```
Exemple : prédire si un client va churner dans les 30 prochains jours.

Feature "nb_connexions_30_derniers_jours" calculée sur [t-30j, t+30j]
                                                                  ↑
                                              inclut accidentellement le futur
```

Le modèle apprend : "si le client ne s'est pas connecté dans les 30 derniers jours (qui incluent les jours où il était déjà parti), il va churner." C'est une tautologie — on utilise le résultat pour prédire le résultat.

**Conséquence :** des métriques parfaites en entraînement/validation, un modèle inutilisable en production car les données futures ne sont pas disponibles au moment de la prédiction.

**Comment le détecter :**
- Vérifier rigoureusement les bornes temporelles de chaque feature (`<` strict vs `<=`)
- Tracer un schéma des fenêtres temporelles pour chaque feature et pour le target
- Tester en "simulation de production" : n'alimenter le modèle qu'avec les données disponibles à t=0

---

#### Type B — Feature proxy du target (duplication de champ)

**Définition** : la feature est une reformulation directe ou quasi-directe de la variable cible.

**Comment ça arrive :**

```
Exemple : prédire si un patient va être hospitalisé.

Feature "nb_médicaments_prescrits_pour_hospitalisation" → évidemment corrélée à la cible
Feature "diagnostic_sévère_posé_aujourd'hui"           → événement simultané à la cible
```

Le modèle n'apprend pas à *anticiper* — il apprend à *reconnaître* une information qui arrive au même moment que la cible, voire qui *est* la cible encodée différemment.

**Conséquence :** le modèle est performant en test mais inutile en production, car la feature proxy n'est pas disponible avant que l'événement se produise.

**Forme plus subtile — le signal réactif :**

```
Exemple : prédire une panne machine dans les 24h.

Feature "incident_sévère_dans_les_24h_précédentes"

→ Si les pannes se produisent en séquences (panne → réparation partielle → re-panne),
  la feature encode "la machine est en train de tomber en panne EN CE MOMENT"
  plutôt que "elle VA tomber en panne".
```

Ce n'est pas une fuite au sens strict — les fenêtres temporelles sont disjointes — mais le modèle devient **réactif** plutôt que **prédictif**. En production, un opérateur qui voit déjà la machine tomber en panne n'a pas besoin d'un modèle pour lui dire qu'elle va retomber en panne.

**Comment le détecter :**
```python
# Entraîner le modèle avec et sans la feature suspecte
pr_avec = average_precision_score(y_val, modele_avec.predict_proba(X_val)[:, 1])
pr_sans = average_precision_score(y_val, modele_sans.predict_proba(X_val)[:, 1])

delta = pr_avec - pr_sans
# delta < 0.05 → la feature est utile mais pas critique ✅
# delta > 0.15 → le modèle dépend quasi-exclusivement de cette feature ⚠️
```

---

### Règle générale

> Une feature à forte importance n'est pas forcément un problème — mais elle **doit s'expliquer** par une logique métier claire et vérifiable. Si on ne peut pas expliquer *pourquoi* cette feature prédit le target sans mentionner le target lui-même, c'est un signal d'alerte.

---

## 2. Recall anormalement élevé

### Le signal

Le modèle détecte une très grande proportion des cas positifs (Recall proche de 90–100 %) — surtout si ce résultat apparaît sans effort particulier de réglage du seuil.

### Pourquoi c'est suspect

Un recall très élevé peut cacher deux pathologies très différentes :

---

#### Pathologie A — Le modèle prédit tout positif

**Mécanisme :** face à un fort déséquilibre de classes (ex. 3.6 % de positifs), le modèle apprend parfois la stratégie paresseuse : prédire systématiquement la classe majoritaire... ou systématiquement la classe minoritaire si le coût d'erreur est asymétrique.

```
Dataset : 3.6 % positifs (pannes), 96.4 % négatifs (OK)

Modèle naïf "prédit tout positif" :
  Recall    = 100 %   ← impressionnant en apparence
  Precision = 3.6 %   ← catastrophique
  F1        = 0.069   ← révèle le problème

Notre modèle XGBoost :
  Recall    = 87.9 %
  Precision = 63.6 %  ← 17× supérieure au naïf → discrimination réelle
```

**Comment le détecter immédiatement :**
```python
# Precision du classifieur naïf "tout positif"
precision_naive = y_val.mean()   # = taux de positifs = 3.6 %

# Si precision_modele ≈ precision_naive → le modèle prédit tout positif
# Si precision_modele >> precision_naive → le modèle discrimine réellement
```

**Règle de survie** : ne jamais regarder le Recall seul. Toujours le lire avec la Precision ou le PR-AUC.

---

#### Pathologie B — Mémorisation par entité (data leakage structurel)

**Mécanisme :** plus subtil et plus dangereux. Le modèle n'a pas appris un signal physique généralisable — il a mémorisé quelles *entités* (machines, clients, patients) ont tendance à générer des événements.

```
Exemple :
  Machine M7 tombe en panne 80 fois dans le train set.
  Machine M7 tombe en panne 20 fois dans le val set.

→ Le modèle apprend "M7 = panne" comme identifiant,
  pas comme signal télémetrique.
→ En validation : Recall excellent sur M7.
→ En production sur une nouvelle machine M_new : le modèle est aveugle.
```

**Comment le détecter :**
```python
# Concentration des vrais positifs par entité
tp_par_entite = (df_val[df_val['TP'] == 1]
                 .groupby('machine_id')
                 .size()
                 .sort_values(ascending=False))

top3_pct = tp_par_entite.head(3).sum() / tp_par_entite.sum()

# top3_pct < 30 % → signal généralisé ✅
# top3_pct > 70 % → mémorisation d'entités ⚠️
```

**Prévention :** utiliser une validation croisée par groupe (`GroupKFold`) plutôt que temporelle si le risque est avéré.

**Résultat observé (TP8b §9 — B8 out-of-entity test sur MACH-13) :**

| Set | TP | PR-AUC complet | PR-AUC OOE | Δ | Verdict |
|-----|----|---------------|------------|---|---------|
| Val | 239 | 0.8025 | 0.6825 | −0.120 | ⚠️ mémorisation partielle |
| Test | 24 | 0.9911 | 1.0000 | +0.009 | ⚠️ non représentatif (trop peu de TP) |

La mesure fiable est le **val set** (239 TP) : Δ = −0.120 confirme une dépendance partielle à MACH-13.
Le résultat test (24 TP, PR-AUC=1.0) est une coincidence statistique sur un trop petit échantillon.

**Règle** : pour un out-of-entity test, privilégier le set avec le plus grand nombre de positifs pour la machine testée. Un PR-AUC parfait sur &lt;50 positifs n'est pas interprétable.

---

### Pourquoi le PR-AUC est plus fiable que le Recall seul

Le Recall mesure la performance à un seuil fixé. On peut toujours obtenir Recall = 100 % en abaissant le seuil à 0 — au prix d'un Precision de 3.6 %.

Le **PR-AUC** (aire sous la courbe Précision-Rappel) mesure la performance sur *tous les seuils possibles*. Un PR-AUC élevé signifie que le modèle maintient une bonne précision même quand on demande un recall élevé — c'est impossible à falsifier par un seuil seul.

```
Modèle naïf "tout positif" :
  PR-AUC ≈ 0.036   (= taux de positifs — plancher théorique)

Modèle aléatoire :
  PR-AUC ≈ 0.036

Notre XGBoost optimisé :
  PR-AUC val = 0.862   → 24× au-dessus du plancher
```

---

## Résumé — checklist de vigilance

| Question à se poser | Signe rassurant | Signe d'alerte |
|---|---|---|
| Y a-t-il une feature à > 40 % d'importance ? | Explicable par la logique métier | Inexplicable sans mentionner le target |
| Les fenêtres temporelles sont-elles disjointes ? | `occurred_at < t` (strict) | `occurred_at <= t` ou calcul ambigu |
| Le Recall est-il élevé ? | Precision >> taux de positifs | Precision ≈ taux de positifs |
| Les TP sont-ils distribués ? | Répartis sur toutes les entités | Concentrés sur 2–3 entités |
| Le PR-AUC est-il élevé ? | PR-AUC >> taux de positifs | PR-AUC ≈ taux de positifs |
| Le modèle performe-t-il sans la feature suspecte ? | Δ PR-AUC < 0.05 | Δ PR-AUC > 0.15 |

---

> **Principe fondamental** : un bon modèle de ML doit pouvoir expliquer *comment* il prédit, pas seulement *que* il prédit bien. Des métriques excellentes sans explication convaincante sont un signal d'alarme, pas une garantie de qualité.

---

## Cas documenté — fuite via `feature_row_id` (TP8b)

### Contexte

Projet de maintenance prédictive. Modèle XGBoost binaire (`label_failure_next_24h`). Lors de l'analyse SHAP post-Optuna, `feature_row_id` apparaît en **position #3** avec un score moyen |SHAP| de 1.031, supérieur à la plupart des features physiques.

### Mécanisme de la fuite

`feature_row_id` est un identifiant de ligne auto-incrémenté. Sur un dataset ordonné chronologiquement :

```
row_id = 1      → janvier (peu de pannes en début de déploiement)
row_id = 50000  → décembre (machine en fin de vie, plus de pannes)
```

Le modèle apprend que les **IDs élevés prédisent les pannes** — ce n'est pas un signal physique, c'est une tendance temporelle encodée sous forme d'identifiant. En production avec de nouvelles données, les IDs continuent de croître mais le modèle n'a aucune référence valide.

Cette fuite n'est pas dans `LEAKAGE_COLS` car ce n'est pas une variable target ni un label futur — elle passe sous les radars des vérifications habituelles.

### Impact mesuré (TP8b)

| Modèle | feature_row_id | PR-AUC val | Δ vs baseline |
|--------|---------------|------------|---------------|
| B5 — XGBoost baseline | ✅ inclus | 0.8174 | — |
| B7 — XGBoost Optuna   | ✅ inclus | **0.8617** | +0.045 |
| B5c — baseline corrigé | ❌ exclu | **0.8349** | — |
| B7c — Optuna corrigé   | ❌ exclu | **0.7504** | −0.084 vs B5c |

**Conséquences observées :**
- B7 était **gonflé de +0.111 PR-AUC** par la fuite
- Optuna a **sur-optimisé autour de la fuite** : les hyperparamètres de B7 exploitaient `feature_row_id` comme levier — une fois retiré, B7c performe moins bien que le baseline B5c non tuné
- Le SHAP corrigé révèle `pressure_zscore_machine` en #2 (2.253) — signal per-machine existant dans le gold dataset mais masqué par la fuite

### Correction appliquée

```python
# Dans la cellule de chargement des données
LEAKAGE_COLS = [
    "machine_id", "ingestion_batch_id", "window_start", "window_end", "split_set",
    "label_failure_next_6h", "label_failure_next_12h", "label_failure_next_48h",
    TARGET,
    "feature_row_id",  # ← ajouté : ID séquentiel = fuite temporelle
]
```

### Comment détecter ce type de fuite

```python
# Vérifier tous les identifiants et colonnes non-feature dans les données
suspect_cols = [c for c in df.columns
                if any(k in c.lower() for k in ["id", "index", "row", "seq", "num"])
                and c not in LEAKAGE_COLS]

for col in suspect_cols:
    corr = df[col].corr(df[TARGET])
    if abs(corr) > 0.05:
        print(f"⚠️  {col} corrélé au target ({corr:.3f}) — vérifier si c'est un identifiant")
```

**Règle** : tout identifiant séquentiel ou auto-incrémenté doit être dans `LEAKAGE_COLS`, même s'il ne contient pas directement la cible. Un ID encode implicitement le temps dans un dataset ordonné chronologiquement.

---

### Étape suivante — B8 : re-tuner après correction

Retirer la fuite ne suffit pas si les hyperparamètres ont été optimisés avec elle. Optuna a cherché à maximiser un PR-AUC qui incluait le signal parasite — les paramètres trouvés (ex. `max_depth`, `min_child_weight`) exploitaient `feature_row_id` comme levier. Une fois retiré, ces paramètres sont sous-optimaux pour le vrai signal, d'où B7c < B5c.

**Cycle correct après détection d'une fuite :**

```
1. Détecter la fuite (SHAP, corrélation, analyse des features)
2. Corriger  → retirer la feature de FEATURE_COLS
3. Re-tuner  → relancer Optuna sur le dataset corrigé (B8)
4. Comparer  → B8 doit dépasser B5c pour valider que le tuning apporte de la valeur réelle
```

Si B8 ne dépasse pas B5c après re-tuning, cela indique que le gain observé de B7 était **entièrement dû à la fuite** — le tuning n'avait aucune valeur propre sur ce signal.

**Résultat observé (TP8b) :**

| Modèle | PR-AUC val | Δ overfitting | Conclusion |
|--------|-----------|---------------|------------|
| B5c — baseline corrigé | 0.8349 | 0.1651 | référence |
| B7c — B7 params corrigés | 0.7504 | 0.2482 | Optuna inutilisable sans la fuite |
| B8 — re-tuné corrigé | **0.8497** | **0.1503** | ✓ B8 > B5c — tuning valide |

- Le tuning apporte **+0.015 PR-AUC val** sur le vrai signal (vs +0.044 annoncé avec la fuite — facteur 3 d'inflation)
- PR-AUC **test = 0.8232**, F1 test = 0.7672 — généralisation correcte (val→test Δ = 0.026)
- L'overfitting est légèrement réduit (Δ 0.1503 < 0.1651)
- La prochaine marge d'amélioration viendrait de l'enrichissement des features télémétriques, pas du tuning

**Empreinte de la fuite dans les hyperparamètres :**

| Hyperparamètre | B7 (avec fuite) | B8 (corrigé) | Ce que ça révèle |
|----------------|----------------|--------------|------------------|
| `max_depth` | 4 (peu profond) | 8 (profond) | Un ID séquentiel dominant ne nécessite pas de profondeur. Le vrai signal physique est plus complexe. |
| `reg_lambda` | 7.877 (fort L2) | 0.00014 (minimal) | La forte régularisation contraignait le modèle à n'utiliser que la feature dominante (la fuite). |
| `learning_rate` | 0.061 (lent) | 0.194 (rapide) | Signal propre = convergence plus directe. |
| `n_estimators` | 444 | 382 | Moins d'arbres suffisent sur un signal propre. |

Les hyperparamètres d'un modèle tuné avec une fuite portent la signature de cette fuite. Après correction, Optuna converge vers une architecture radicalement différente — c'est un test indirect de validité du tuning.

---

## Comment éviter ces dangers

### 1. Construire le pipeline dans le bon ordre

L'erreur la plus fréquente est de calculer les features *après* avoir vu les données de validation ou de test.

```
❌ Ordre incorrect
   Charger toutes les données
   → Calculer les features (moyennes, lags, encodages...)
   → Splitter en train / val / test

✅ Ordre correct
   Splitter en train / val / test   ← D'ABORD
   → Calculer les features sur chaque split séparément
   → Ou : utiliser uniquement des features causales (passé → présent)
```

**Règle** : tout ce qui est appris sur les données (scalers, encodeurs, statistiques de remplissage) doit être `fit` sur le train uniquement, puis `transform` appliqué à val et test.

```python
# ❌ Fuite via StandardScaler
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)          # voit val et test
X_train, X_val = train_test_split(X_scaled)

# ✅ Correct
X_train, X_val = train_test_split(X)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)     # fit sur train seulement
X_val   = scaler.transform(X_val)           # transform sans refit
```

---

### 2. Imposer des bornes temporelles strictes

Pour toute feature calculée sur une fenêtre glissante, la borne supérieure doit être **strictement inférieure** à l'instant d'observation.

```python
# ❌ Borne inclusive — peut inclure l'instant t
w = events[(events['time'] >= t - pd.Timedelta(hours=24)) &
           (events['time'] <= t)]                              # ← <= dangereux

# ✅ Borne exclusive — t non inclus
w = events[(events['time'] >= t - pd.Timedelta(hours=24)) &
           (events['time'] <  t)]                              # ← < strict
```

Pour les séries temporelles, utiliser `shift(1)` systématiquement pour les lags :

```python
# Lag de 1 période — la valeur d'hier pour prédire aujourd'hui
df['feature_lag1'] = df.groupby('machine_id')['valeur'].shift(1)
#                                                              ↑
#                              shift(1) garantit que t n'utilise que t-1
```

---

### 3. Valider avec une simulation de production

La validation classique (train/val split) ne simule pas les conditions réelles. La **walk-forward validation** rejoue le déploiement pas à pas.

```
Walk-forward validation :

  Fold 1 : train [Jan–Mar]  → val [Avr]
  Fold 2 : train [Jan–Avr]  → val [Mai]
  Fold 3 : train [Jan–Mai]  → val [Jun]
  ...

À chaque fold, le modèle ne voit que ce qu'il aurait eu en production.
```

```python
from sklearn.model_selection import TimeSeriesSplit

tscv = TimeSeriesSplit(n_splits=5)
scores = []
for train_idx, val_idx in tscv.split(X):
    X_tr, X_v = X.iloc[train_idx], X.iloc[val_idx]
    y_tr, y_v = y.iloc[train_idx], y.iloc[val_idx]
    model.fit(X_tr, y_tr)
    scores.append(average_precision_score(y_v, model.predict_proba(X_v)[:, 1]))

print(f"PR-AUC moyen : {np.mean(scores):.4f} ± {np.std(scores):.4f}")
# Une forte variance entre folds est un signal d'alerte supplémentaire
```

---

### 4. Tester la robustesse des features importantes

Pour chaque feature à forte importance, appliquer le test d'ablation :

```python
def ablation_study(model, X_train, y_train, X_val, y_val, features):
    """Mesure l'impact de retirer chaque feature."""
    baseline = average_precision_score(
        y_val, model.predict_proba(X_val)[:, 1])

    results = []
    for feat in features:
        cols_sans = [c for c in features if c != feat]
        m = clone(model)
        m.fit(X_train[cols_sans], y_train)
        score = average_precision_score(
            y_val, m.predict_proba(X_val[cols_sans])[:, 1])
        results.append({'feature': feat, 'delta': baseline - score})

    return pd.DataFrame(results).sort_values('delta', ascending=False)
```

**Lecture** :
- `delta` élevé → la feature est importante ET irremplaçable → à inspecter
- `delta` faible → la feature est remplaçable par les autres → moins de risque

---

### 5. Évaluer sur une population que le modèle n'a pas vue

Le test set n'est pas suffisant s'il contient les mêmes entités que le train. Construire un **test out-of-entity** :

```python
# Séparer des machines entières pour le test — jamais vues en train
machines      = df['machine_id'].unique()
test_machines = np.random.choice(machines, size=int(len(machines)*0.2), replace=False)

df_test_oe = df[df['machine_id'].isin(test_machines)]
df_train   = df[~df['machine_id'].isin(test_machines)]
```

Si les performances chutent fortement sur ce test out-of-entity, le modèle a mémorisé des identifiants, pas un signal.

---

### 6. Maintenir un journal des décisions

Documenter chaque choix de feature avec sa justification causale :

| Feature | Justification | Risque identifié | Vérification faite |
|---|---|---|---|
| `temp_mean_24h` | Surchauffe précède la panne | Aucun | Corrélation partielle ✅ |
| `incident_max_severity_prev_24h` | Récurrence des pannes | Signal réactif possible | Test ablation Δ=0.08 ✅ |
| `machine_id` encodé | — | Mémorisation d'entité | **À ne pas inclure** ❌ |

---

### Résumé des garde-fous

| Danger | Garde-fou principal | Vérification |
|---|---|---|
| Fuite temporelle | Bornes strictes `< t` + `shift(1)` | Schéma des fenêtres |
| Feature proxy | Justification causale pour chaque feature | Test d'ablation |
| Recall artificiel | Toujours lire Precision + PR-AUC | Comparer au classifieur naïf |
| Mémorisation d'entité | Test out-of-entity | Concentration des TP par entité |
| Fuite via preprocessing | `fit` sur train uniquement | Pipeline sklearn strict |
| Suroptimisation du val | Test set utilisé une seule fois | Walk-forward validation |
