# B7 — Optimisation, éco-conception, explicabilité & industrialisation

> **Pré-requis projet :** vous repartez de **vos modèles déjà entraînés en notebook** :
> le **modèle tabulaire de maintenance** (régression logistique, Random Forest, XGBoost — B5)
> et l'**auto-encodeur de détection d'anomalies** (B6). Tous deux suivis dans MLflow.

## Scénario

**Product Owner** : Deux choses me remontent du comité. D'abord les questions habituelles sur le modèle
de panne : *peut-on faire mieux sans exploser la facture de calcul ? combien ça coûte en énergie ? et
pourquoi le modèle décide ce qu'il décide ?* Ensuite — plus embêtant — **plus personne n'arrive à rejouer
vos résultats**. Tout vit dans des notebooks, les chemins sont codés en dur, l'ordre des cellules change
les scores… On ne peut ni tester, ni versionner, ni relancer proprement.

**Developer** : Deux chantiers, donc. Le premier, **analyser** le modèle sous trois angles :

- **Optimiser** — chercher de meilleurs hyperparamètres avec **Optuna**, mais de façon *bornée* (budget + critère),
  pas en force brute.
- **Mesurer** — instrumenter l'entraînement avec **CodeCarbon** pour chiffrer l'énergie et le CO₂, et arbitrer
  performance **vs** coût.
- **Expliquer** — ouvrir la boîte noire avec **SHAP** pour comprendre *quelles features pèsent* et vérifier que
  c'est cohérent métier.

Le second chantier, **industrialiser** : sortir du notebook. On **remet tout le code dans des fichiers `.py`**
proprement organisés (`src/`), et on écrit des **pipelines exécutables** (`scripts/`) qui rejouent tout
l'enchaînement *data → entraînement → optimisation → évaluation → explicabilité → sauvegarde* en **une seule
commande** — pour le **Machine Learning** (maintenance) **et** le **Deep Learning** (anomalies).

> ⚠️ **Cadre du TP.** On ne change pas le problème ni la donnée : on part de **vos modèles** et de **vos datasets**.
> L'objectif n'est pas un score brillant — sur le jeu de maintenance, les features sont **peu déterminantes** —
> mais d'apprendre à **optimiser proprement, mesurer le coût, lire ce que le modèle a appris**, et surtout à
> **transformer un prototype notebook en code réutilisable et rejouable**. Un notebook qui explore, c'est bien ;
> un notebook qui *est* la seule copie de votre pipeline, c'est une dette.

## Objectifs pédagogiques

À l'issue du TP, vous saurez :

1. Repartir d'une **baseline** et figer une **métrique de référence** honnête (CV temporelle, sans fuite).
2. Lancer une **étude Optuna** bornée (sampler TPE, pruning, budget, reproductibilité) et **analyser** ses résultats.
3. **Instrumenter** un entraînement avec CodeCarbon et raisonner en **coût par point de performance**.
4. Produire des explications **globales** (summary plot) et **locales** (waterfall) avec **SHAP**, et distinguer
   **importance** (corrélation) de **causalité**.
5. **Refactorer** du code de notebook vers une **arborescence `src/` propre** : séparation des responsabilités,
   aucun chemin en dur, configuration centralisée.
6. Écrire des **scripts de pipeline** (`scripts/`) qui orchestrent les briques `src/` de bout en bout, avec
   **arguments CLI**, **graine fixée** et **artefacts reproductibles**.
7. Appliquer la même démarche d'industrialisation au **ML tabulaire** *et* au **Deep Learning** (vision).
8. Distinguer les trois couches d'un projet propre : **notebook** (explorer/raconter) · **`src/`** (logique testable) ·
   **`scripts/`** (pipeline rejouable).

## Les deux chantiers

### Chantier 1 — Trois lentilles sur le même modèle

```
                          ┌──────────────┐
   Gold dataset  ───────► │   XGBoost     │ ──► prédiction de panne
   (votre split temporel) │  (baseline)   │
                          └──────┬───────┘
                                 │
        ┌────────────────────────┼────────────────────────┐
        ▼                        ▼                         ▼
   OPTUNA                   CODECARBON                   SHAP
   « peut-on faire mieux,   « combien ça coûte           « pourquoi le modèle
     dans un budget ? »       en énergie / CO₂ ? »         décide cela ? »
   → meilleurs HP            → kWh, gCO₂eq                → poids des features
   → coût par point d'AUC    → arbitrage frugal           → cohérence métier
```

### Chantier 2 — Du notebook au package rejouable

```
   AVANT (B5/B6)                          APRÈS (B7)
   ────────────                           ──────────
   notebook.ipynb                         notebook.ipynb        ← explore, raconte, visualise
   ├── chemins en dur                          │ importe
   ├── logique mélangée à l'affichage          ▼
   ├── ordre des cellules fragile         src/indusense/…       ← LOGIQUE réutilisable, testable
   └── impossible à rejouer en CLI        ├── data / model / train / evaluate
                                          ├── tune / carbon / explain (ML)
                                          └── dataset / model / train / anomaly (DL)
                                                │ orchestré par
                                                ▼
                                          scripts/run_*_pipeline.py   ← PIPELINE : 1 commande, bout en bout
```

**La règle des trois couches.** Le **notebook** explore et raconte (figures, commentaires) ; il *importe* la
logique, il ne la contient pas. Le paquet **`src/`** contient la logique — fonctions pures, testables, sans
chemin en dur ni `print` décoratif. Les **`scripts/`** sont les points d'entrée exécutables : ils *orchestrent*
les briques `src/` en un enchaînement rejouable via `python scripts/….py`.

## Données & point de départ

- **ML (maintenance)** : votre `indusense_gold.parquet` — features rolling 6h/12h/24h, lookback incidents,
  labels multi-horizons. On travaille sur **un horizon** (ex. `label_failure_next_24h`). Réutilisez **votre split
  temporel** (`train < val < test`) ; le test reste **intouché** jusqu'à la fin. Modèle de départ : votre
  meilleur **XGBoost** baseline + ses métriques MLflow.
- **DL (anomalies)** : MVTec AD (votre catégorrie), votre **auto-encodeur** entraîné sur les pièces saines (B6) (et PatchCore),
  avec son score d'anomalie et son seuil calibré.

> ⚠️ **Déséquilibre (ML).** Les pannes sont rares. Préférez la **PR-AUC** (average precision) à l'accuracy, et
> pensez à `scale_pos_weight`. La **CV doit respecter le temps** (`TimeSeriesSplit` ou folds chronologiques) —
> un `KFold` aléatoire fuiterait l'avenir et gonflerait artificiellement le score.

## Prérequis

Le projet utilise `uv`. Les dépendances sont isolées en groupes :

```bash
# ML : optuna, codecarbon, shap, xgboost, scikit-learn, mlflow
uv sync --group ml
# DL : tensorflow, albumentations, opencv, scikit-image (pour la pipeline vision)
uv sync --group dl
```

> CodeCarbon écrit un `emissions.csv` et estime le CO₂ à partir du mix électrique. En salle, renseignez
> `country_iso_code="FRA"` pour une estimation réaliste (mix français bas carbone) — voir `config.COUNTRY_ISO_CODE`.

## Étapes du TP

### Partie A — Optimisation raisonnée avec Optuna *(ML)*

#### Étape 1 — Repartir de la baseline
- Charger le Gold dataset, reconstruire **votre split temporel**, recharger les hyperparamètres et le score du
  **meilleur XGBoost** baseline.
- *Attendu* : un tableau de référence `modèle baseline → PR-AUC (val), AUC (val)` que l'on cherchera à battre.
- 🔎 **Point de réflexion** : pourquoi figer la métrique de référence **avant** de tuner ? (Indice : sinon on
  ajuste le critère de succès *après coup* — biais de confirmation.)

#### Étape 2 — Définir l'espace de recherche et l'objectif
- Définir un **espace borné** et plausible pour XGBoost : `n_estimators`, `max_depth`, `learning_rate`,
  `subsample`, `colsample_bytree`, `min_child_weight`, `reg_lambda`, `reg_alpha`, `scale_pos_weight`.
- Fonction objectif : **PR-AUC moyenne en CV temporelle** (et non sur le test).
- *Attendu* : une fonction `objective(trial)` qui échantillonne les HP, entraîne, et renvoie la PR-AUC CV.
- 🔎 **Point de réflexion** : pourquoi borner les plages plutôt que d'ouvrir grand ? (Indice : un espace immense
  coûte cher et trouve souvent des optima fragiles, non reproductibles.)

#### Étape 3 — Lancer l'étude (TPE + pruning + budget)
- Créer une `study` Optuna : sampler **TPE**, **pruner** (`MedianPruner` ou Hyperband), `seed` fixé,
  **budget borné** (`n_trials` ex. 40 **et** `timeout`). Activer le **pruning** via les itérations XGBoost.
- *Attendu* : une étude reproductible ; le meilleur essai bat (ou non) la baseline — **les deux cas sont instructifs**.
- 🔎 **Point de réflexion** : sur un signal faible, le gain de tuning est souvent **petit**. À partir de quand
  l'amélioration ne « vaut » plus le calcul dépensé ?

#### Étape 4 — Analyser les résultats
- Produire : **historique d'optimisation**, **importance des hyperparamètres** (`optuna.importance`),
  **slice plot** sur les 2-3 HP dominants.
- *Attendu* : une lecture de la **stabilité** (variance entre essais proches) et des HP qui comptent vraiment.
- 🔎 **Point de réflexion** : un écart `val` flatteur mais une importance d'HP erratique = signe de
  **sur-ajustement du tuning**. Comment s'en prémunir ? (Indice : CV plus robuste, budget raisonnable, re-test.)

### Partie B — Éco-conception avec CodeCarbon *(ML & DL)*

#### Étape 5 — Instrumenter l'entraînement
- Encapsuler l'entraînement (ou l'étude entière) dans un `EmissionsTracker` ; relever **kWh** et **gCO₂eq**.
  La même instrumentation servira pour la pipeline DL (entraînement de l'auto-encodeur).
- *Attendu* : un `emissions.csv` et un tableau `étape → durée, kWh, gCO₂eq, perf obtenue`.

#### Étape 6 — Étude lourde vs étude frugale
- Comparer deux stratégies : **(a)** large espace + 40 essais sans pruning ; **(b)** espace resserré + pruning
  agressif + 15 essais. Calculer le **coût par point de PR-AUC** gagné par rapport à la baseline.
- *Attendu* : un graphe **performance vs CO₂** (ou vs temps) positionnant baseline, frugal, lourd.
- 🔎 **Point de réflexion clé** : l'étude lourde apporte-t-elle un gain **proportionné** à son coût ? Quand la
  **parcimonie** (modèle plus simple, budget réduit) est-elle la bonne décision green AI ?

### Partie C — Explicabilité avec SHAP *(ML)*

#### Étape 7 — Importance globale
- Construire un `TreeExplainer` sur le **XGBoost retenu**, calculer les valeurs SHAP sur un échantillon de
  validation, afficher le **summary plot** (beeswarm) et l'**importance barre** (moyenne des |SHAP|).
- *Attendu* : le **top 10** des features par impact + lecture de la direction (valeur haute → pousse vers panne ?).
- 🔎 **Point de réflexion** : une feature **anormalement dominante** (qui « explique tout ») doit alerter →
  vérifier une **fuite** (corrélation feature ↔ futur). Lien direct avec la checklist anti-leakage du Gold.

#### Étape 8 — Explication locale & cohérence métier
- Choisir une **prédiction individuelle** (un `machine × heure` flagué) et produire un **waterfall** : quelles
  features ont poussé la décision, et de combien. Confronter à votre **intuition métier**.
- *Attendu* : une explication lisible d'un cas + un commentaire « plausible / surprenant ».

#### Étape 9 — Dépendances & diagnostic
- Tracer des **dependence plots** sur 2-3 features du top. Puis synthèse : **l'impact est-il concentré sur
  quelques features fortes, ou diffus sur beaucoup de features faibles ?**
- *Attendu* : un diagnostic argumenté. Sur ce jeu, attendez-vous à une **importance diffuse** → peu de signal.
- 🔎 **Point de réflexion (décision)** : trois suites — **(1)** retour au **feature engineering** (lookback plus
  long, interactions), **(2)** changer **d'horizon** de label (48 h capte parfois plus que 6 h), **(3)** **accepter**
  un modèle modeste mais honnête. Laquelle, et pourquoi ?

### Partie D — Industrialisation : du notebook au package *(ML & DL)*

> C'est le cœur ajouté de ce module. Jusqu'ici, la logique vivait dans les notebooks B5/B6. On la **remet
> proprement dans des fichiers `.py`** et on écrit les **pipelines** qui la rejouent en une commande.

#### Étape 10 — Cartographier le code du notebook
- Reprenez vos notebooks `01_maintenance_ml.ipynb` (ML) et `04_deep_learning_anomalies.ipynb` (DL). Pour chaque
  cellule, classez son contenu par **responsabilité** : *chargement/split données · construction du modèle ·
  entraînement · évaluation · optimisation · mesure carbone · explicabilité · affichage/figures*.
- *Attendu* : un tableau `cellule → responsabilité → module `src/` cible`. L'affichage (`plt.show`, tableaux
  commentés) **reste** au notebook ; tout le reste **descend** dans `src/`.
- 🔎 **Point de réflexion** : une même cellule fait souvent *calcul + affichage* mélangés. Où couper ? (Indice :
  une fonction `src/` **renvoie** des objets/valeurs ; elle ne `print` pas et ne trace pas de figure décorative.)

#### Étape 11 — Refactorer dans une arborescence `src/` propre
- Déplacez la logique dans les modules par domaine, **sans chemin en dur** : tous les chemins passent par
  `config.py` (`config.MODELS_DIR`, `config.RAW_DIR`, `config.FIGURES_DIR`…). Fixez la **graine** partout
  (`config.RANDOM_SEED`).
- Découpage attendu :
  - **ML** → `maintenance/` : `data.py` (chargement + split), `model.py` (factory XGBoost + baselines),
    `train.py` (fit + MLflow), `tune.py` (Optuna), `carbon.py` (CodeCarbon), `explain.py` (SHAP),
    `evaluate.py` (PR-AUC/AUC, tableau d'arbitrage).
  - **DL** → `vision/` : `dataset.py` (MVTec + split), `augment.py` (Albumentations), `model.py` (auto-encodeur),
    `train.py` (fit + MLflow), `anomaly.py` (score, seuil, AUROC, heatmaps).
- *Attendu* : le notebook se réduit à des `import` + appels + figures. Vérifiez qu'il **tourne encore** après
  refactor (aucune régression de résultats).
- 🔎 **Point de réflexion** : pourquoi interdire les chemins en dur (`C:\Users\…`) ? (Indice : le code doit tourner
  sur la machine du collègue et en CI, quel que soit le dossier de lancement — d'où `config.py` relatif à la racine.)

#### Étape 12 — Écrire la pipeline ML (`scripts/run_maintenance_pipeline.py`)
- Un **script exécutable** qui orchestre les briques `maintenance/` de bout en bout :
  `charger données → construire modèles → entraîner + MLflow → [option] tuner Optuna → mesurer CodeCarbon →
  évaluer + tableau d'arbitrage → expliquer SHAP → sauvegarder le meilleur modèle (models/) + rapports (reports/)`.
- Exposez des **arguments CLI** (`argparse`) : `--horizon 24`, `--tune`, `--n-trials 40`, `--no-mlflow`.
- *Attendu* : la commande ci-dessous rejoue **tout** et produit modèle + figures + `emissions.csv` + tableau CSV.

```bash
uv run python scripts/run_maintenance_pipeline.py --horizon 24 --tune --n-trials 40
```

- 🔎 **Point de réflexion** : que doit renvoyer le script s'il est relancé deux fois de suite (idempotence,
  écrasement des artefacts, nom de run MLflow) ? Comment garantir que **deux exécutions donnent le même score** ?

#### Étape 13 — Écrire la pipeline DL (`scripts/run_vision_pipeline.py`)
- Même démarche pour le Deep Learning : `charger MVTec + normaliser + split → augmenter → construire
  l'auto-encodeur → entraîner + MLflow + CodeCarbon → calculer score/seuil/AUROC → heatmaps → sauvegarder
  modèle (models/) + figures (reports/figures/)`.
- Arguments CLI : `--epochs 30`, `--img-size 128`, `--batch-size 32`.
- *Attendu* : une seconde pipeline **symétrique** de la première, prouvant que la démarche d'industrialisation
  vaut pour le ML **comme** pour le DL.

```bash
uv run python scripts/run_vision_pipeline.py --epochs 30 --img-size 128
```

- 🔎 **Point de réflexion** : qu'est-ce qui change entre pipeline ML et DL (device/GPU, taille des artefacts,
  durée d'entraînement, mesure carbone) et qu'est-ce qui reste **identique** (structure, config, MLflow, CLI) ?

### Partie E — Synthèse (passerelle vers le Module B8)

- Remplir un **tableau d'arbitrage** : `modèle | PR-AUC | gCO₂eq | interprétabilité (SHAP) | décision`.
- *Attendu* : une recommandation écrite d'**un** modèle candidat v1 — point de départ de la **note d'arbitrage**
  et de la **model card** (B8), désormais **régénérable** via `scripts/run_maintenance_pipeline.py`.

## Livrables attendus

1. Les notebooks `02_optimisation_explicabilite.ipynb` (et les notebooks B5/B6 **allégés** après refactor)
   qui **importent** `src/` et se limitent à l'orchestration + figures.
2. Le **tableau baseline → tuné** (PR-AUC, AUC) et l'**historique Optuna** + importance des HP.
3. Le **comparatif éco** : tableau `durée / kWh / gCO₂eq / perf` et le graphe **perf vs CO₂**.
4. Au moins **un summary plot** (global) et **un waterfall** (local) SHAP commentés + le **diagnostic** features
   (concentré vs diffus) et la **décision** retenue.
5. **Code industrialisé** : la logique déplacée dans `src/indusense/{maintenance,vision}/`, **sans chemin en dur**.
6. **Deux pipelines exécutables** : `scripts/run_maintenance_pipeline.py` (ML) et `scripts/run_vision_pipeline.py`
   (DL), lançables en une commande, avec arguments CLI, graine fixée et artefacts reproductibles.
7. Le **tableau d'arbitrage** (partie E) avec une recommandation argumentée.

## Implémentation de référence (optionnelle)

Les briques réutilisables existent déjà dans `src/indusense/` — c'est vers elles que doit converger votre refactor :

| Domaine | Module | Rôle |
|---|---|---|
| ML | `maintenance/data.py`     | chargement Gold, split temporel, séparation X/y par horizon |
| ML | `maintenance/model.py`    | factory XGBoost + baselines (logreg, RF), `scale_pos_weight` |
| ML | `maintenance/train.py`    | `fit` + journalisation MLflow (backend SQLite) |
| ML | `maintenance/tune.py`     | espace de recherche, `objective`, étude Optuna (TPE, pruning, seed) |
| ML | `maintenance/carbon.py`   | wrapper `EmissionsTracker`, agrégation `emissions.csv` |
| ML | `maintenance/explain.py`  | `TreeExplainer`, summary / waterfall / dependence, top features |
| ML | `maintenance/evaluate.py` | PR-AUC, AUC, CV temporelle, tableau d'arbitrage |
| DL | `vision/dataset.py`       | chargement MVTec, normalisation, split saines/défauts |
| DL | `vision/augment.py`       | pipeline d'augmentation Albumentations |
| DL | `vision/model.py`         | auto-encodeur 3 conv / 3 déconv (pertes MSE/SSIM) |
| DL | `vision/train.py`         | entraînement + suivi MLflow |
| DL | `vision/anomaly.py`       | erreur de reconstruction, seuil, AUROC, heatmaps |

Les **scripts de pipeline** attendus (`scripts/run_maintenance_pipeline.py`, `scripts/run_vision_pipeline.py`)
ne font qu'**orchestrer** ces briques dans l'ordre, avec `argparse` pour les options et `config.py` pour les
chemins. Aucune logique métier ne doit vivre dans le script : il enchaîne, il ne calcule pas.

> Rappel `config.py` : ne jamais coder un chemin en dur. Utilisez `config.MODELS_DIR`, `config.RAW_DIR`,
> `config.FIGURES_DIR`, `config.REPORTS_DIR`, `config.RANDOM_SEED`, `config.COUNTRY_ISO_CODE`.

## Pour aller plus loin

- **Au-delà de TPE** : sampler CMA-ES, optimisation multi-objectif Optuna (**PR-AUC ↑ et CO₂ ↓** → front de Pareto).
- **Au-delà de SHAP par arbre** : `KernelExplainer` (agnostique mais lent), interactions SHAP, importance par permutation.
- **Green AI** : quantification, réduction du nombre d'arbres à perte négligeable, `CodeCarbon` en continu (CI).
- **MLOps — la suite logique de la partie D** :
  - **Tests** : `pytest` sur les fonctions `src/` (un `data.py` testable est la vraie récompense du refactor).
  - **Un point d'entrée unique** : un `Makefile` ou des `[project.scripts]` dans `pyproject.toml`
    (`indusense-train = "…"`) pour lancer les pipelines par un nom.
  - **CI** : GitHub Actions qui lance `ruff` + `pytest` + un run court de pipeline sur un échantillon.
  - **Versionner données & modèles** : `DVC` ou le Model Registry MLflow, pour tracer *quel* code a produit *quel* modèle.
  - **Configuration** : passer les hyperparamètres via un fichier (`Hydra`, `pydantic-settings`) plutôt que des flags.

## Ressources

- [Optuna — documentation](https://optuna.org/) · [pruning & samplers](https://optuna.readthedocs.io/en/stable/tutorial/index.html)
- [CodeCarbon — mesurer l'empreinte](https://codecarbon.io/)
- [SHAP — documentation](https://shap.readthedocs.io/en/latest/) · [TreeExplainer (papier)](https://www.nature.com/articles/s42256-019-0138-9)
- [XGBoost — paramètres](https://xgboost.readthedocs.io/en/stable/parameter.html)
- [scikit-learn — PR-AUC](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.average_precision_score.html) · [TimeSeriesSplit](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.TimeSeriesSplit.html)
- [argparse — interfaces en ligne de commande](https://docs.python.org/3/library/argparse.html)
- [Cookiecutter Data Science — structurer un projet ML](https://cookiecutter-data-science.drivendata.org/)
- [MLflow — tracking & Model Registry](https://mlflow.org/docs/latest/tracking.html)

---

⬅️ **Précédent** : [B6 (partie 2) — Auto-encodeur & détection d'anomalies](b6_deep_learning_auto_encodeur.md) ·
➡️ **Suivant** : [B8 — Model card](b8_modelcard.md).
