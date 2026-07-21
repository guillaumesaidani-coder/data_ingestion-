# B6 (partie 2) — Auto-encodeur & détection d'anomalies (Deep Learning)

> Couvre les modules 15 à 18 du parcours (partie modèle).
> Durée indicative : ~2 h. Niveau : intermédiaire (Python, NumPy, bases CNN).
> **Prérequis** : [B6 (partie 1) — Préparation des données images (MVTec AD)](b6_deep_learning_donnees.md). Les données doivent être chargées, normalisées, splittées et augmentées.

## Rappel du contexte

On ne dispose pas d'images de défauts annotées. On adopte donc une **approche non supervisée par reconstruction** : un auto-encodeur CNN apprend à reconstruire **uniquement des pièces saines** ; tout ce qu'il reconstruit mal devient suspect.

> ⚠️ **Cadre du TP.** L'auto-encodeur est choisi pour sa valeur **pédagogique** : il met en jeu convolution, déconvolution, détection non supervisée et visualisation. Ce n'est pas l'état de l'art (voir [Pour aller plus loin](#pour-aller-plus-loin)). L'un des objectifs est justement de **comprendre pourquoi** un auto-encodeur simple peut échouer.

## Objectifs pédagogiques (partie modèle)

À l'issue de cette partie, vous saurez :

1. Concevoir un **auto-encodeur convolutionnel** (Keras/TensorFlow) et lire son `summary()`.
2. Entraîner le modèle sur les seules pièces saines et suivre l'entraînement (MLflow).
3. Construire un **score d'anomalie** et **calibrer un seuil** de décision.
4. Produire des **heatmaps de reconstruction** pour localiser les défauts et les comparer aux masques de vérité terrain.
5. **Évaluer quantitativement** la détection (AUC-ROC) et **critiquer** les résultats.

## Le principe : apprendre le normal, signaler l'inattendu

Un auto-encodeur compresse une image vers un **espace latent** (encodeur) puis tente de la **reconstruire** (décodeur). Entraîné uniquement sur des pièces saines, il devrait reconstruire fidèlement le normal et **mal** reconstruire ce qu'il n'a jamais vu (un défaut). L'écart entre l'image et sa reconstruction — l'**erreur de reconstruction** — sert alors de signal d'anomalie.

```
image saine  ──► encodeur ──► latent ──► décodeur ──► reconstruction ≈ image      (erreur faible)
image défaut ──► encodeur ──► latent ──► décodeur ──► reconstruction ≠ image      (erreur forte → anomalie)
```

## Étapes du TP

### Étape 1 — Concevoir l'auto-encodeur
- Architecture demandée : **3 couches de convolution** (encodeur) + **3 couches de déconvolution** (décodeur), sortie `sigmoid`.
- *Attendu* : afficher `model.summary()` et **commenter le nombre de paramètres** ainsi que la **taille du goulot d'étranglement**.
- 🔎 **Point de réflexion clé** : calculez le ratio de compression (nb de valeurs en entrée ÷ nb de valeurs dans le latent). S'il est proche de 1, le modèle peut apprendre une quasi-**identité** et reconstruire *aussi* les défauts → le signal d'anomalie disparaît. C'est le piège central de ce TP.

### Étape 2 — Entraîner
- Entraîner avec la cible = l'entrée (reconstruction), `Adam`, perte `MSE` (essayez aussi **`SSIM`**, plus sensible aux altérations de texture).
- *Attendu* : courbe d'apprentissage train/validation, exemples de reconstruction, suivi du run dans **MLflow**.

### Étape 3 — Score d'anomalie & seuil
- Calculer l'erreur de reconstruction par image. **Calibrer le seuil sur les saines de validation** (ex. centile 99, ou `moyenne + k·écart-type`).
- *Attendu* : histogramme des scores saines vs défauts avec la ligne de seuil.
- 🔎 **Point de réflexion** : abaisser le seuil augmente le **rappel** (on rate moins de défauts) mais aussi les **fausses alertes**. Quel arbitrage selon le coût métier d'une panne manquée ?

### Étape 4 — Heatmaps & évaluation
- Produire la **carte d'erreur par pixel** (heatmap) et l'afficher : `original | reconstruction | heatmap | masque ground-truth`.
- **Évaluer quantitativement** : **AUROC niveau image** (et, si possible, **AUROC niveau pixel** via les masques). Ne pas se contenter du visuel.
- *Attendu* : matrice de confusion, AUROC, et une **analyse critique** des cas ratés.

## Livrables attendus (partie modèle)

1. Un notebook exécuté de bout en bout (`tp_deep_learning.ipynb`) couvrant les étapes de conception, entraînement et évaluation.
2. Le `summary()` du modèle commenté.
3. Au moins une figure de **heatmaps** sur des défauts réels.
4. Les **métriques chiffrées** (AUROC image au minimum) + matrice de confusion.
5. Un court paragraphe d'**analyse critique** : qu'est-ce qui marche, qu'est-ce qui échoue, et pourquoi.

## Implémentation de référence (optionnelle)

Trois modules réutilisables couvrent cette partie dans `src/indusense/vision/` :

| Module | Rôle |
|---|---|
| `model.py`    | auto-encodeur 3 conv / 3 déconv + `summary` (pertes MSE/SSIM) |
| `train.py`    | entraînement + suivi MLflow |
| `anomaly.py`  | erreur de reconstruction, seuil, métriques (AUROC), heatmaps |

Le notebook `tp_deep_learning.ipynb` orchestre ces briques avec celles de la [partie 1](b6_deep_learning_donnees.md) (`dataset.py`, `augment.py`). Vous pouvez soit le suivre, soit ré-implémenter les étapes vous-même.

## Pour aller plus loin

L'auto-encodeur de reconstruction est un **baseline**. Il est aujourd'hui largement dépassé sur MVTec AD par des méthodes basées sur des **features pré-entraînées**, qui répondent au même besoin (peu/pas de défauts, pas d'annotation) avec de bien meilleurs scores :

- **PatchCore**, **PaDiM**, **SPADE** — mémoire de features pré-entraînées (souvent > 98 % AUROC image).
- **Student-Teacher**, **Normalizing Flows (FastFlow, CFLOW)**.
- Côté reconstruction : **MemAE**, **denoising AE**, ou perte **SSIM**, pour atténuer le problème d'identité évoqué à l'étape 1.

Pistes d'amélioration directes sur ce TP : resserrer le goulot d'étranglement, passer la perte en `SSIM`, comparer objet vs texture, et confronter le baseline AE à PatchCore (lib [anomalib](https://github.com/openvinotoolkit/anomalib)).

## Ressources

- [CNN — introduction](https://fr.mathworks.com/discovery/convolutional-neural-network.html)
- [YOLO — pour comparaison (approche supervisée)](https://larevueia.fr/yolo/)
- [MLflow — tracking](https://mlflow.org/docs/latest/tracking.html)
- [anomalib — bibliothèque de méthodes SOTA](https://github.com/openvinotoolkit/anomalib)

---

⬅️ **Précédent** : [B6 (partie 1) — Préparation des données images (MVTec AD)](b6_deep_learning_donnees.md).
