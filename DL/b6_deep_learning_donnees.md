# B6 (partie 1) — Préparation des données images (MVTec AD)


## Scénario

**Product Owner** : Ce serait parfait si on pouvait ajouter une dimension « perceptive » au système : repérer automatiquement les défauts visuels sur les pièces produites. Le problème, c'est qu'on ne dispose pas d'images de pièces défectueuses annotées. Tu as une idée ?

**Developer** : Oui, et il faut bien distinguer deux familles d'approches :

- **Classification / localisation supervisée (type YOLO).** Très efficace pour localiser des objets précis, mais il faut des **milliers d'exemples annotés manuellement** pour chaque type de défaut. Irréaliste à ce stade.

- **Approche non supervisée par reconstruction (auto-encodeur CNN).** On apprend au modèle à reconstruire **uniquement des pièces saines**. Tout ce qu'il reconstruit mal devient suspect. Avantage : **aucun défaut annoté requis**. Limite : c'est un *baseline* — la performance dépend fortement de l'architecture et du choix du seuil.

> ⚠️ **Cadre du TP.** On choisit délibérément l'auto-encodeur pour sa valeur **pédagogique**. Ce n'est pas l'état de l'art. Cette première partie prépare le terrain : sans données propres et une augmentation bien pensée, le modèle de la partie 2 ne peut pas bien fonctionner.

## Objectifs pédagogiques (partie données)

À l'issue de cette partie, vous saurez :

1. Charger et préparer un dataset d'images industriel (MVTec AD).
2. Mettre en place une **augmentation de données** (Albumentations) et en discuter les effets.

## Jeu de données : MVTec AD

[MVTec AD](https://www.mvtec.com/company/research/datasets/mvtec-ad) est un standard académique de détection d'anomalies industrielles : objets et textures, images saines pour l'entraînement, images défectueuses **et masques de vérité terrain** pour le test.

- On travaille sur **une seule catégorie** (ex. `bottle`, ~250 Mo) — inutile de télécharger les ~5 Go complets.
- Structure attendue :

```
bottle/
├── train/good/            # images saines  → entraînement
├── test/good/             # images saines  → évaluation
├── test/<defect>/         # défauts (broken_large, broken_small, contamination)
└── ground_truth/<defect>/ # masques binaires des défauts (pour l'évaluation pixel)
```

## Prérequis

Le projet utilise `uv`. Les dépendances Deep Learning sont isolées dans un groupe `dl` :

```bash
# Installer le groupe dl (tensorflow, albumentations, opencv, scikit-image, pillow)
uv sync --group dl
```

## Étapes du TP

### Étape 1 — Préparer les données
- Télécharger la catégorie `bottle`, charger les images, **redimensionner** (256×256 ou 128×128) et **normaliser** dans `[0, 1]`.
- Réserver une fraction des **saines** pour la **validation** (servira à calibrer le seuil en partie 2).
- *Attendu* : afficher quelques saines et quelques défauts pour vérifier le chargement.

### Étape 2 — Augmenter les données (Albumentations)
- Construire un pipeline appliqué **uniquement aux saines d'entraînement** (flips, rotations légères, translation/échelle, variations d'éclairage).
- *Attendu* : visualiser original vs versions augmentées.
- 🔎 **Point de réflexion** : l'augmentation géométrique forte est-elle pertinente sur un objet **centré et aligné** comme `bottle` ? (Indice : elle élargit la notion de « normal » et peut rendre le modèle plus tolérant aux défauts. Comparez objet vs texture.)

## Livrables attendus (partie données)

1. Un notebook (ou la première section de `tp_deep_learning.ipynb`) couvrant les étapes 1 et 2.
2. Une figure montrant des images saines et défectueuses après chargement/normalisation.
3. Une figure original vs versions augmentées.
4. Un court paragraphe justifiant les transformations retenues (et écartées) pour la catégorie choisie.

## Implémentation de référence 

Deux modules réutilisables couvrent cette partie dans `src/indusense/vision/` :

| Module | Rôle |
|---|---|
| `dataset.py`  | téléchargement MVTec, chargement, normalisation, split, masques GT |
| `augment.py`  | pipeline Albumentations sur les saines |

## Ressources

- [MVTec AD — dataset](https://www.mvtec.com/company/research/datasets/mvtec-ad)
- [Albumentations — guide augmentation](https://www.datacamp.com/fr/tutorial/complete-guide-data-augmentation)
- [CNN — introduction](https://fr.mathworks.com/discovery/convolutional-neural-network.html)

---
