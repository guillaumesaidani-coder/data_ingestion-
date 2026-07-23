---
datasets:
- mvtec-ad
library_name: keras
license: cc-by-nc-sa-4.0
tags:
- computer-vision
- anomaly-detection
- autoencoder
- manufacturing
- mvtec-ad
model-index:
- name: indusense-ae-bottle-v3
  results:
  - task:
      type: anomaly-detection
    dataset:
      name: MVTec AD — bottle
      type: mvtec-ad-bottle
    metrics:
    - type: auroc
      value: 0.618
      name: AUROC (image-level)
---

# Model Card for indusense-ae-bottle-v3

<!-- Provide a quick summary of what the model is/does. -->

Auto-encodeur convolutionnel pour la détection d'anomalies visuelles sur la catégorie *bottle* de MVTec AD. Score d'anomalie = erreur de reconstruction (MSE + SSIM) ; un score au-delà d'un seuil calibré signale un défaut candidat.

## Model Details

### Model Description

<!-- Provide a longer summary of what this model is. -->

Le modèle apprend à reconstruire des images de bouteilles **saines uniquement**. Un goulot d'étranglement resserré (ratio de compression 12x) empêche une reconstruction fidèle des défauts absents de l'entraînement : leur erreur de reconstruction est donc statistiquement plus élevée que celle des pièces saines. Voir TP3/TP4 pour la démarche complète.

- **Developed by:** Guillaume Saïdani
- **Funded by [optional]:** [More Information Needed]
- **Shared by [optional]:** [More Information Needed]
- **Model type:** Auto-encodeur convolutionnel (Conv2D / Conv2DTranspose), perte MSE+SSIM
- **Language(s) (NLP):** n/a (vision par ordinateur, pas de NLP)
- **License:** CC BY-NC-SA 4.0 (héritée du dataset MVTec AD — usage non commercial)
- **Finetuned from model [optional]:** Aucun — entraîné from scratch (pas de fine-tuning)

### Model Sources [optional]

<!-- Provide the basic links for the model. -->

- **Repository:** DL/ (ce dépôt) — checkpoints/ae_v3_best.keras
- **Paper [optional]:** [More Information Needed]
- **Demo [optional]:** [More Information Needed]

## Uses

<!-- Address questions around how the model is intended to be used, including the foreseeable users of the model and those affected by the model. -->

### Direct Use

<!-- This section is for the model use without fine-tuning or plugging into a larger ecosystem/app. -->

Scorer une image de bouteille (256x256, RGB, [0,1]) et comparer son erreur de reconstruction MSE+SSIM au seuil calibré (0.00081, percentile 99 sur validation saine) pour obtenir un signal *candidat défaut / probablement sain*, à l'usage exclusif d'un opérateur humain en contrôle qualité.

### Downstream Use [optional]

<!-- This section is for the model use when fine-tuned for a task, or when plugged into a larger ecosystem/app -->

Intégration dans un tableau de bord de contrôle qualité comme signal d'aide à la décision (score + heatmap d'erreur, cf. TP3/TP4), en complément d'une inspection humaine — ou combiné à PatchCore-lite (TP4 §3, AUROC 0.859) pour un score ensembliste.

### Out-of-Scope Use

<!-- This section addresses misuse, malicious use, and uses that the model will not work well for. -->

- Décision automatique de rejet/acceptation sans supervision humaine.
- Toute catégorie MVTec AD autre que *bottle*, ou tout produit hors MVTec AD, sans ré-entraînement et recalibration complète du seuil.
- Détection de types de défauts absents du jeu de test (seuls broken_large, broken_small, contamination sont couverts).
- Usage réglementaire ou de certification sécurité — ce modèle n'a fait l'objet d'aucune validation de ce type.

## Bias, Risks, and Limitations

<!-- This section is meant to convey both technical and sociotechnical limitations. -->

- **Performance modeste** : AUROC = 0.618 sur le test MVTec AD bottle (rappel 22.2%, 49 défauts non détectés sur 63). Une alternative interne, PatchCore-lite (backbone ResNet50 pré-entraîné, TP4 §3), atteint un AUROC de 0.859 sur les mêmes données — ce modèle-ci reste documenté pour sa valeur pédagogique et parce qu'il est celui industrialisé par `scripts/run_vision_pipeline.py` (TP7), pas comme la meilleure option disponible.
- **Risque d'identité résiduel** : un goulot trop large réapprend une quasi-copie de l'entrée (cf. TP3, AUROC 0.465 avant resserrement) ; le ratio actuel (12x) réduit ce risque sans l'éliminer totalement.
- **Seuil sensible à la distribution** : calibré au 99e percentile des erreurs de validation saine — un changement d'éclairage, de fond ou de caméra en production peut invalider ce seuil sans que le modèle ne le signale.
- **Un seul produit** : entraîné et évalué uniquement sur *bottle* — aucune garantie de généralisation à d'autres catégories ou lignes de production.

### Recommendations

<!-- This section is meant to convey recommendations with respect to the bias, risk, and technical limitations. -->

Toujours faire valider les alertes par un opérateur humain. Recalibrer le seuil (`calibrate_threshold`) après tout changement d'éclairage/caméra/fond. Envisager PatchCore-lite (TP4 §3) si le rappel de ce modèle est insuffisant pour l'usage visé.

## How to Get Started with the Model

Use the code below to get started with the model.

```python
from tensorflow import keras
from indusense.vision.model import mse_ssim_loss
from indusense.vision.anomaly import reconstruction_errors

model = keras.models.load_model('checkpoints/ae_v3_best.keras',
    custom_objects={'mse80_ssim19': mse_ssim_loss(alpha=0.8)})
errors = reconstruction_errors(model, images)  # images: (N,256,256,3) float32 [0,1]
is_anomaly = errors >= 0.00081
```

## Training Details

### Training Data

<!-- This should link to a Dataset Card, perhaps with a short stub of information on what the training data is all about as well as documentation related to data pre-processing or additional filtering. -->

MVTec AD, catégorie *bottle* — 168 images saines (entraînement), 41 images saines (validation), split fixe (seed=42, val_ratio=0.2). Licence CC BY-NC-SA 4.0. Voir `dataset_card.yaml` pour la traçabilité complète (source, prétraitement, split).

### Training Procedure

<!-- This relates heavily to the Technical Specifications. Content here should link to that section when it is relevant to the training procedure. -->

#### Preprocessing [optional]

Redimensionnement 256x256 par padding centré (letterbox, ratio préservé), interpolation LANCZOS, normalisation [0,1] (division par 255).


#### Training Hyperparameters

- **Training regime:** fp32, Adam (lr=1e-3), EarlyStopping (patience=10) + ReduceLROnPlateau, augmentation Albumentations (flip, rotation, luminosité, teinte, bruit) <!--fp32, fp16 mixed precision, bf16 mixed precision, bf16 non-mixed precision, fp16 non-mixed precision, fp8 mixed precision -->

#### Speeds, Sizes, Times [optional]

<!-- This section provides information about throughput, start/end time, checkpoint size if relevant, etc. -->

Checkpoint : 4.1 Mo — 334,019 paramètres

## Evaluation

<!-- This section describes the evaluation protocols and provides the results. -->

### Testing Data, Factors & Metrics

#### Testing Data

<!-- This should link to a Dataset Card if possible. -->

MVTec AD, catégorie *bottle*, split test complet — 83 images (20 saines, 63 défectueuses : broken_large, broken_small, contamination).

#### Factors

<!-- These are the things the evaluation is disaggregating by, e.g., subpopulations or domains. -->

Aucune stratification par sous-population — évaluation globale toutes classes de défaut confondues.

#### Metrics

<!-- These are the evaluation metrics being used, ideally with a description of why. -->

AUROC (image-level), rappel, précision, spécificité — seuil calibré au 99e percentile des erreurs de reconstruction sur validation saine.

### Results

AUROC = 0.618 · Seuil = 0.00081 · TP=14 TN=19 FP=1 FN=49 · Rappel = 22.2% · Précision = 93.3% · Spécificité = 95.0%

#### Summary

Précision 93%, rappel 22% : le modèle ne se trompe quasiment jamais quand il signale un défaut, mais en manque une majorité. Adapté à un usage de pré-filtrage assisté, pas de détection exhaustive autonome.

## Model Examination [optional]

<!-- Relevant interpretability work for the model goes here -->

Non réalisé pour ce modèle — voir TP5 (SHAP GradientExplainer) pour une analyse d'attribution pixel sur cette même architecture.

## Environmental Impact

<!-- Total emissions (in grams of CO2eq) and additional considerations, such as electricity usage, go here. Edit the suggested text below accordingly -->

Carbon emissions can be estimated using the [Machine Learning Impact calculator](https://mlco2.github.io/impact#compute) presented in [Lacoste et al. (2019)](https://arxiv.org/abs/1910.09700).

- **Hardware Type:** Intel Core i7-12700H (CPU) + NVIDIA RTX 3050 Ti Laptop (non utilisé par TF, TF>=2.11 sans support GPU natif Windows)
- **Hours used:** 0.0263 h (10 epochs, run représentatif — voir note ci-dessus)
- **Cloud Provider:** Aucun — poste de travail local
- **Compute Region:** France (FRA)
- **Carbon Emitted:** 0.0515 gCO2eq (0.918 Wh, mix 56 gCO2/kWh) — mesure TP6, run représentatif non-identique au run exact du checkpoint

## Technical Specifications [optional]

### Model Architecture and Objective

Auto-encodeur : encodeur/décodeur symétrique filters=(32,64,128,64), goulot 16x16x64, ratio compression 12x, perte 0.8*MSE + 0.2*(1-SSIM)

### Compute Infrastructure

Poste de travail local (pas de cluster) — reproductible via `scripts/run_vision_pipeline.py` (TP7)

#### Hardware

CPU suffisant (pas de dépendance GPU stricte) — TensorFlow >=2.11 sans support GPU natif Windows dans cet environnement

#### Software

TensorFlow 2.21, Python 3.13, indusense.vision (ce dépôt)

## Citation [optional]

<!-- If there is a paper or blog post introducing the model, the APA and Bibtex information for that should go in this section. -->

**BibTeX:**

[More Information Needed]

**APA:**

[More Information Needed]

## Glossary [optional]

<!-- If relevant, include terms and calculations in this section that can help readers understand the model or model card. -->

[More Information Needed]

## More Information [optional]

[More Information Needed]

## Model Card Authors [optional]

Guillaume Saïdani

## Model Card Contact

guillaume.saidani@ext.aelion.fr