"""Génère la model card (format Hugging Face) de l'auto-encodeur de détection d'anomalies.

Recharge le checkpoint validé (`ae_v3_best.keras`), recalcule les métriques de test en
direct (mêmes fonctions `indusense.vision`, même seuil que TP4 — jamais copiées d'un
ancien run), puis écrit `reports/model_card.md` via le template officiel Hugging Face.

Le livrable est le fichier `.md` — ce script n'est que l'outil qui le produit.

Usage:
    uv run python scripts/generate_model_card.py
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from tensorflow import keras

import config
from indusense.vision.dataset import load_train_val, load_test
from indusense.vision.model import mse_ssim_loss, compression_ratio
from indusense.vision.anomaly import reconstruction_errors, calibrate_threshold, evaluate

from huggingface_hub import ModelCard, ModelCardData
from huggingface_hub.repocard_data import EvalResult

CODECARBON_MEASURE = {
    "source": "TP6.ipynb §1 — run représentatif, PAS le run exact du checkpoint ae_v3_best.keras",
    "hardware": "Intel Core i7-12700H (CPU) + NVIDIA RTX 3050 Ti Laptop (non utilisé par TF, TF>=2.11 sans support GPU natif Windows)",
    "emissions_gco2eq": 0.0515,
    "energy_wh": 0.918,
    "duration_s": 94.8,
    "epochs_measured": 10,
    "country": "France (FRA)",
    "grid_intensity_gco2_kwh": 56,
}


def load_model_and_metrics():
    loss_fn_v3 = mse_ssim_loss(alpha=0.8)
    model_v3 = keras.models.load_model(
        config.CHECKPOINTS_DIR / "ae_v3_best.keras",
        custom_objects={loss_fn_v3.__name__: loss_fn_v3},  # "mse80_ssim19" — arrondi flottant
    )

    X_train, X_val = load_train_val(config.BOTTLE_ROOT, val_ratio=config.VAL_RATIO, seed=config.RANDOM_SEED)
    X_test, y_test, _ = load_test(config.BOTTLE_ROOT)

    errors_val = reconstruction_errors(model_v3, X_val)
    errors_test = reconstruction_errors(model_v3, X_test)
    threshold = calibrate_threshold(errors_val, method="percentile", percentile=99)
    metrics = evaluate(y_test, errors_test, threshold)

    return model_v3, X_train, X_val, X_test, metrics, threshold


def build_card(model_v3, n_train, n_val, n_test, metrics, threshold):
    ratio = compression_ratio(model_v3)

    card_data = ModelCardData(
        model_name="indusense-ae-bottle-v3",
        license="cc-by-nc-sa-4.0",
        library_name="keras",
        tags=["computer-vision", "anomaly-detection", "autoencoder", "manufacturing", "mvtec-ad"],
        datasets=["mvtec-ad"],
        eval_results=[
            EvalResult(
                task_type="anomaly-detection",
                dataset_type="mvtec-ad-bottle",
                dataset_name="MVTec AD — bottle",
                metric_type="auroc",
                metric_value=round(metrics["auroc"], 3),
                metric_name="AUROC (image-level)",
            ),
        ],
    )

    template_kwargs = dict(
        model_id="indusense-ae-bottle-v3",
        model_summary=(
            "Auto-encodeur convolutionnel pour la détection d'anomalies visuelles sur la "
            "catégorie *bottle* de MVTec AD. Score d'anomalie = erreur de reconstruction "
            "(MSE + SSIM) ; un score au-delà d'un seuil calibré signale un défaut candidat."
        ),
        model_description=(
            "Le modèle apprend à reconstruire des images de bouteilles **saines uniquement**. "
            "Un goulot d'étranglement resserré (ratio de compression "
            f"{ratio:.0f}x) empêche une reconstruction fidèle des défauts "
            "absents de l'entraînement : leur erreur de reconstruction est donc statistiquement "
            "plus élevée que celle des pièces saines. Voir TP3/TP4 pour la démarche complète."
        ),
        developers="Guillaume Saïdani",
        model_type="Auto-encodeur convolutionnel (Conv2D / Conv2DTranspose), perte MSE+SSIM",
        language="n/a (vision par ordinateur, pas de NLP)",
        license="CC BY-NC-SA 4.0 (héritée du dataset MVTec AD — usage non commercial)",
        base_model="Aucun — entraîné from scratch (pas de fine-tuning)",
        repo="DL/ (ce dépôt) — checkpoints/ae_v3_best.keras",
        direct_use=(
            "Scorer une image de bouteille (256x256, RGB, [0,1]) et comparer son erreur de "
            "reconstruction MSE+SSIM au seuil calibré "
            f"({threshold:.5f}, percentile 99 sur validation saine) pour obtenir un signal "
            "*candidat défaut / probablement sain*, à l'usage exclusif d'un opérateur humain "
            "en contrôle qualité."
        ),
        downstream_use=(
            "Intégration dans un tableau de bord de contrôle qualité comme signal d'aide à la "
            "décision (score + heatmap d'erreur, cf. TP3/TP4), en complément d'une inspection "
            "humaine — ou combiné à PatchCore-lite (TP4 §3, AUROC 0.859) pour un score ensembliste."
        ),
        out_of_scope_use=(
            "- Décision automatique de rejet/acceptation sans supervision humaine.\n"
            "- Toute catégorie MVTec AD autre que *bottle*, ou tout produit hors MVTec AD, "
            "sans ré-entraînement et recalibration complète du seuil.\n"
            "- Détection de types de défauts absents du jeu de test "
            "(seuls broken_large, broken_small, contamination sont couverts).\n"
            "- Usage réglementaire ou de certification sécurité — ce modèle n'a fait l'objet "
            "d'aucune validation de ce type."
        ),
        bias_risks_limitations=(
            f"- **Performance modeste** : AUROC = {metrics['auroc']:.3f} sur le test MVTec AD "
            f"bottle (rappel {metrics['recall']:.1%}, {metrics['fn']} défauts non détectés sur "
            f"{metrics['fn']+metrics['tp']}). Une alternative interne, PatchCore-lite "
            "(backbone ResNet50 pré-entraîné, TP4 §3), atteint un AUROC de 0.859 sur les mêmes "
            "données — ce modèle-ci reste documenté pour sa valeur pédagogique et parce qu'il "
            "est celui industrialisé par `scripts/run_vision_pipeline.py` (TP7), pas comme "
            "la meilleure option disponible.\n"
            "- **Risque d'identité résiduel** : un goulot trop large réapprend une quasi-copie "
            "de l'entrée (cf. TP3, AUROC 0.465 avant resserrement) ; le ratio actuel "
            f"({ratio:.0f}x) réduit ce risque sans l'éliminer totalement.\n"
            "- **Seuil sensible à la distribution** : calibré au 99e percentile des erreurs de "
            "validation saine — un changement d'éclairage, de fond ou de caméra en production "
            "peut invalider ce seuil sans que le modèle ne le signale.\n"
            "- **Un seul produit** : entraîné et évalué uniquement sur *bottle* — aucune garantie "
            "de généralisation à d'autres catégories ou lignes de production."
        ),
        bias_recommendations=(
            "Toujours faire valider les alertes par un opérateur humain. Recalibrer le seuil "
            "(`calibrate_threshold`) après tout changement d'éclairage/caméra/fond. Envisager "
            "PatchCore-lite (TP4 §3) si le rappel de ce modèle est insuffisant pour l'usage visé."
        ),
        get_started_code=(
            "```python\n"
            "from tensorflow import keras\n"
            "from indusense.vision.model import mse_ssim_loss\n"
            "from indusense.vision.anomaly import reconstruction_errors\n\n"
            "model = keras.models.load_model('checkpoints/ae_v3_best.keras',\n"
            "    custom_objects={'mse80_ssim19': mse_ssim_loss(alpha=0.8)})\n"
            "errors = reconstruction_errors(model, images)  # images: (N,256,256,3) float32 [0,1]\n"
            f"is_anomaly = errors >= {threshold:.5f}\n"
            "```"
        ),
        training_data=(
            f"MVTec AD, catégorie *bottle* — {n_train} images saines (entraînement), "
            f"{n_val} images saines (validation), split fixe (seed={config.RANDOM_SEED}, "
            f"val_ratio={config.VAL_RATIO}). Licence CC BY-NC-SA 4.0. "
            "Voir `dataset_card.yaml` pour la traçabilité complète (source, prétraitement, split)."
        ),
        preprocessing=(
            "Redimensionnement 256x256 par padding centré (letterbox, ratio préservé), "
            "interpolation LANCZOS, normalisation [0,1] (division par 255)."
        ),
        training_regime="fp32, Adam (lr=1e-3), EarlyStopping (patience=10) + ReduceLROnPlateau, augmentation Albumentations (flip, rotation, luminosité, teinte, bruit)",
        speeds_sizes_times=f"Checkpoint : {(config.CHECKPOINTS_DIR / 'ae_v3_best.keras').stat().st_size / 1e6:.1f} Mo — {model_v3.count_params():,} paramètres",
        testing_data=(
            f"MVTec AD, catégorie *bottle*, split test complet — {n_test} images "
            f"({metrics['tn']+metrics['fp']} saines, {metrics['tp']+metrics['fn']} défectueuses : "
            "broken_large, broken_small, contamination)."
        ),
        testing_factors="Aucune stratification par sous-population — évaluation globale toutes classes de défaut confondues.",
        testing_metrics="AUROC (image-level), rappel, précision, spécificité — seuil calibré au 99e percentile des erreurs de reconstruction sur validation saine.",
        results=(
            f"AUROC = {metrics['auroc']:.3f} · Seuil = {threshold:.5f} · "
            f"TP={metrics['tp']} TN={metrics['tn']} FP={metrics['fp']} FN={metrics['fn']} · "
            f"Rappel = {metrics['recall']:.1%} · Précision = {metrics['precision']:.1%} · "
            f"Spécificité = {metrics['specificity']:.1%}"
        ),
        results_summary=(
            f"Précision {metrics['precision']:.0%}, rappel {metrics['recall']:.0%} : le modèle "
            "ne se trompe quasiment jamais quand il signale un défaut, mais en manque une "
            "majorité. Adapté à un usage de pré-filtrage assisté, pas de détection exhaustive autonome."
        ),
        model_examination="Non réalisé pour ce modèle — voir TP5 (SHAP GradientExplainer) pour une analyse d'attribution pixel sur cette même architecture.",
        hardware_type=CODECARBON_MEASURE["hardware"],
        hours_used=f"{CODECARBON_MEASURE['duration_s']/3600:.4f} h ({CODECARBON_MEASURE['epochs_measured']} epochs, run représentatif — voir note ci-dessus)",
        cloud_provider="Aucun — poste de travail local",
        cloud_region=f"{CODECARBON_MEASURE['country']}",
        co2_emitted=f"{CODECARBON_MEASURE['emissions_gco2eq']:.4f} gCO2eq ({CODECARBON_MEASURE['energy_wh']:.3f} Wh, mix {CODECARBON_MEASURE['grid_intensity_gco2_kwh']} gCO2/kWh) — mesure TP6, run représentatif non-identique au run exact du checkpoint",
        model_specs=f"Auto-encodeur : encodeur/décodeur symétrique filters=(32,64,128,64), goulot 16x16x64, ratio compression {ratio:.0f}x, perte 0.8*MSE + 0.2*(1-SSIM)",
        compute_infrastructure="Poste de travail local (pas de cluster) — reproductible via `scripts/run_vision_pipeline.py` (TP7)",
        hardware_requirements="CPU suffisant (pas de dépendance GPU stricte) — TensorFlow >=2.11 sans support GPU natif Windows dans cet environnement",
        software="TensorFlow 2.21, Python 3.13, indusense.vision (ce dépôt)",
        model_card_authors="Guillaume Saïdani",
        model_card_contact="guillaume.saidani@ext.aelion.fr",
    )

    return ModelCard.from_template(card_data, **template_kwargs)


def main():
    print("[1/3] Chargement du modèle validé + recalcul des métriques (identique TP4)...")
    model_v3, X_train, X_val, X_test, metrics, threshold = load_model_and_metrics()
    print(f"  AUROC={metrics['auroc']:.3f}  seuil={threshold:.5f}")

    print("[2/3] Génération de la model card (template Hugging Face officiel)...")
    card = build_card(model_v3, len(X_train), len(X_val), len(X_test), metrics, threshold)
    card.validate()

    print("[3/3] Sauvegarde...")
    card.save(config.REPORTS_DIR / "model_card.md")
    print(f"  Model card sauvegardée : {config.REPORTS_DIR / 'model_card.md'} ({len(str(card))} caractères)")


if __name__ == "__main__":
    main()
