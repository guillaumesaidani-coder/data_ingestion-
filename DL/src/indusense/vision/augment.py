"""Pipeline d'augmentation Albumentations pour MVTec AD — catégorie bottle.

Stratégie retenue (objet centré et aligné) :
  ✓ HorizontalFlip       — bouteille symétrique gauche/droite
  ✓ Rotate(limit=15)     — légère inclinaison réaliste
  ✓ RandomBrightnessContrast — variation d'éclairage industriel
  ✓ HueSaturationValue   — teintes légèrement différentes selon lot
  ✓ GaussNoise           — bruit capteur caméra

  ✗ VerticalFlip         — bouteille a un haut et un bas
  ✗ ElasticTransform     — déformation irréaliste sur objet rigide
  ✗ RandomCrop / Scale fort — risque de couper la bouteille
"""

import numpy as np
import albumentations as A


def build_pipeline() -> A.Compose:
    return A.Compose([
        A.HorizontalFlip(p=0.5),
        A.Rotate(limit=15, border_mode=0, p=0.7),
        A.RandomBrightnessContrast(brightness_limit=0.15, contrast_limit=0.15, p=0.7),
        A.HueSaturationValue(hue_shift_limit=5, sat_shift_limit=15, val_shift_limit=10, p=0.5),
        A.GaussNoise(p=0.3),
    ])


def augment_batch(
    images: np.ndarray,
    pipeline: A.Compose,
    n_aug: int = 1,
) -> np.ndarray:
    """Applique le pipeline à chaque image, n_aug fois.

    Args:
        images  : (N, H, W, 3) float32 dans [0, 1]
        pipeline: pipeline Albumentations
        n_aug   : nombre de versions augmentées par image

    Returns:
        (N * n_aug, H, W, 3) float32 dans [0, 1]

    Note: fixer np.random.seed() avant l'appel pour la reproductibilité.
    """
    uint8 = (images * 255).astype(np.uint8)
    augmented = []
    for img in uint8:
        for _ in range(n_aug):
            result = pipeline(image=img)
            augmented.append(result["image"].astype(np.float32) / 255.0)
    return np.stack(augmented)
