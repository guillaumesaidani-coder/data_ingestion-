"""Chargement et préparation du dataset MVTec AD (catégorie bottle)."""

from pathlib import Path

import numpy as np
from PIL import Image

IMG_SIZE: tuple[int, int] = (128, 128)
VAL_RATIO: float = 0.2
SEED: int = 42


def _load_image(path: Path, size: tuple[int, int] = IMG_SIZE) -> np.ndarray:
    img = Image.open(path).convert("RGB")
    img = img.resize(size, Image.LANCZOS)
    return np.array(img, dtype=np.float32) / 255.0


def _load_mask(path: Path, size: tuple[int, int] = IMG_SIZE) -> np.ndarray:
    mask = Image.open(path).convert("L")
    mask = mask.resize(size, Image.NEAREST)
    return (np.array(mask) > 0).astype(np.uint8)


def load_train_val(
    root: Path | str,
    val_ratio: float = VAL_RATIO,
    seed: int = SEED,
) -> tuple[np.ndarray, np.ndarray]:
    """Charge train/good et retourne (X_train, X_val) en float32 [0, 1].

    Split fixe reproductible : val_ratio des images saines → validation.
    """
    root = Path(root)
    paths = sorted((root / "train" / "good").glob("*.png"))
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(paths))
    n_val = max(1, int(len(paths) * val_ratio))
    val_idx, train_idx = idx[:n_val], idx[n_val:]

    X_train = np.stack([_load_image(paths[i]) for i in train_idx])
    X_val = np.stack([_load_image(paths[i]) for i in val_idx])
    return X_train, X_val


def load_test(root: Path | str) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Charge toutes les images de test (saines + défauts).

    Returns:
        images  : (N, 128, 128, 3) float32
        labels  : (N,) int — 0 = saine, 1 = défaut
        classes : liste de str — classe par image ("good", "broken_large", ...)
    """
    root = Path(root)
    images, labels, classes = [], [], []
    for cls_dir in sorted((root / "test").iterdir()):
        cls = cls_dir.name
        label = 0 if cls == "good" else 1
        for p in sorted(cls_dir.glob("*.png")):
            images.append(_load_image(p))
            labels.append(label)
            classes.append(cls)
    return np.stack(images), np.array(labels, dtype=np.int8), classes


def load_masks(root: Path | str) -> dict[tuple[str, str], np.ndarray]:
    """Charge les masques de vérité terrain.

    Returns:
        dict (defect_class, image_stem) → masque binaire (128, 128) uint8
    """
    root = Path(root)
    masks: dict[tuple[str, str], np.ndarray] = {}
    for cls_dir in sorted((root / "ground_truth").iterdir()):
        cls = cls_dir.name
        for p in sorted(cls_dir.glob("*_mask.png")):
            stem = p.stem.replace("_mask", "")
            masks[(cls, stem)] = _load_mask(p)
    return masks
