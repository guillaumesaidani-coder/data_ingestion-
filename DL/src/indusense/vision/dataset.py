"""Chargement et préparation du dataset MVTec AD (catégorie bottle).

Conventions :
  - Taille cible : 256×256 (suffisante pour préserver les défauts visuels)
  - Redimensionnement : padding centré (letterbox) — ratio préservé
  - Interpolation images : LANCZOS  |  masques : NEAREST (labels binaires)
  - Normalisation étape 1 : mise à l'échelle [0, 1] par division par 255
  - Normalisation étape 2 : standardisation (x − μ) / σ par canal,
                            μ et σ calculés sur le train uniquement
"""

import json
from pathlib import Path

import numpy as np
from PIL import Image

IMG_SIZE: tuple[int, int] = (256, 256)
VAL_RATIO: float = 0.2
SEED: int = 42


def _resize_pad(pil_img: Image.Image, size: tuple[int, int], interp) -> Image.Image:
    """Redimensionne avec préservation du ratio + padding centré (letterbox)."""
    target_w, target_h = size
    w, h = pil_img.size
    scale = min(target_w / w, target_h / h)
    new_w, new_h = int(w * scale), int(h * scale)

    resized = pil_img.resize((new_w, new_h), interp)
    canvas = Image.new(pil_img.mode, size, 0)
    left = (target_w - new_w) // 2
    top = (target_h - new_h) // 2
    canvas.paste(resized, (left, top))
    return canvas


def _load_image(path: Path, size: tuple[int, int] = IMG_SIZE) -> np.ndarray:
    """Charge une image RGB, redimensionne avec padding, normalise dans [0, 1]."""
    pil = Image.open(path).convert("RGB")
    pil = _resize_pad(pil, (size[1], size[0]), Image.LANCZOS)
    return np.array(pil, dtype=np.float32) / 255.0


def _load_mask(path: Path, size: tuple[int, int] = IMG_SIZE) -> np.ndarray:
    """Charge un masque binaire, redimensionne avec NEAREST (pas d'interpolation)."""
    pil = Image.open(path).convert("L")
    pil = _resize_pad(pil, (size[1], size[0]), Image.NEAREST)
    return (np.array(pil) > 0).astype(np.uint8)


def load_train_val(
    root: Path | str,
    val_ratio: float = VAL_RATIO,
    seed: int = SEED,
    size: tuple[int, int] = IMG_SIZE,
) -> tuple[np.ndarray, np.ndarray]:
    """Charge train/good et retourne (X_train, X_val) en float32 [0, 1].

    Split fixe reproductible : val_ratio des images saines → validation.
    Redimensionnement : padding centré 256×256, LANCZOS.
    """
    root = Path(root)
    paths = sorted((root / "train" / "good").glob("*.png"))
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(paths))
    n_val = max(1, int(len(paths) * val_ratio))
    val_idx, train_idx = idx[:n_val], idx[n_val:]

    X_train = np.stack([_load_image(paths[i], size) for i in train_idx])
    X_val   = np.stack([_load_image(paths[i], size) for i in val_idx])
    return X_train, X_val


def load_test(
    root: Path | str,
    size: tuple[int, int] = IMG_SIZE,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Charge toutes les images de test (saines + défauts).

    Returns:
        images  : (N, H, W, 3) float32
        labels  : (N,) int8 — 0 = saine, 1 = défaut
        classes : liste de str — classe par image ("good", "broken_large", ...)
    """
    root = Path(root)
    images, labels, classes = [], [], []
    for cls_dir in sorted((root / "test").iterdir()):
        cls = cls_dir.name
        label = 0 if cls == "good" else 1
        for p in sorted(cls_dir.glob("*.png")):
            images.append(_load_image(p, size))
            labels.append(label)
            classes.append(cls)
    return np.stack(images), np.array(labels, dtype=np.int8), classes


def compute_stats(X_train: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Calcule moyenne et écart-type par canal RGB sur le set d'entraînement.

    Args:
        X_train : (N, H, W, 3) float32 dans [0, 1]

    Returns:
        mean : (3,) — moyenne par canal
        std  : (3,) — écart-type par canal (jamais nul)
    """
    mean = X_train.mean(axis=(0, 1, 2))
    std  = X_train.std(axis=(0, 1, 2))
    std  = np.where(std < 1e-8, 1.0, std)  # évite la division par zéro
    return mean, std


def standardize(
    X: np.ndarray,
    mean: np.ndarray,
    std: np.ndarray,
) -> np.ndarray:
    """Applique la standardisation (x − mean) / std, par canal.

    Les stats (mean, std) doivent être calculées sur X_train uniquement
    via compute_stats(), puis réutilisées telles quelles sur val et test.

    Args:
        X    : (N, H, W, 3) float32
        mean : (3,) — issu de compute_stats(X_train)
        std  : (3,) — issu de compute_stats(X_train)

    Returns:
        (N, H, W, 3) float32 standardisé
    """
    return (X - mean) / std


def save_stats(
    mean: np.ndarray,
    std: np.ndarray,
    path: Path | str = "norm_stats.json",
) -> None:
    """Persiste mean et std dans un fichier JSON versionnable.

    À sauvegarder avec le modèle : toute inférence future doit utiliser
    les mêmes stats que celles vues à l'entraînement.
    """
    data = {"mean_rgb": mean.tolist(), "std_rgb": std.tolist()}
    Path(path).write_text(json.dumps(data, indent=2))


def load_stats(path: Path | str = "norm_stats.json") -> tuple[np.ndarray, np.ndarray]:
    """Charge mean et std depuis un fichier JSON sauvegardé par save_stats()."""
    data = json.loads(Path(path).read_text())
    return np.array(data["mean_rgb"], dtype=np.float32), np.array(data["std_rgb"], dtype=np.float32)


def load_masks(
    root: Path | str,
    size: tuple[int, int] = IMG_SIZE,
) -> dict[tuple[str, str], np.ndarray]:
    """Charge les masques de vérité terrain.

    Returns:
        dict (defect_class, image_stem) → masque binaire (H, W) uint8
    """
    root = Path(root)
    masks: dict[tuple[str, str], np.ndarray] = {}
    for cls_dir in sorted((root / "ground_truth").iterdir()):
        cls = cls_dir.name
        for p in sorted(cls_dir.glob("*_mask.png")):
            stem = p.stem.replace("_mask", "")
            masks[(cls, stem)] = _load_mask(p, size)
    return masks
