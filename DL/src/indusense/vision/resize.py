"""Stratégies de redimensionnement pour images industrielles (MVTec AD).

Deux approches :
  - stretch  : redimensionnement direct, ratio non préservé → déformation
  - pad      : scale + padding centré (letterbox), ratio préservé → zones noires
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

INTERP = {
    "nearest":  Image.NEAREST,
    "bilinear": Image.BILINEAR,
    "bicubic":  Image.BICUBIC,
    "lanczos":  Image.LANCZOS,
}


def load_original(path: Path | str) -> np.ndarray:
    """Charge une image à sa résolution originale, normalisée dans [0, 1]."""
    img = Image.open(path).convert("RGB")
    return np.array(img, dtype=np.float32) / 255.0


def resize_stretch(
    img: np.ndarray,
    size: tuple[int, int] = (256, 256),
    method: str = "bilinear",
) -> np.ndarray:
    """Redimensionnement naïf — ratio non préservé (déformation si non carré)."""
    pil = Image.fromarray((img * 255).astype(np.uint8))
    return np.array(pil.resize((size[1], size[0]), INTERP[method]), dtype=np.float32) / 255.0


def resize_pad(
    img: np.ndarray,
    size: tuple[int, int] = (256, 256),
    method: str = "lanczos",
    fill: float = 0.0,
) -> np.ndarray:
    """Redimensionnement avec préservation du ratio + padding centré (letterbox).

    Args:
        img   : (H, W, 3) float32 dans [0, 1]
        size  : (target_H, target_W)
        method: méthode d'interpolation
        fill  : valeur de remplissage pour le padding (0.0 = noir)

    Returns:
        (target_H, target_W, 3) float32 dans [0, 1]
    """
    h, w = img.shape[:2]
    target_h, target_w = size
    scale = min(target_w / w, target_h / h)
    new_w, new_h = int(w * scale), int(h * scale)

    pil = Image.fromarray((img * 255).astype(np.uint8))
    resized = np.array(pil.resize((new_w, new_h), INTERP[method]), dtype=np.uint8)

    canvas = np.full((target_h, target_w, 3), int(fill * 255), dtype=np.uint8)
    top  = (target_h - new_h) // 2
    left = (target_w - new_w) // 2
    canvas[top:top + new_h, left:left + new_w] = resized

    return canvas.astype(np.float32) / 255.0


def compare_interpolations(
    img: np.ndarray,
    size: tuple[int, int] = (256, 256),
) -> dict[str, np.ndarray]:
    """Retourne un dict {method_name: resized_array} pour les 4 méthodes."""
    return {name: resize_stretch(img, size, method=name) for name in INTERP}


def compare_resolutions(
    img: np.ndarray,
    resolutions: list[int] | None = None,
) -> dict[int, np.ndarray]:
    """Redimensionne la même image à différentes résolutions (carrées)."""
    if resolutions is None:
        resolutions = [32, 64, 128, 256]
    return {r: resize_stretch(img, (r, r), method="lanczos") for r in resolutions}
