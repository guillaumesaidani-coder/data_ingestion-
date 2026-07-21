"""Auto-encodeur convolutionnel pour détection d'anomalies (MVTec AD).

Architecture dynamique — nombre de couches déterminé par le tuple `filters` :
  Encodeur : N × Conv2D(stride=2, relu)        256×256 → ... → goulot
  Décodeur : N × Conv2DTranspose(stride=2)     goulot → ... → 256×256
  Sortie   : sigmoid → [0, 1]  (compatible MSE/SSIM sur images non standardisées)

Exemples de ratio de compression (256×256×3 = 196 608 valeurs) :
  filters=(32, 64, 128)     → 32×32×128 = 131 072  → ratio ≈  1.5  ⚠ quasi-identité
  filters=(32, 64, 128, 64) → 16×16×64  =  16 384  → ratio ≈ 12.0  ✓ compact
"""

from __future__ import annotations

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


def build_autoencoder(
    input_shape: tuple[int, int, int] = (256, 256, 3),
    filters: tuple[int, ...] = (32, 64, 128),
) -> keras.Model:
    """Construit un auto-encodeur avec N couches conv / N couches déconv.

    Le décodeur est le miroir de l'encodeur : les filtres sont inversés.
    La dernière couche du décodeur a `input_shape[2]` filtres + sigmoid.

    Args:
        input_shape : (H, W, C) — ex. (256, 256, 3)
        filters     : filtres par couche encodeur — ex. (32, 64, 128, 64)
                      Le dernier élément est la taille du goulot d'étranglement.
    """
    inp = keras.Input(shape=input_shape, name="input")
    x = inp

    # ── Encodeur ──────────────────────────────────────────────────────────────
    for i, f in enumerate(filters):
        x = layers.Conv2D(f, 3, strides=2, activation="relu",
                          padding="same", name=f"enc{i + 1}")(x)

    # ── Décodeur — miroir de l'encodeur ───────────────────────────────────────
    dec_filters = list(reversed(filters[:-1]))           # ex. [64, 32] pour (32,64,128)
    for i, f in enumerate(dec_filters):
        x = layers.Conv2DTranspose(f, 3, strides=2, activation="relu",
                                   padding="same", name=f"dec{i + 1}")(x)
    x = layers.Conv2DTranspose(input_shape[2], 3, strides=2, activation="sigmoid",
                               padding="same", name=f"dec{len(dec_filters) + 1}")(x)

    return keras.Model(inp, x, name="autoencoder")


def compression_ratio(model: keras.Model) -> float:
    """Ratio nb_valeurs_entrée / nb_valeurs_goulot d'étranglement.

    Un ratio proche de 1 → risque d'identité (le modèle apprend à tout copier).
    Un ratio ≥ 4 force une représentation vraiment compacte.
    Fonctionne quel que soit le nombre de couches encodeur.
    """
    h, w, c = model.input_shape[1], model.input_shape[2], model.input_shape[3]
    enc_layers = sorted(
        [l for l in model.layers if l.name.startswith("enc")],
        key=lambda l: int(l.name[3:]),
    )
    bottleneck_out = enc_layers[-1].output
    bh, bw, bc = bottleneck_out.shape[1], bottleneck_out.shape[2], bottleneck_out.shape[3]
    return (h * w * c) / (bh * bw * bc)


def ssim_loss(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
    """1 − SSIM moyen sur le batch. Plus sensible aux altérations de texture que MSE."""
    return 1.0 - tf.reduce_mean(tf.image.ssim(y_true, y_pred, max_val=1.0))


def mse_ssim_loss(alpha: float = 0.8):
    """Perte combinée : alpha × MSE + (1 − alpha) × (1 − SSIM)."""
    def _loss(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
        mse  = tf.reduce_mean(tf.square(y_true - y_pred))
        ssim = 1.0 - tf.reduce_mean(tf.image.ssim(y_true, y_pred, max_val=1.0))
        return alpha * mse + (1.0 - alpha) * ssim
    _loss.__name__ = f"mse{int(alpha * 100)}_ssim{int((1 - alpha) * 100)}"
    return _loss
