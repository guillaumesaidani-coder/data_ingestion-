"""Score d'anomalie, calibration du seuil, métriques et visualisation."""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sklearn.metrics import roc_auc_score, confusion_matrix


def reconstruction_errors(
    model,
    X: np.ndarray,
    batch_size: int = 32,
) -> np.ndarray:
    """MSE de reconstruction par image — shape (N,) float32.

    Args:
        X : (N, H, W, 3) float32 dans [0, 1]
    """
    X_pred = model.predict(X, batch_size=batch_size, verbose=0)
    return np.mean((X - X_pred) ** 2, axis=(1, 2, 3)).astype(np.float32)


def calibrate_threshold(
    errors_val: np.ndarray,
    method: str = "percentile",
    percentile: float = 99.0,
    k: float = 3.0,
) -> float:
    """Seuil calibré sur les images saines de validation uniquement.

    Args:
        method      : 'percentile' (centile) ou 'sigma' (moyenne + k·σ)
        percentile  : centile utilisé si method='percentile' (ex. 99)
        k           : nombre d'écarts-types si method='sigma'

    Returns:
        seuil scalaire float

    Note:
        Abaisser le seuil → plus de rappel, plus de fausses alertes.
        Le choix dépend du coût métier d'une panne manquée vs d'une fausse alarme.
    """
    if method == "percentile":
        return float(np.percentile(errors_val, percentile))
    elif method == "sigma":
        return float(errors_val.mean() + k * errors_val.std())
    raise ValueError(f"method doit être 'percentile' ou 'sigma', reçu: {method!r}")


def pixel_error_map(model, image: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Carte d'erreur MSE par pixel et reconstruction.

    Args:
        image : (H, W, 3) float32 dans [0, 1]

    Returns:
        reconstruction : (H, W, 3) float32
        error_map      : (H, W)    float32 — MSE par pixel (moyenne sur canaux)
    """
    reconstruction = model.predict(image[np.newaxis], verbose=0)[0]
    error_map = np.mean((image - reconstruction) ** 2, axis=-1).astype(np.float32)
    return reconstruction, error_map


def plot_heatmap(
    original: np.ndarray,
    reconstruction: np.ndarray,
    error_map: np.ndarray,
    mask_gt: np.ndarray | None = None,
    title: str = "",
    save_path: str | None = None,
) -> None:
    """Affiche : original | reconstruction | heatmap erreur | masque GT (si fourni)."""
    n_cols = 4 if mask_gt is not None else 3
    fig, axes = plt.subplots(1, n_cols, figsize=(4.2 * n_cols, 4))
    if title:
        fig.suptitle(title, fontsize=11, fontweight="bold")

    axes[0].imshow(np.clip(original, 0, 1))
    axes[0].set_title("Original", fontsize=9)
    axes[0].axis("off")

    axes[1].imshow(np.clip(reconstruction, 0, 1))
    axes[1].set_title("Reconstruction", fontsize=9)
    axes[1].axis("off")

    im = axes[2].imshow(error_map, cmap="hot", interpolation="none")
    axes[2].set_title("Erreur pixel (MSE)", fontsize=9)
    axes[2].axis("off")
    plt.colorbar(im, ax=axes[2], fraction=0.046, pad=0.04)

    if mask_gt is not None:
        axes[3].imshow(mask_gt, cmap="gray")
        axes[3].set_title("Masque GT", fontsize=9)
        axes[3].axis("off")

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=120, bbox_inches="tight")
    plt.show()


def plot_score_histogram(
    errors_val: np.ndarray,
    errors_test_good: np.ndarray,
    errors_test_defect: np.ndarray,
    threshold: float,
    save_path: str | None = None,
) -> None:
    """Histogramme : saines val / saines test / défauts, avec ligne de seuil."""
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.hist(errors_val,         bins=25, alpha=0.6, color="#16A34A", label="Saines — validation")
    ax.hist(errors_test_good,   bins=20, alpha=0.6, color="#22C55E", label="Saines — test")
    ax.hist(errors_test_defect, bins=30, alpha=0.7, color="#DC2626", label="Défauts — test")
    ax.axvline(threshold, color="#1D4ED8", lw=2, ls="--", label=f"Seuil = {threshold:.5f}")
    ax.set_xlabel("Erreur de reconstruction (MSE)", fontsize=10)
    ax.set_ylabel("Nb images", fontsize=10)
    ax.set_title("Distribution des scores d'anomalie", fontsize=11, fontweight="bold")
    ax.legend(fontsize=9)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=120, bbox_inches="tight")
    plt.show()


def evaluate(
    y_true: np.ndarray,
    scores: np.ndarray,
    threshold: float,
) -> dict:
    """AUROC, matrice de confusion et métriques dérivées.

    Args:
        y_true    : (N,) int — 0=sain, 1=défaut
        scores    : (N,) float — erreur de reconstruction
        threshold : seuil de décision

    Returns:
        dict avec auroc, tp, tn, fp, fn, recall, precision, specificity
    """
    auroc  = roc_auc_score(y_true, scores)
    y_pred = (scores >= threshold).astype(int)
    cm     = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    return {
        "auroc":       float(auroc),
        "threshold":   float(threshold),
        "tp": int(tp), "tn": int(tn), "fp": int(fp), "fn": int(fn),
        "recall":      tp / (tp + fn) if (tp + fn) > 0 else 0.0,
        "precision":   tp / (tp + fp) if (tp + fp) > 0 else 0.0,
        "specificity": tn / (tn + fp) if (tn + fp) > 0 else 0.0,
    }


def plot_confusion_matrix(metrics: dict, save_path: str | None = None) -> None:
    """Affiche la matrice de confusion à partir du dict retourné par evaluate()."""
    cm_arr = np.array([[metrics["tn"], metrics["fp"]],
                       [metrics["fn"], metrics["tp"]]])
    fig, ax = plt.subplots(figsize=(4.5, 3.8))
    im = ax.imshow(cm_arr, cmap="Blues")
    ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
    ax.set_xticklabels(["Prédit Sain", "Prédit Défaut"])
    ax.set_yticklabels(["Réel Sain", "Réel Défaut"])
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(cm_arr[i, j]), ha="center", va="center",
                    fontsize=14, fontweight="bold",
                    color="white" if cm_arr[i, j] > cm_arr.max() / 2 else "black")
    ax.set_title(
        f"Matrice de confusion\nAUROC = {metrics['auroc']:.3f} | "
        f"Rappel = {metrics['recall']:.2%} | Spécificité = {metrics['specificity']:.2%}",
        fontsize=9
    )
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=120, bbox_inches="tight")
    plt.show()
