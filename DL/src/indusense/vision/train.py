"""Entraînement de l'auto-encodeur avec suivi MLflow optionnel."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import tensorflow as tf
from tensorflow import keras

try:
    import mlflow
    import mlflow.keras
    _MLFLOW = True
except ImportError:
    _MLFLOW = False


def train(
    model: keras.Model,
    X_train: np.ndarray,
    X_val: np.ndarray,
    pipeline=None,
    epochs: int = 50,
    batch_size: int = 16,
    learning_rate: float = 1e-3,
    loss="mse",
    run_name: str = "autoencoder",
    checkpoint_path: Path | str | None = "checkpoints/ae_best.keras",
    use_mlflow: bool = True,
) -> keras.callbacks.History:
    """Entraîne l'auto-encodeur (cible = entrée) et retourne l'historique.

    Args:
        pipeline       : pipeline Albumentations — si fourni, augmente X_train avant fit.
        checkpoint_path: chemin pour sauvegarder le meilleur modèle (val_loss).
        use_mlflow     : log params/métriques dans MLflow si disponible.

    Notes:
        - EarlyStopping patience=10, ReduceLROnPlateau patience=5.
        - X_train et X_val doivent être en [0, 1] (pas standardisés) pour
          être cohérents avec la sortie sigmoid du décodeur.
    """
    loss_name = loss if isinstance(loss, str) else getattr(loss, "__name__", str(loss))
    model.compile(optimizer=keras.optimizers.Adam(learning_rate), loss=loss)

    callbacks = [
        keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True, verbose=1),
        keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=5, verbose=1, min_lr=1e-6),
    ]

    if checkpoint_path is not None:
        Path(checkpoint_path).parent.mkdir(parents=True, exist_ok=True)
        callbacks.append(
            keras.callbacks.ModelCheckpoint(
                str(checkpoint_path), save_best_only=True, monitor="val_loss", verbose=0
            )
        )

    # Augmentation optionnelle
    if pipeline is not None:
        from indusense.vision.augment import augment_batch
        np.random.seed(42)
        X_aug = augment_batch(X_train, pipeline, n_aug=1)
        X_fit = np.concatenate([X_train, X_aug], axis=0)
    else:
        X_fit = X_train

    if use_mlflow and _MLFLOW:
        mlflow.start_run(run_name=run_name)
        mlflow.log_params({
            "epochs": epochs,
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            "loss": loss_name,
            "n_train": len(X_fit),
            "n_val": len(X_val),
            "augmented": pipeline is not None,
        })

    history = model.fit(
        X_fit, X_fit,
        validation_data=(X_val, X_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks,
        verbose=1,
    )

    if use_mlflow and _MLFLOW:
        mlflow.log_metrics({
            "best_val_loss":   min(history.history["val_loss"]),
            "final_train_loss": history.history["loss"][-1],
            "epochs_run":      len(history.history["loss"]),
        })
        if checkpoint_path and Path(checkpoint_path).exists():
            mlflow.log_artifact(str(checkpoint_path))
        mlflow.end_run()

    return history
