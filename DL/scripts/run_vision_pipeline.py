"""Pipeline vision bout-en-bout — MVTec AD (bottle).

Orchestre : charger MVTec (+ normaliser) -> [augmenter] -> construire l'auto-encodeur
-> entraîner (+ MLflow + CodeCarbon) -> score/seuil/AUROC -> heatmaps -> sauvegarder.

Ce script ne contient aucune logique métier : il enchaîne les briques de
`indusense.vision`. Toute la logique réutilisable vit dans `src/indusense/vision/`.

Usage:
    uv run python scripts/run_vision_pipeline.py --epochs 30 --img-size 128 --batch-size 32
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

import matplotlib
matplotlib.use("Agg")  # script headless — jamais de fenêtre GUI

import numpy as np
import tensorflow as tf
from tensorflow import keras

import config
from indusense.vision.dataset import load_train_val, load_test
from indusense.vision.model import build_autoencoder, compression_ratio, mse_ssim_loss
from indusense.vision.train import train
from indusense.vision.anomaly import (
    calibrate_threshold,
    evaluate,
    plot_confusion_matrix,
    plot_score_histogram,
    reconstruction_errors,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Pipeline vision bout-en-bout (MVTec bottle)")
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--img-size", type=int, default=config.IMG_SIZE[0])
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--filters", type=int, nargs="+", default=list(config.DEFAULT_FILTERS))
    p.add_argument("--alpha-loss", type=float, default=config.DEFAULT_ALPHA_LOSS)
    p.add_argument("--augment", action="store_true", help="Augmente le train set (Albumentations)")
    p.add_argument("--no-mlflow", action="store_true")
    p.add_argument("--no-carbon", action="store_true")
    p.add_argument("--run-name", type=str, default="vision_pipeline")
    return p.parse_args(argv)


def set_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


def main(argv: list[str] | None = None) -> dict:
    args = parse_args(argv)
    set_seeds(config.RANDOM_SEED)
    size = (args.img_size, args.img_size)

    print(f"[1/6] Chargement MVTec bottle ({size[0]}x{size[1]})...")
    X_train, X_val = load_train_val(
        config.BOTTLE_ROOT, val_ratio=config.VAL_RATIO, seed=config.RANDOM_SEED, size=size
    )
    X_test, y_test, _ = load_test(config.BOTTLE_ROOT, size=size)
    print(f"  Train={len(X_train)}  Val={len(X_val)}  Test={len(X_test)}")

    pipeline = None
    if args.augment:
        from indusense.vision.augment import build_pipeline
        pipeline = build_pipeline()

    print(f"[2/6] Construction de l'auto-encodeur filters={tuple(args.filters)}...")
    model = build_autoencoder(input_shape=(*size, 3), filters=tuple(args.filters))
    loss_fn = mse_ssim_loss(alpha=args.alpha_loss)
    ratio = compression_ratio(model)
    print(f"  {model.count_params():,} paramètres — ratio de compression ~= {ratio:.1f}")

    checkpoint_path = config.CHECKPOINTS_DIR / f"{args.run_name}.keras"

    tracker = None
    if not args.no_carbon:
        from codecarbon import OfflineEmissionsTracker
        tracker = OfflineEmissionsTracker(
            project_name=args.run_name,
            country_iso_code=config.COUNTRY_ISO_CODE,
            output_dir=str(config.EMISSIONS_DIR),
            log_level="error",
            save_to_file=True,
        )
        tracker.start()

    print(f"[3/6] Entraînement ({args.epochs} epochs, batch={args.batch_size})...")
    history = train(
        model, X_train, X_val,
        pipeline=pipeline,
        epochs=args.epochs,
        batch_size=args.batch_size,
        loss=loss_fn,
        run_name=args.run_name,
        checkpoint_path=checkpoint_path,
        use_mlflow=not args.no_mlflow,
    )

    emissions_gco2eq, energy_wh = None, None
    if tracker is not None:
        emissions_kg = tracker.stop()
        data = tracker.final_emissions_data
        emissions_gco2eq = emissions_kg * 1000
        energy_wh = data.energy_consumed * 1000

    print("[4/6] Score d'anomalie, calibration du seuil, AUROC...")
    best_model = keras.models.load_model(
        checkpoint_path, custom_objects={loss_fn.__name__: loss_fn}
    )
    errors_val = reconstruction_errors(best_model, X_val, batch_size=args.batch_size)
    errors_test = reconstruction_errors(best_model, X_test, batch_size=args.batch_size)
    threshold = calibrate_threshold(errors_val, method="percentile", percentile=99)
    metrics = evaluate(y_test, errors_test, threshold)
    print(f"  AUROC={metrics['auroc']:.3f}  seuil={threshold:.6f}")

    print("[5/6] Génération des figures...")
    plot_confusion_matrix(
        metrics, save_path=str(config.FIGURES_DIR / f"{args.run_name}_confusion.png")
    )
    plot_score_histogram(
        errors_val, errors_test[y_test == 0], errors_test[y_test == 1], threshold,
        save_path=str(config.FIGURES_DIR / f"{args.run_name}_histogram.png"),
    )

    print("[6/6] Écriture du rapport...")
    report = {
        "run_name": args.run_name,
        "args": {
            "epochs": args.epochs,
            "img_size": args.img_size,
            "batch_size": args.batch_size,
            "filters": list(args.filters),
            "alpha_loss": args.alpha_loss,
            "augment": args.augment,
        },
        "n_params": int(model.count_params()),
        "ratio_compression": ratio,
        "epochs_run": len(history.history["loss"]),
        "best_val_loss": float(min(history.history["val_loss"])),
        "metrics": metrics,
        "emissions_gco2eq": emissions_gco2eq,
        "energy_wh": energy_wh,
        "checkpoint": str(checkpoint_path),
    }
    report_path = config.REPORTS_DIR / f"{args.run_name}_report.json"
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"  Rapport : {report_path}")
    return report


if __name__ == "__main__":
    main()
