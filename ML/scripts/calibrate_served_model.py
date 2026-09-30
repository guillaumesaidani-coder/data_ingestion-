#!/usr/bin/env python
"""Calibre le modèle servi sans le réentraîner : mêmes étapes (imputer +
XGBoost, arbres identiques), dans un CalibratedPipeline dont
`predict_proba` passe par la correction de Platt apprise en GroupKFold
par machine (`modeling.train.fit_calibration`, les hyperparamètres du
modèle servi). Classement, PR-AUC et alertes inchangés — vérifié ici
avant d'écrire, sinon le script s'arrête.

Un réentraînement change le modèle lui-même et passe par l'arbitrage
champion/challenger ; ce script ne change que l'échelle du score. Le
fichier produit a un nouveau `model_version` : enchaîner `make
model-card` (runbook, « Après tout changement de model.joblib »).

Usage :
    uv run --frozen python scripts/calibrate_served_model.py
"""

import argparse
import sys
from pathlib import Path

import joblib
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from indusense.config import get_engine, get_model_path, get_model_version
from indusense.modeling.dataset import load_gold_dataset
from indusense.modeling.pipeline import CalibratedPipeline, predict_alert
from indusense.modeling.train import evaluate, fit_calibration

XGB_PARAMS = ["n_estimators", "max_depth", "learning_rate", "subsample", "colsample_bytree",
              "min_child_weight", "reg_alpha", "reg_lambda", "random_state", "verbosity"]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--model", type=Path, default=get_model_path())
    args = parser.parse_args()

    served = joblib.load(args.model)
    if isinstance(served, CalibratedPipeline):
        raise SystemExit(f"{args.model} est déjà calibré (seuil {served.threshold_:.4f})")

    gold = load_gold_dataset(get_engine())
    xgb = served.named_steps["model"].get_params()
    params = {k: xgb[k] for k in XGB_PARAMS if xgb.get(k) is not None}
    slope, intercept = fit_calibration(gold.X_tv, gold.y_tv, gold.groups, params)

    calibrated = CalibratedPipeline(served.steps).set_calibration(slope, intercept)
    before = evaluate(served, gold.X_tv, gold.y_tv, gold.X_test, gold.y_test)
    after = evaluate(calibrated, gold.X_tv, gold.y_tv, gold.X_test, gold.y_test)
    same_alerts = np.array_equal(predict_alert(served, gold.X_test), predict_alert(calibrated, gold.X_test))
    if not same_alerts or before["pr_auc_test"] != after["pr_auc_test"]:
        raise SystemExit("La calibration change les alertes ou le classement : fichier non écrit")

    print(f"Modèle servi {get_model_version(args.model)} — Platt : pente {slope:.4f}, ordonnée {intercept:.4f}")
    print(f"  seuil calibré {calibrated.threshold_:.4f} (score brut 0,5), alertes test identiques : "
          f"{after['tp'] + after['fp']}")
    print(f"  Brier test {before['brier_test']} -> {after['brier_test']}, "
          f"PR-AUC test {after['pr_auc_test']} (inchangée)")
    joblib.dump(calibrated, args.model)
    print(f"Écrit : {args.model} (model_version {get_model_version(args.model)})")
    print("Suite : make model-card, puis dvc add artifacts/models/model.joblib")


if __name__ == "__main__":
    main()
