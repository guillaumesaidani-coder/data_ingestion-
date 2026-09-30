#!/usr/bin/env python
"""Seuil d'alerte et calibration du modèle b11-gkf. Le seuil 0,5 de
base_connaissance.yaml est le défaut de `predict`, jamais choisi, et
`predict_proba` n'est pas calibré (scale_pos_weight ≈ 27 gonfle les
scores). Ce script ne change rien au modèle servi, il donne de quoi
décider :

1. probabilités hors échantillon sur train+validation, une machine
   cachée par fold (hyperparamètres b11) : c'est sur elles qu'on choisit
   le seuil et qu'on apprend la calibration, jamais sur le test ;
2. seuils candidats (0,5 actuel, F1 max, rappel visé 80/90/95 %) :
   rappel, précision, alertes par jour sur la flotte, puis les mêmes
   seuils appliqués au modèle de prod sur le test ;
3. calibration isotonique apprise sur (1), appliquée au modèle de prod
   sur le test : Brier score et courbe de fiabilité avant / après.

Les métriques sont horaires, comme le reste du projet : une panne donne
jusqu'à 24 lignes positives, une alerte = une heure-machine au-dessus
du seuil.

15 entraînements, environ 2 minutes sur CPU.

Usage :
    uv run --frozen python scripts/analyze_threshold_calibration.py
    # sans Postgres : Gold exporté par DVC
    uv run --frozen python scripts/analyze_threshold_calibration.py \\
        --gold-csv data/gold/gold_dataset.csv
"""

import argparse
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import brier_score_loss, precision_recall_curve

ML_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ML_DIR / "src"))

from indusense.config import get_engine
from indusense.modeling.dataset import LEAKAGE_COLS, TARGET
from indusense.modeling.heterogeneity import leave_one_group_out_proba
from indusense.modeling.train import B11_PARAMS
from indusense.scoring import model_feature_cols

CURRENT_THRESHOLD = 0.5
TARGET_RECALLS = [0.80, 0.90, 0.95]
RELIABILITY_BINS = [0, 0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0]


def at_threshold(y: pd.Series, proba: pd.Series, threshold: float, n_days: float) -> dict:
    alert = proba >= threshold
    tp = int((alert & y).sum())
    return {
        "seuil": round(float(threshold), 4),
        "rappel": round(tp / int(y.sum()), 4),
        "precision": round(tp / int(alert.sum()), 4) if alert.any() else np.nan,
        "alertes_par_jour": round(int(alert.sum()) / n_days, 1),
    }


def candidate_thresholds(y: pd.Series, proba: pd.Series) -> dict[str, float]:
    precision, recall, thresholds = precision_recall_curve(y, proba)
    precision, recall = precision[:-1], recall[:-1]  # alignés sur thresholds
    f1 = 2 * precision * recall / np.clip(precision + recall, 1e-12, None)
    out = {"actuel_0.5": CURRENT_THRESHOLD, "f1_max": float(thresholds[f1.argmax()])}
    for r in TARGET_RECALLS:
        # seuil le plus haut qui atteint encore le rappel visé
        out[f"rappel_{r:.0%}"] = float(thresholds[recall >= r].max())
    return out


def n_days(df: pd.DataFrame) -> float:
    return (df["window_start"].max() - df["window_start"].min()).total_seconds() / 86400


def reliability(y: pd.Series, proba: pd.Series) -> pd.DataFrame:
    bins = pd.cut(proba, RELIABILITY_BINS, include_lowest=True)
    return pd.DataFrame({"lignes": y.groupby(bins, observed=False).size(),
                         "proba_moyenne": proba.groupby(bins, observed=False).mean(),
                         "taux_panne_observe": y.groupby(bins, observed=False).mean()})


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--gold-csv", type=Path, help="Gold exporté (sinon Postgres)")
    parser.add_argument("--model", type=Path, default=ML_DIR / "artifacts/models/model.joblib")
    parser.add_argument("--out", type=Path, default=ML_DIR / "artifacts/threshold_calibration.json")
    args = parser.parse_args()

    if args.gold_csv:
        df = pd.read_csv(args.gold_csv, parse_dates=["window_start"])
    else:
        df = pd.read_sql("SELECT * FROM gold_machine_hourly_feature", get_engine(),
                         parse_dates=["window_start"])
    df[TARGET] = df[TARGET].astype(bool)
    prod = joblib.load(args.model)
    feats = model_feature_cols(prod, [c for c in df.columns if c not in LEAKAGE_COLS])
    report = {}

    tv = df[df["split_set"].isin(["train", "validation"])].copy()
    test = df[df["split_set"] == "test"]

    # 1. Probabilités hors échantillon (une machine cachée par fold)
    print("  validation croisée « machine cachée »...", flush=True)
    oof = leave_one_group_out_proba(tv, feats, TARGET, "machine_id", B11_PARAMS)
    test_proba = pd.Series(prod.predict_proba(test.reindex(columns=feats))[:, 1], index=test.index)

    # 2. Seuils : choisis sur la CV, vérifiés sur le test
    thresholds = candidate_thresholds(tv[TARGET], oof)
    rows = []
    for name, t in thresholds.items():
        cv = at_threshold(tv[TARGET], oof, t, n_days(tv))
        te = at_threshold(test[TARGET], test_proba, t, n_days(test))
        rows.append({"candidat": name, "seuil": cv["seuil"],
                     **{f"cv_{k}": v for k, v in cv.items() if k != "seuil"},
                     **{f"test_{k}": v for k, v in te.items() if k != "seuil"}})
    table = pd.DataFrame(rows).set_index("candidat")
    print("\n== 2. Seuils candidats (choisis sur la CV, appliqués au modèle de prod sur le test) ==\n",
          table.to_string(),
          f"\n pannes-heures : {int(tv[TARGET].sum())} en CV sur {n_days(tv):.0f} j, "
          f"{int(test[TARGET].sum())} en test sur {n_days(test):.0f} j\n")
    report["seuils"] = table.reset_index().to_dict("records")

    # 3. Calibration isotonique apprise sur la CV
    iso = IsotonicRegression(out_of_bounds="clip").fit(oof, tv[TARGET])
    test_cal = pd.Series(iso.predict(test_proba), index=test.index)
    brier = {"brut": brier_score_loss(test[TARGET], test_proba),
             "calibre": brier_score_loss(test[TARGET], test_cal),
             "reference_taux_constant": brier_score_loss(
                 test[TARGET], np.full(len(test), tv[TARGET].mean()))}
    print("== 3. Calibration (test) ==\n Brier :", {k: round(v, 4) for k, v in brier.items()})
    for label, p in [("brut", test_proba), ("calibre", test_cal)]:
        rel = reliability(test[TARGET], p)
        print(f"\n fiabilité {label} :\n", rel.round(3).to_string())
        report[f"fiabilite_{label}"] = rel.round(4).reset_index(names="tranche").astype(
            {"tranche": str}).to_dict("records")
    t_cal = float(iso.predict([CURRENT_THRESHOLD])[0])
    print(f"\n un score brut de {CURRENT_THRESHOLD} correspond à une probabilité calibrée de {t_cal:.3f}")
    report["brier_test"] = {k: round(v, 4) for k, v in brier.items()}
    report["proba_calibree_au_seuil_actuel"] = round(t_cal, 4)

    args.out.write_text(json.dumps(report, indent=1, ensure_ascii=False, default=str), encoding="utf-8")
    print(f"\nRapport : {args.out}")


if __name__ == "__main__":
    main()
