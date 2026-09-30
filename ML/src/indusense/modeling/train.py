"""Entraînement + évaluation du modèle tabulaire retenu (b11-gkf).

B11_PARAMS extrait de generate_model_card.py (BEST_PARAMS_BASE — figés
depuis la recherche Optuna de TP11.ipynb). train_and_evaluate() = fit +
calibration (si `groups`) + evaluate() ; evaluate() seul mesure un
modèle déjà entraîné (la model card décrit le modèle servi, elle ne
ré-entraîne plus).
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline

from indusense.modeling.pipeline import alert_threshold, build_xgb_pipeline, predict_alert

RANDOM_STATE = 42
CALIBRATION_FOLDS = 5  # même découpage que l'objectif Optuna de TP11

B11_PARAMS = {
    "n_estimators": 287,
    "max_depth": 9,
    "learning_rate": 0.02887139049912187,
    "subsample": 0.8252019146348859,
    "colsample_bytree": 0.5736306798333981,
    "min_child_weight": 11,
    "reg_alpha": 0.01918528344873483,
    "reg_lambda": 0.6228756133555158,
    "random_state": RANDOM_STATE,
    "verbosity": 0,
}


def compute_scale_pos_weight(y) -> float:
    return round(float((y == 0).sum() / (y == 1).sum()), 2)


def fit_calibration(X, y, groups, params: dict, n_splits: int = CALIBRATION_FOLDS) -> tuple[float, float]:
    """Pente et ordonnée de Platt apprises sur des marges hors échantillon :
    GroupKFold par machine (chaque marge vient d'un modèle qui n'a jamais
    vu sa machine, comme en production pour une machine neuve),
    `scale_pos_weight` recalculé sur chaque fold. Jamais sur le test."""
    y = pd.Series(y).astype(int).reset_index(drop=True)
    X = X.reset_index(drop=True)
    groups = pd.Series(groups).reset_index(drop=True)
    margins = np.zeros(len(y))
    folds = GroupKFold(n_splits=min(n_splits, groups.nunique()))
    for train_idx, val_idx in folds.split(X, y, groups):
        pipe = build_xgb_pipeline(
            {**params, "scale_pos_weight": compute_scale_pos_weight(y.iloc[train_idx])}
        )
        pipe.fit(X.iloc[train_idx], y.iloc[train_idx])
        margins[val_idx] = pipe[-1].predict(
            pipe[:-1].transform(X.iloc[val_idx]), output_margin=True
        )
    platt = LogisticRegression(C=1e6).fit(margins.reshape(-1, 1), y)
    return float(platt.coef_[0, 0]), float(platt.intercept_[0])


def train_and_evaluate(
    X_tv, y_tv, X_test, y_test, params: dict, groups=None
) -> tuple[Pipeline, dict]:
    """Avec `groups` (machine_id de chaque ligne de X_tv), le modèle
    produit est un CalibratedPipeline — le cas de production. Sans, un
    pipeline brut (expériences de fuite de la certification, tests)."""
    pipe = build_xgb_pipeline(params, calibrated=groups is not None)
    pipe.fit(X_tv, y_tv)
    if groups is not None:
        pipe.set_calibration(*fit_calibration(X_tv, y_tv, groups, params))
    return pipe, evaluate(pipe, X_tv, y_tv, X_test, y_test)


def evaluate(pipe: Pipeline, X_tv, y_tv, X_test, y_test) -> dict:
    """Métriques d'un modèle déjà entraîné, au seuil d'alerte du modèle
    (0,5 brut, ou son équivalent calibré). Sert aussi à mesurer le modèle
    servi sans le ré-entraîner (model card)."""
    y_prob_test = pipe.predict_proba(X_test)[:, 1]
    y_pred_test = predict_alert(pipe, X_test)
    y_prob_tv = pipe.predict_proba(X_tv)[:, 1]

    tn, fp, fn, tp = confusion_matrix(y_test, y_pred_test).ravel()

    metrics = {
        "pr_auc_train": round(float(average_precision_score(y_tv, y_prob_tv)), 4),
        "pr_auc_test": round(float(average_precision_score(y_test, y_prob_test)), 4),
        "roc_auc_test": round(float(roc_auc_score(y_test, y_prob_test)), 4),
        "f1_test": round(float(f1_score(y_test, y_pred_test, zero_division=0)), 4),
        "precision_test": round(float(precision_score(y_test, y_pred_test, zero_division=0)), 4),
        "recall_test": round(float(recall_score(y_test, y_pred_test, zero_division=0)), 4),
        "brier_test": round(float(brier_score_loss(y_test, y_prob_test)), 4),
        "tp": int(tp),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "threshold": round(alert_threshold(pipe), 4),
    }
    return metrics
