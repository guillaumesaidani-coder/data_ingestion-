"""Entraînement + évaluation du modèle tabulaire retenu (b11-gkf).

B11_PARAMS extrait de generate_model_card.py (BEST_PARAMS_BASE — figés
depuis la recherche Optuna de TP11.ipynb). train_and_evaluate() = fit +
evaluate() ; evaluate() seul mesure un modèle déjà entraîné (la model
card décrit le modèle servi, elle ne ré-entraîne plus).
"""

from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline

from indusense.modeling.pipeline import build_xgb_pipeline

RANDOM_STATE = 42

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


def train_and_evaluate(X_tv, y_tv, X_test, y_test, params: dict) -> tuple[Pipeline, dict]:
    pipe = build_xgb_pipeline(params)
    pipe.fit(X_tv, y_tv)
    return pipe, evaluate(pipe, X_tv, y_tv, X_test, y_test)


def evaluate(pipe: Pipeline, X_tv, y_tv, X_test, y_test) -> dict:
    """Métriques d'un modèle déjà entraîné (seuil 0,5). Sert aussi à
    mesurer le modèle servi sans le ré-entraîner (model card)."""
    y_prob_test = pipe.predict_proba(X_test)[:, 1]
    y_pred_test = pipe.predict(X_test)
    y_prob_tv = pipe.predict_proba(X_tv)[:, 1]

    tn, fp, fn, tp = confusion_matrix(y_test, y_pred_test).ravel()

    metrics = {
        "pr_auc_train": round(float(average_precision_score(y_tv, y_prob_tv)), 4),
        "pr_auc_test": round(float(average_precision_score(y_test, y_prob_test)), 4),
        "roc_auc_test": round(float(roc_auc_score(y_test, y_prob_test)), 4),
        "f1_test": round(float(f1_score(y_test, y_pred_test, zero_division=0)), 4),
        "precision_test": round(float(precision_score(y_test, y_pred_test, zero_division=0)), 4),
        "recall_test": round(float(recall_score(y_test, y_pred_test, zero_division=0)), 4),
        "tp": int(tp),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "threshold": 0.5,
    }
    return metrics
