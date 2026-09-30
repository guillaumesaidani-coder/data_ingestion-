"""Pipeline scikit-learn standard du projet (imputation médiane + XGBoost),
répétée à l'identique dans TP9.ipynb, TP11.ipynb et generate_model_card.py.

`CalibratedPipeline` garde exactement ces deux étapes (l'explication
SHAP d'explain.py et `model_feature_cols` lisent `named_steps`) et
recalibre seulement la sortie de `predict_proba` : scale_pos_weight ≈ 27
gonfle les scores bruts (un score de 0,6 correspond à ~30 % de pannes
observées, scripts/analyze_threshold_calibration.py). Calibration de
Platt sur la marge XGBoost, strictement croissante : même classement,
même PR-AUC, mêmes alertes qu'avant — seul le chiffre affiché change.
"""

import numpy as np
from scipy.special import expit
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

# Seuil historique des modèles non calibrés : score brut 0,5 = marge 0.
RAW_THRESHOLD = 0.5


def build_xgb_pipeline(params: dict, calibrated: bool = False) -> Pipeline:
    steps = [
        ("imputer", SimpleImputer(strategy="median")),
        ("model", XGBClassifier(**params)),
    ]
    return CalibratedPipeline(steps) if calibrated else Pipeline(steps)


class CalibratedPipeline(Pipeline):
    """Imputer + XGBoost dont `predict_proba` renvoie
    sigmoïde(pente × marge + ordonnée), paramètres appris hors
    échantillon (`modeling.train.fit_calibration`) puis posés par
    `set_calibration`. `threshold_` est le score calibré qui correspond
    au score brut 0,5 : `predict` lève donc les mêmes alertes qu'avant."""

    def set_calibration(self, slope: float, intercept: float) -> "CalibratedPipeline":
        if slope <= 0:
            # une pente négative inverserait le classement du modèle
            raise ValueError(f"Pente de calibration non positive : {slope}")
        self.calibration_ = {"slope": float(slope), "intercept": float(intercept)}
        return self

    def margin(self, X) -> np.ndarray:
        """Marge (logit) XGBoost brute, avant calibration."""
        return self[-1].predict(self[:-1].transform(X), output_margin=True)

    def predict_proba_raw(self, X) -> np.ndarray:
        return super().predict_proba(X)

    def predict_proba(self, X) -> np.ndarray:
        c = self.calibration_
        p = expit(c["slope"] * self.margin(X).astype(float) + c["intercept"])
        return np.column_stack([1 - p, p])

    @property
    def threshold_(self) -> float:
        return float(expit(self.calibration_["intercept"]))  # marge 0 = score brut 0,5

    def predict(self, X) -> np.ndarray:
        return (self.predict_proba(X)[:, 1] >= self.threshold_).astype(int)


def alert_threshold(model) -> float:
    """Seuil d'alerte sur l'échelle de `predict_proba` de ce modèle : le
    seuil calibré porté par le modèle, 0,5 pour un ancien pipeline brut."""
    return getattr(model, "threshold_", RAW_THRESHOLD)


def predict_alert(model, X) -> np.ndarray:
    return (model.predict_proba(X)[:, 1] >= alert_threshold(model)).astype(int)
