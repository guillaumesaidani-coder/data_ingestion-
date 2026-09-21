"""Pipeline scikit-learn standard du projet (imputation médiane + XGBoost),
répétée à l'identique dans TP9.ipynb, TP11.ipynb et generate_model_card.py.
"""

from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier


def build_xgb_pipeline(params: dict) -> Pipeline:
    return Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("model", XGBClassifier(**params)),
        ]
    )
