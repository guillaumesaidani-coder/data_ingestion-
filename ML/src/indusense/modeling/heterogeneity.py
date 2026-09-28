"""Hétérogénéité entre machines et entre types de presse (InduPress-X1,
X2, X3, Z1 dans la table `machine`) : validation croisée « une machine
cachée » et « un type caché » avec les hyperparamètres b11, partagée
entre generate_model_card.py (limites chiffrées en direct, plus de
valeurs recopiées d'un ancien run) et scripts/analyze_machine_types.py.

Le modèle ne reçoit ni `machine_id` ni le type de presse : cacher un
type entier mesure ce qu'il vaut sur un modèle de presse jamais vu, le
cas d'un ajout de machine d'un nouveau type dans l'usine.
"""

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score
from sqlalchemy import text
from sqlalchemy.engine import Engine

from indusense.modeling.pipeline import build_xgb_pipeline
from indusense.modeling.train import compute_scale_pos_weight

MACHINE_REFERENCE_QUERY = (
    "SELECT machine_code AS machine_id, model AS machine_type, commissioning_date FROM machine"
)


def load_machine_reference(engine: Engine) -> pd.DataFrame:
    return pd.read_sql(text(MACHINE_REFERENCE_QUERY), engine, parse_dates=["commissioning_date"])


def leave_one_group_out_proba(
    df: pd.DataFrame, feature_cols: list[str], target: str, group_col: str, params: dict
) -> pd.Series:
    """Probabilité de panne de chaque ligne, prédite par un modèle entraîné
    sans aucune ligne de son groupe (machine ou type). `scale_pos_weight`
    est recalculé sur chaque entraînement, comme pour le modèle final."""
    proba = pd.Series(np.nan, index=df.index)
    for group in sorted(df[group_col].unique()):
        held_out = df[group_col] == group
        train = df[~held_out]
        pipe = build_xgb_pipeline(
            {**params, "scale_pos_weight": compute_scale_pos_weight(train[target])}
        )
        pipe.fit(train[feature_cols], train[target])
        proba[held_out] = pipe.predict_proba(df.loc[held_out, feature_cols])[:, 1]
    return proba


def pr_auc_by(df: pd.DataFrame, target: str, proba: pd.Series, by: str) -> pd.Series:
    """PR-AUC par groupe ; NaN pour un groupe sans panne (ou sans ligne
    saine), où l'average precision n'est pas définie."""

    def _ap(idx: pd.Index) -> float:
        y = df.loc[idx, target]
        return float(average_precision_score(y, proba[idx])) if y.nunique() == 2 else np.nan

    return pd.Series({group: _ap(idx) for group, idx in df.groupby(by).groups.items()})
