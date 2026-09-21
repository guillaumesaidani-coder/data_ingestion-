"""Chargement du Gold dataset pour l'entraînement (extrait de TP9.ipynb /
TP11.ipynb, où ce bloc était dupliqué à l'identique — et de
generate_model_card.py, où la même logique avait divergé en oubliant les
colonnes future_incident_count_* de LEAKAGE_COLS, cf. le commit qui a
corrigé cette fuite).
"""
from dataclasses import dataclass

import pandas as pd
from sqlalchemy.engine import Engine

TARGET = "label_failure_next_24h"

LEAKAGE_COLS = [
    "machine_id", "ingestion_batch_id", "window_start", "window_end", "split_set",
    "label_failure_next_6h", "label_failure_next_12h", "label_failure_next_48h",
    TARGET, "feature_row_id",
    # Compte brut d'incidents futurs — equivalent mathematique du label (fuite directe)
    "future_incident_count_6h", "future_incident_count_12h",
    "future_incident_count_24h", "future_incident_count_48h",
]


@dataclass
class GoldDataset:
    df: pd.DataFrame
    trainval_df: pd.DataFrame
    test_df: pd.DataFrame
    feature_cols: list[str]
    X_tv: pd.DataFrame
    y_tv: pd.Series
    groups: pd.Series
    X_test: pd.DataFrame
    y_test: pd.Series


def load_gold_dataset(engine: Engine) -> GoldDataset:
    """Lit gold_machine_hourly_feature, calcule FEATURE_COLS (toutes les
    colonnes sauf LEAKAGE_COLS) et sépare train+validation (pour la CV) du
    test holdout, comme dans TP9/TP11."""
    df = pd.read_sql(
        "SELECT * FROM gold_machine_hourly_feature ORDER BY machine_id, window_start",
        engine,
    )

    feature_cols = [c for c in df.columns if c not in LEAKAGE_COLS]

    trainval_df = df[df["split_set"].isin(["train", "validation"])].copy()
    test_df     = df[df["split_set"] == "test"].copy()

    X_tv   = trainval_df[feature_cols]
    y_tv   = trainval_df[TARGET]
    groups = trainval_df["machine_id"]

    X_test, y_test = test_df[feature_cols], test_df[TARGET]

    return GoldDataset(
        df=df, trainval_df=trainval_df, test_df=test_df,
        feature_cols=feature_cols,
        X_tv=X_tv, y_tv=y_tv, groups=groups,
        X_test=X_test, y_test=y_test,
    )
