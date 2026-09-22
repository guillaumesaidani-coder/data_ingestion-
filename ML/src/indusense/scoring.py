"""Scoring d'un DataFrame de features avec le modèle b11-gkf (modules
35-36) : logique partagée entre le scoring horaire
(`flows.predict_flow.predict_latest`) et le backfill historique
(`scripts/backfill_predictions.py`) — une seule implémentation de
« quelles colonnes le modèle attend » et « comment figer une photo de
features en JSON », pas deux qui pourraient diverger.
"""

import pandas as pd


def model_feature_cols(model, available_cols: list[str]) -> list[str]:
    """Colonnes attendues par le modèle, dans l'ordre appris à
    l'entraînement (`imputer.feature_names_in_`) — retombe sur les
    colonnes disponibles hors identifiants si le pipeline n'a pas
    d'imputer (jamais notre cas réel, mais pas de raison de planter)."""
    imputer = model.named_steps.get("imputer")
    if imputer is not None and hasattr(imputer, "feature_names_in_"):
        return list(imputer.feature_names_in_)
    return [c for c in available_cols if c not in ("machine_id", "window_start")]


def score_features(model, df: pd.DataFrame) -> tuple[pd.Series, list[dict]]:
    """Probabilité de panne + photo JSON-sérialisable des features
    utilisées (NaN -> None, `json.dumps` ne produit pas de JSON valide
    sur un NaN brut)."""
    feature_cols = model_feature_cols(model, list(df.columns))
    features = df.reindex(columns=feature_cols)
    proba = model.predict_proba(features)[:, 1]
    payloads = [
        {k: (None if pd.isna(v) else v) for k, v in row.items()}
        for row in features.to_dict(orient="records")
    ]
    return proba, payloads
