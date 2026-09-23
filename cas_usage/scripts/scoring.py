"""Esquisse d'API de scoring — churn + CLV + priorisation + action recommandée.

Charge les deux modèles sérialisés (train_and_serialize.py) et expose une fonction
`score_client` qui traduit une prédiction brute en information actionnable pour un
Customer Success Manager (cf. TP1) — un score seul n'est pas un livrable suffisant.

Usage (démo) :
    python scripts/scoring.py
"""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = ROOT / "models"

# Seuils de risque (TP6 : seuil de décision retenu sous contrainte de capacité ~40%)
SEUIL_RISQUE_ELEVE = 0.44
SEUIL_RISQUE_MODERE = 0.20

# Matrice de priorisation (risque x CLV) — cf. exemple TP1 (§ cadrage métier)
MATRICE_PRIORITE = {
    ("Élevé", "Élevée"):  "Très élevée",
    ("Élevé", "Moyenne"): "Élevée",
    ("Élevé", "Faible"):  "Modérée",
    ("Modéré", "Élevée"): "Surveillance",
    ("Modéré", "Moyenne"):"Modérée",
    ("Modéré", "Faible"): "Faible",
    ("Faible", "Élevée"): "Surveillance",
    ("Faible", "Moyenne"):"Faible",
    ("Faible", "Faible"): "Faible",
}


def load_models():
    model_churn = joblib.load(MODELS_DIR / "model_churn.joblib")
    model_clv = joblib.load(MODELS_DIR / "model_clv.joblib")
    metadata = json.loads((MODELS_DIR / "metadata.json").read_text(encoding="utf-8"))
    return model_churn, model_clv, metadata


def _tier_risque(proba: float) -> str:
    if proba >= SEUIL_RISQUE_ELEVE:
        return "Élevé"
    if proba >= SEUIL_RISQUE_MODERE:
        return "Modéré"
    return "Faible"


def _tier_clv(clv: float, p25: float, p75: float) -> str:
    if clv >= p75:
        return "Élevée"
    if clv >= p25:
        return "Moyenne"
    return "Faible"


def _action_recommandee(client: pd.Series) -> str:
    """Règle simple : le signal le plus préoccupant détermine l'action prioritaire."""
    if client["derniere_connexion_jours"] > 30:
        return "Contacter rapidement le client (inactif depuis plus de 30 jours)"
    if client["taux_adoption_pct"] < 30:
        return "Proposer une formation / démonstration personnalisée (faible adoption)"
    if client["nb_integrations"] == 0:
        return "Accompagner le déploiement (aucune intégration connectée)"
    if client["tickets_support_90j"] >= 5:
        return "Prioriser le support, suivi rapproché (nombreux tickets récents)"
    if client["retards_paiement_12m"] > 0:
        return "Proposer un nouvel échéancier (retards de paiement constatés)"
    return "Suivi standard"


def score_client(client_row: pd.Series, model_churn, model_clv, metadata) -> dict:
    feature_cols = metadata["feature_cols"]
    X = pd.DataFrame([client_row[feature_cols]])

    proba_churn = float(model_churn.predict_proba(X)[0, 1])
    clv_predite = float(model_clv.predict(X)[0])

    tier_risque = _tier_risque(proba_churn)
    tier_clv = _tier_clv(clv_predite, metadata["clv"]["clv_p25"], metadata["clv"]["clv_p75"])
    priorite = MATRICE_PRIORITE[(tier_risque, tier_clv)]

    return {
        "proba_churn": round(proba_churn, 3),
        "tier_risque": tier_risque,
        "clv_predite_eur": round(clv_predite, 0),
        "tier_clv": tier_clv,
        "priorite": priorite,
        "action_recommandee": _action_recommandee(client_row),
    }


def main():
    from sqlalchemy import create_engine
    from sqlalchemy.engine import URL

    model_churn, model_clv, metadata = load_models()

    db_url = URL.create(
        drivername="postgresql+psycopg2",
        username="indusense_user", password="ThEP@ssW0rd",
        host="localhost", port=5432, database="churn_saas_db",
    )
    engine = create_engine(db_url)
    df = pd.read_sql("SELECT * FROM clients_churn LIMIT 5", engine)

    for _, row in df.iterrows():
        resultat = score_client(row, model_churn, model_clv, metadata)
        print(f"{row['client_id']} -> {resultat}")


if __name__ == "__main__":
    main()
