"""Entraîne les deux modèles finaux (churn + CLV) et les sérialise.

Reprend exactement les configurations validées : régression logistique pour le churn
(TP5/TP6), Random Forest pour la CLV (TP8). Ce script ne contient aucune nouvelle
logique de modélisation — il rejoue ce qui a déjà été décidé et vérifié dans les
notebooks, pour produire des artefacts réutilisables (joblib), pas un notebook qui
prétendrait être le livrable.

Usage:
    python scripts/train_and_serialize.py
"""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import pandas as pd
from sqlalchemy import create_engine
from sqlalchemy.engine import URL
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.metrics import roc_auc_score, average_precision_score

RANDOM_STATE = 42
ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = ROOT / "models"
MODELS_DIR.mkdir(exist_ok=True)

FEATURE_COLS_CHURN = [
    "secteur", "pays", "taille_entreprise", "plan", "anciennete_mois", "sieges_souscrits",
    "utilisateurs_actifs", "taux_adoption_pct", "connexions_30j", "heures_usage_30j",
    "fonctionnalites_total", "fonctionnalites_utilisees", "nb_integrations",
    "derniere_connexion_jours", "tickets_support_90j", "delai_reponse_support_h", "csat",
    "retards_paiement_12m", "revenu_mensuel_recurrent_eur", "prix_mensuel_par_siege_eur",
    "fonctionnalites_incluses", "sla_reponse_h", "quota_stockage_go", "support_dedie",
]
# Identiques, seule la cible change — churn n'est jamais une feature du modèle CLV.
FEATURE_COLS_CLV = FEATURE_COLS_CHURN
CATEGORICAL_FEATURES = ["secteur", "pays", "taille_entreprise", "plan", "support_dedie"]
NUMERIC_FEATURES = [c for c in FEATURE_COLS_CHURN if c not in CATEGORICAL_FEATURES]

SEUIL_DECISION_CHURN = 0.44  # capacité CS ~40%, choisi en TP6


def load_data() -> pd.DataFrame:
    db_url = URL.create(
        drivername="postgresql+psycopg2",
        username="indusense_user", password="ThEP@ssW0rd",
        host="localhost", port=5432, database="churn_saas_db",
    )
    engine = create_engine(db_url)
    return pd.read_sql("SELECT * FROM clients_churn", engine)


def train_churn_model(df: pd.DataFrame) -> tuple[Pipeline, dict]:
    X = df[FEATURE_COLS_CHURN]
    y = df["churn"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE
    )
    preprocessor = ColumnTransformer([
        ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_FEATURES),
        ("num", StandardScaler(), NUMERIC_FEATURES),
    ])
    model = Pipeline([
        ("prep", preprocessor),
        ("model", LogisticRegression(class_weight="balanced", max_iter=1000, random_state=RANDOM_STATE)),
    ])
    model.fit(X_train, y_train)

    y_prob = model.predict_proba(X_test)[:, 1]
    metrics = {
        "roc_auc": round(float(roc_auc_score(y_test, y_prob)), 4),
        "pr_auc": round(float(average_precision_score(y_test, y_prob)), 4),
        "seuil_decision": SEUIL_DECISION_CHURN,
    }
    return model, metrics


def train_clv_model(df: pd.DataFrame) -> tuple[Pipeline, dict]:
    X = df[FEATURE_COLS_CLV]
    y = df["valeur_vie_client_eur"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE
    )
    preprocessor = ColumnTransformer([
        ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_FEATURES),
        ("num", StandardScaler(), NUMERIC_FEATURES),
    ])
    model = Pipeline([
        ("prep", preprocessor),
        ("model", RandomForestRegressor(n_estimators=300, random_state=RANDOM_STATE, n_jobs=-1)),
    ])
    model.fit(X_train, y_train)

    from sklearn.metrics import mean_absolute_error, r2_score
    pred = model.predict(X_test)
    metrics = {
        "mae_eur": round(float(mean_absolute_error(y_test, pred)), 2),
        "r2": round(float(r2_score(y_test, pred)), 4),
        "clv_p25": round(float(y_train.quantile(0.25)), 2),
        "clv_p75": round(float(y_train.quantile(0.75)), 2),
    }
    return model, metrics


def main():
    print("[1/3] Chargement des données...")
    df = load_data()
    print(f"  {len(df):,} lignes")

    print("[2/3] Entraînement + sérialisation du modèle churn (régression logistique)...")
    model_churn, metrics_churn = train_churn_model(df)
    joblib.dump(model_churn, MODELS_DIR / "model_churn.joblib")
    print(f"  ROC-AUC={metrics_churn['roc_auc']}  PR-AUC={metrics_churn['pr_auc']}")

    print("[3/3] Entraînement + sérialisation du modèle CLV (Random Forest)...")
    model_clv, metrics_clv = train_clv_model(df)
    joblib.dump(model_clv, MODELS_DIR / "model_clv.joblib")
    print(f"  MAE={metrics_clv['mae_eur']}€  R²={metrics_clv['r2']}")

    metadata = {
        "feature_cols": FEATURE_COLS_CHURN,
        "categorical_features": CATEGORICAL_FEATURES,
        "numeric_features": NUMERIC_FEATURES,
        "churn": metrics_churn,
        "clv": metrics_clv,
    }
    (MODELS_DIR / "metadata.json").write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nArtefacts écrits dans {MODELS_DIR}/ : model_churn.joblib, model_clv.joblib, metadata.json")


if __name__ == "__main__":
    main()
