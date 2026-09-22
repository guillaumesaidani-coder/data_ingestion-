#!/usr/bin/env python
"""Module 37 : arbitrage champion (modèle de production actuel) vs
challenger, sur des données réelles — matrice gain/stabilité/
régression/angle mort de la feuille de route (§5), jamais un
pourcentage moyen seul.

Le challenger n'est PAS un réentraînement à l'identique (qui ne
prouverait rien de neuf sur un modèle déjà fort, rappel 0,9106) : il
est entraîné sur `trainval` (identique au champion) **augmenté** des
fenêtres avril-mai 2026 déjà revues par la boucle HITL (module 36,
`predictions.ground_truth`) — une période que le champion n'a jamais
vue. Les fenêtres jamais revues (`A_VALIDER`, aucune alerte ni incident
signalé) sont traitées comme négatif implicite : absence de signal =
machine présumée saine, pratique standard en production.

Juin 2026 reste totalement à l'écart de l'entraînement des deux modèles
— c'est le seul terrain d'arbitrage, jamais vu ni par le champion ni
par l'augmentation du challenger.

Usage : uv run --frozen python scripts/arbitrate_challenger.py
"""

import sys
from datetime import UTC, datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import recall_score

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from indusense.config import get_engine, get_model_path, get_predictions_engine
from indusense.modeling.dataset import TARGET, load_gold_dataset
from indusense.modeling.train import (
    B11_PARAMS,
    compute_scale_pos_weight,
    train_and_evaluate,
)

CUTOFF = pd.Timestamp("2026-06-01", tz="UTC")
REPORT_CSV = Path(__file__).resolve().parent.parent / "reports" / "hitl" / "arbitration_log.csv"
DEFAULT_OUT = Path(__file__).resolve().parent.parent / "artifacts" / "models" / "challenger.joblib"


def arbitration_matrix(
    y_true: np.ndarray, pred_champion: np.ndarray, pred_challenger: np.ndarray
) -> dict:
    """Matrice gain/stabilité/régression/angle mort (feuille de route
    §5) — jamais un pourcentage moyen seul."""
    succes_champion = pred_champion == y_true
    succes_challenger = pred_challenger == y_true
    return {
        "gains": int(np.sum((~succes_champion) & succes_challenger)),
        "stables": int(np.sum(succes_champion & succes_challenger)),
        "regressions": int(np.sum(succes_champion & (~succes_challenger))),
        "angles_morts": int(np.sum((~succes_champion) & (~succes_challenger))),
    }


def arbitration_decision(gains: int, regressions: int) -> str:
    """Règle d'or de la feuille de route (§5) : jamais valider sur un score
    global seul. Refuse tout candidat qui régresse sans compensation
    massive et quasi sans exception."""
    if regressions == 0 and gains > 0:
        return "ACCEPTATION_DIRECTE"
    if gains > regressions * 3 and regressions <= 1:
        return "ACCEPTATION_SOUS_DEROGATION"
    return "REJET_DU_MODELE_N"


def _augmented_training_set(gold, predictions_engine):
    reviews = pd.read_sql(
        "SELECT machine_id, window_start, ground_truth FROM predictions",
        predictions_engine,
    )
    reviews["window_start"] = pd.to_datetime(reviews["window_start"])

    early = gold.test_df[gold.test_df["window_start"] < CUTOFF].copy()
    early = early.merge(reviews, on=["machine_id", "window_start"], how="left")
    # Verite HITL si la fenetre a ete revue (alerte confirmee/infirmee ou
    # incident declare) ; sinon negatif implicite (aucun signal remonte).
    early["label_augmente"] = early["ground_truth"].fillna(False).astype(int)

    X_new = early[gold.feature_cols]
    y_new = early["label_augmente"].astype(int)

    X_aug = pd.concat([gold.X_tv, X_new], ignore_index=True)
    # y_tv est bool (colonne Postgres), y_new est int : concat sans caster les
    # deux au meme type produit un Series 'object' que sklearn refuse
    # (ValueError: unknown format is not supported).
    y_aug = pd.concat([gold.y_tv.astype(int), y_new], ignore_index=True)
    return X_aug, y_aug, len(early), int(y_new.sum())


def main() -> int:
    gold = load_gold_dataset(get_engine())
    predictions_engine = get_predictions_engine()

    X_aug, y_aug, n_new, n_new_failures = _augmented_training_set(gold, predictions_engine)
    print(
        f"Entraînement augmenté : {len(gold.X_tv)} lignes (champion) + {n_new} lignes "
        f"avril-mai revues par la boucle HITL ({n_new_failures} pannes confirmées/déclarées) "
        f"= {len(X_aug)} lignes"
    )

    june = gold.test_df[gold.test_df["window_start"] >= CUTOFF].copy()
    X_june, y_june = june[gold.feature_cols], june[TARGET].astype(int)
    print(f"Arbitrage sur juin 2026 : {len(june)} fenêtres, jamais vues par aucun des deux modèles")

    params = {**B11_PARAMS, "scale_pos_weight": compute_scale_pos_weight(y_aug)}
    challenger_pipe, challenger_metrics = train_and_evaluate(X_aug, y_aug, X_june, y_june, params)
    print(
        f"Challenger (mesuré sur juin) : rappel={challenger_metrics['recall_test']} "
        f"précision={challenger_metrics['precision_test']} "
        f"ROC-AUC={challenger_metrics['roc_auc_test']}"
    )

    champion = joblib.load(get_model_path())
    champion_proba = champion.predict_proba(X_june)[:, 1]
    challenger_proba = challenger_pipe.predict_proba(X_june)[:, 1]

    y_true = y_june.to_numpy()
    pred_champion = (champion_proba >= 0.5).astype(int)
    pred_challenger = (challenger_proba >= 0.5).astype(int)

    matrix = arbitration_matrix(y_true, pred_champion, pred_challenger)
    gains, stables = matrix["gains"], matrix["stables"]
    regressions, angles_morts = matrix["regressions"], matrix["angles_morts"]

    print(f"Gain pur (champion faux, challenger juste)   : +{gains}")
    print(f"Stabilité (les deux justes)                  : {stables}")
    print(f"Régression (champion juste, challenger faux) : -{regressions}")
    print(f"Angle mort (les deux faux)                   : {angles_morts}")
    print(f"Bilan net : {gains - regressions}")

    decision = arbitration_decision(gains, regressions)
    print(f"Décision : {decision}")

    DEFAULT_OUT.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(challenger_pipe, DEFAULT_OUT)
    print(f"Challenger sauvegardé -> {DEFAULT_OUT} (non activé — bascule séparée, module 39)")

    REPORT_CSV.parent.mkdir(parents=True, exist_ok=True)
    row = pd.DataFrame(
        [
            {
                "horodatage": datetime.now(UTC).isoformat(),
                "n_train_augmente": n_new,
                "n_pannes_augmentees": n_new_failures,
                "n_arbitrage": len(june),
                "gains": gains,
                "stables": stables,
                "regressions": regressions,
                "angles_morts": angles_morts,
                "rappel_champion_juin": round(float(recall_score(y_true, pred_champion)), 4),
                "rappel_challenger_juin": round(float(recall_score(y_true, pred_challenger)), 4),
                "decision": decision,
            }
        ]
    )
    row.to_csv(REPORT_CSV, mode="a", header=not REPORT_CSV.exists(), index=False)
    print(f"Arbitrage journalisé -> {REPORT_CSV}")

    return 0 if decision != "REJET_DU_MODELE_N" else 1


if __name__ == "__main__":
    sys.exit(main())
