"""Arbitrage champion vs challenger (module 37-38) : le challenger n'est
pas un réentraînement à l'identique (qui ne prouverait rien de neuf sur
un modèle déjà fort, rappel 0,9106) — il est entraîné sur `trainval`
(identique au champion) **augmenté** des fenêtres déjà revues par la
boucle HITL (module 36, `predictions.ground_truth`), sur une période que
le champion n'a jamais vue. `CUTOFF` reste totalement à l'écart de
l'entraînement des deux modèles — c'est le seul terrain d'arbitrage.

Vit dans le paquet, pas dans `scripts/` : `scripts/arbitrate_challenger.py`
(CLI, module 37) et `indusense.flows.retrain_flow` (Prefect, module 38)
partagent cette même implémentation — même leçon qu'au module 30, un
script à la racine ou un flow ne sont que des façades.
"""

from datetime import UTC, datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import recall_score

from indusense.config import ML_DIR, get_engine, get_model_path, get_predictions_engine
from indusense.modeling.dataset import TARGET, load_gold_dataset
from indusense.modeling.train import (
    B11_PARAMS,
    compute_scale_pos_weight,
    train_and_evaluate,
)

CUTOFF = pd.Timestamp("2026-06-01", tz="UTC")
DEFAULT_CHALLENGER_PATH = ML_DIR / "artifacts" / "models" / "challenger.joblib"
DEFAULT_REPORT_CSV = ML_DIR / "reports" / "hitl" / "arbitration_log.csv"


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


def augmented_training_set(gold, predictions_engine, cutoff: pd.Timestamp = CUTOFF):
    reviews = pd.read_sql(
        "SELECT machine_id, window_start, ground_truth FROM predictions",
        predictions_engine,
    )
    reviews["window_start"] = pd.to_datetime(reviews["window_start"])

    early = gold.test_df[gold.test_df["window_start"] < cutoff].copy()
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


def run_arbitration(cutoff: pd.Timestamp = CUTOFF) -> dict:
    """Entraîne le challenger et l'arbitre contre le champion. Retourne
    tout ce dont un appelant (CLI ou flow Prefect) a besoin : le pipeline
    entraîné, la matrice, la décision, et les métriques pour journalisation."""
    gold = load_gold_dataset(get_engine())
    predictions_engine = get_predictions_engine()

    X_aug, y_aug, n_new, n_new_failures = augmented_training_set(gold, predictions_engine, cutoff)

    arbitrage_df = gold.test_df[gold.test_df["window_start"] >= cutoff].copy()
    X_arb, y_arb = arbitrage_df[gold.feature_cols], arbitrage_df[TARGET].astype(int)

    params = {**B11_PARAMS, "scale_pos_weight": compute_scale_pos_weight(y_aug)}
    challenger_pipe, challenger_metrics = train_and_evaluate(X_aug, y_aug, X_arb, y_arb, params)

    champion = joblib.load(get_model_path())
    champion_proba = champion.predict_proba(X_arb)[:, 1]
    challenger_proba = challenger_pipe.predict_proba(X_arb)[:, 1]

    y_true = y_arb.to_numpy()
    pred_champion = (champion_proba >= 0.5).astype(int)
    pred_challenger = (challenger_proba >= 0.5).astype(int)

    matrix = arbitration_matrix(y_true, pred_champion, pred_challenger)
    decision = arbitration_decision(matrix["gains"], matrix["regressions"])

    return {
        "challenger_pipe": challenger_pipe,
        "challenger_metrics": challenger_metrics,
        "n_train_augmente": n_new,
        "n_pannes_augmentees": n_new_failures,
        "n_arbitrage": len(arbitrage_df),
        **matrix,
        "rappel_champion": round(float(recall_score(y_true, pred_champion)), 4),
        "rappel_challenger": round(float(recall_score(y_true, pred_challenger)), 4),
        "decision": decision,
    }


def persist_arbitration(
    result: dict,
    out_path: Path = DEFAULT_CHALLENGER_PATH,
    report_csv: Path = DEFAULT_REPORT_CSV,
) -> None:
    """Sauvegarde le challenger (jamais activé — bascule séparée, module
    39) et journalise la décision. Partagé par le CLI (module 37) et le
    flow Prefect (module 38) : une seule façon d'écrire ces preuves."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(result["challenger_pipe"], out_path)

    report_csv.parent.mkdir(parents=True, exist_ok=True)
    row = pd.DataFrame(
        [
            {
                "horodatage": datetime.now(UTC).isoformat(),
                "n_train_augmente": result["n_train_augmente"],
                "n_pannes_augmentees": result["n_pannes_augmentees"],
                "n_arbitrage": result["n_arbitrage"],
                "gains": result["gains"],
                "stables": result["stables"],
                "regressions": result["regressions"],
                "angles_morts": result["angles_morts"],
                "rappel_champion_juin": result["rappel_champion"],
                "rappel_challenger_juin": result["rappel_challenger"],
                "decision": result["decision"],
            }
        ]
    )
    row.to_csv(report_csv, mode="a", header=not report_csv.exists(), index=False)
