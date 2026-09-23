"""Module 38 : décide QUAND lancer le cycle d'arbitrage champion/
challenger (module 37) — cinq feux verts avant de dépenser un
entraînement (feuille de route §6), jamais un déclenchement calendaire
aveugle.

1. Quota de nouveautés   : >= 50 validations humaines enregistrées (module 36).
2. Équilibre des données : >= 10 pannes confirmées/déclarées parmi ces retours.
3. Qualité des données   : aucune valeur de capteur hors bornes physiques
   (indusense.data_quality, bornes reprises du contrat DAT §5.4).
4. Dérive constatée      : PSI (indusense.drift, module 31) entre `trainval`
   et la période proposée pour l'entraînement augmenté dépasse le seuil
   d'alerte (0.25, même convention que modules 31-34) — sans dérive
   mesurée, rien ne justifie de toucher au modèle.
5. Nouveau lot terrain   : au moins un batch Gold (indusense.ingest,
   `ingestion_batch.source_name='gold'`) a été ingéré depuis `cutoff` —
   sans ça, les feux 3/4 rejoueraient les mêmes données déjà vues plutôt
   que de vraies nouvelles mesures terrain.

`ensure_model(..., retrain=True)` existe depuis les modules 29-30 mais
n'était jusqu'ici jamais piloté par un vrai critère : ce flow est ce
critère.
"""

import sys
from pathlib import Path

import pandas as pd
from prefect import flow, get_run_logger, task

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from sqlalchemy import text

from indusense.arbitration import CUTOFF, persist_arbitration, run_arbitration
from indusense.config import get_engine, get_predictions_engine
from indusense.data_quality import check_sensor_quality
from indusense.drift import psi
from indusense.modeling.dataset import load_gold_dataset
from indusense.predictions_store import CREATE_TABLE_SQL

MIN_VALIDATIONS = 50
MIN_CONFIRMED_FAILURES = 10
DRIFT_FEATURES = [
    "temp_mean_24h",
    "temp_std_24h",
    "pressure_mean_24h",
    "pressure_std_24h",
    "voltage_mean_24h",
    "rotation_mean_24h",
    "pieces_produced_sum_24h",
    "incident_count_prev_24h",
]
PSI_ALERT_THRESHOLD = 0.25


def _load_reviews(predictions_engine) -> pd.DataFrame:
    # Environnement neuf (aucun backfill/scoring encore lancé) : la table
    # n'existe pas encore -- meme garde que predictions_store.py, pour que
    # ce feu vert se lise "0 validations, suspendu" plutot que de planter.
    with predictions_engine.begin() as conn:
        conn.execute(text(CREATE_TABLE_SQL))
    return pd.read_sql(
        "SELECT review_status, ground_truth FROM predictions WHERE review_status != 'A_VALIDER'",
        predictions_engine,
    )


def check_validation_quota(reviews: pd.DataFrame) -> tuple[bool, str]:
    n = len(reviews)
    return n >= MIN_VALIDATIONS, f"{n} validations humaines (seuil {MIN_VALIDATIONS})"


def check_confirmed_failures_quota(reviews: pd.DataFrame) -> tuple[bool, str]:
    n = int(reviews["ground_truth"].fillna(False).sum())
    return (
        n >= MIN_CONFIRMED_FAILURES,
        f"{n} pannes confirmées/déclarées (seuil {MIN_CONFIRMED_FAILURES})",
    )


def _load_ingestion_batches(engine, source_name: str = "gold") -> pd.DataFrame:
    return pd.read_sql(
        text("SELECT started_at, status FROM ingestion_batch WHERE source_name = :source_name"),
        engine,
        params={"source_name": source_name},
    )


def check_data_quality(gold, cutoff: pd.Timestamp) -> tuple[bool, str]:
    early = gold.test_df[gold.test_df["window_start"] < cutoff]
    ok, errors = check_sensor_quality(early)
    detail = "aucune valeur hors bornes" if ok else f"{len(errors)} violation(s) : {errors[0]}"
    return ok, detail


def check_drift_signal(gold, cutoff: pd.Timestamp) -> tuple[bool, str]:
    early = gold.test_df[gold.test_df["window_start"] < cutoff]
    psi_values = {f: psi(gold.trainval_df[f], early[f]) for f in DRIFT_FEATURES}
    max_feature = max(psi_values, key=psi_values.get)
    max_psi = psi_values[max_feature]
    ok = max_psi > PSI_ALERT_THRESHOLD
    detail = f"PSI max {max_feature}={max_psi:.4f} (seuil {PSI_ALERT_THRESHOLD})"
    return ok, detail


def check_new_terrain_batch(batches: pd.DataFrame, cutoff: pd.Timestamp) -> tuple[bool, str]:
    """Vrai si au moins un batch Gold clos (indusense.ingest, via la même
    table `ingestion_batch` que TP4/TP5/TP6) a démarré depuis `cutoff` --
    la preuve qu'un nouveau lot de données terrain a bien traversé Bronze ->
    Silver -> Gold, pas seulement un rejeu de l'existant."""
    done = batches[batches["status"] == "done"].copy()
    done["started_at"] = pd.to_datetime(done["started_at"], utc=True)
    fresh = done[done["started_at"] >= cutoff]
    ok = not fresh.empty
    detail = (
        f"{len(fresh)} lot(s) Gold ingéré(s) depuis {cutoff.date()}"
        if ok
        else f"aucun nouveau lot Gold depuis {cutoff.date()}"
    )
    return ok, detail


@task(name="verifier-conditions-reentrainement")
def verifier_conditions_reentrainement(cutoff: pd.Timestamp = CUTOFF) -> tuple[bool, dict]:
    logger = get_run_logger()
    engine = get_engine()
    gold = load_gold_dataset(engine)
    predictions_engine = get_predictions_engine()
    reviews = _load_reviews(predictions_engine)
    batches = _load_ingestion_batches(engine)

    gates = {
        "1_quota_validations": check_validation_quota(reviews),
        "2_quota_pannes_confirmees": check_confirmed_failures_quota(reviews),
        "3_qualite_donnees": check_data_quality(gold, cutoff),
        "4_derive": check_drift_signal(gold, cutoff),
        "5_nouveau_lot_terrain": check_new_terrain_batch(batches, cutoff),
    }
    for name, (ok, detail) in gates.items():
        logger.info("Feu vert '%s' : %s -- %s", name, "OK" if ok else "SUSPENDU", detail)

    tous_verts = all(ok for ok, _ in gates.values())
    return tous_verts, gates


@flow(name="indusense-retrain-cycle")
def retrain_cycle(cutoff: pd.Timestamp = CUTOFF) -> dict:
    logger = get_run_logger()
    tous_verts, gates = verifier_conditions_reentrainement(cutoff)

    if not tous_verts:
        suspendus = [name for name, (ok, _) in gates.items() if not ok]
        logger.info("Cycle suspendu : feux non verts -> %s", suspendus)
        return {"status": "SUSPENDU", "gates": gates}

    logger.info("5 feux verts réunis -- lancement de l'arbitrage champion/challenger")
    result = run_arbitration(cutoff)
    persist_arbitration(result)
    logger.info(
        "Arbitrage terminé : décision=%s (gains=%d, régressions=%d)",
        result["decision"],
        result["gains"],
        result["regressions"],
    )
    return {"status": "ARBITRE", "gates": gates, "decision": result["decision"]}


def main() -> None:
    result = retrain_cycle()
    print(f"Cycle de réentraînement : {result}")


if __name__ == "__main__":
    main()
