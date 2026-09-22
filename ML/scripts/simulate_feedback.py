#!/usr/bin/env python
"""Module 36 : simule le retour terrain sur les prédictions déjà en
base (`scripts/backfill_predictions.py`) — pas un humain qui clique dans
Streamlit, mais le même mécanisme (`review_prediction`), avec la vérité
terrain réelle et déjà connue (`label_failure_next_24h`, incidents réels)
au lieu d'un technicien qui met des semaines à confirmer.

Reproduit puis mesure le biais de sélection que la feuille de route
signale (§5) : un vrai technicien n'inspecte que les machines où le
modèle a levé une alerte. Sans le bouton « incident non prédit », les
pannes que le modèle a manquées (faux négatifs) ne sont *jamais*
visibles dans les retours — mesuré ici, pas affirmé.

Usage :
    uv run --frozen python scripts/simulate_feedback.py                        # biais de selection pur
    uv run --frozen python scripts/simulate_feedback.py --incident-non-predit-rate 1.0   # + bouton
"""

import argparse
import sys
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from indusense.config import get_engine, get_predictions_engine
from indusense.predictions_store import review_prediction

ALERT_THRESHOLD = 0.5


def _load_pending_with_ground_truth(predictions_engine) -> pd.DataFrame:
    predictions = pd.read_sql(
        "SELECT machine_id, window_start, failure_proba_24h "
        "FROM predictions WHERE review_status = 'A_VALIDER'",
        predictions_engine,
    )
    if predictions.empty:
        return predictions

    gold = pd.read_sql(
        "SELECT machine_id, window_start, label_failure_next_24h "
        "FROM gold_machine_hourly_feature",
        get_engine(),
    )
    gold["window_start"] = pd.to_datetime(gold["window_start"]).apply(lambda ts: ts.isoformat())
    return predictions.merge(gold, on=["machine_id", "window_start"], how="left")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--incident-non-predit-rate",
        type=float,
        default=0.0,
        help="Fraction des pannes manquées (faux négatifs) déclarée via le bouton "
        "'incident non prédit' (0.0 = biais de sélection pur, comme un technicien "
        "qui ne regarde que les alertes ; 1.0 = tous les incidents remontent)",
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    engine = get_predictions_engine()
    df = _load_pending_with_ground_truth(engine)
    if df.empty:
        print("Aucune prédiction en attente de revue (A_VALIDER) — lancez backfill_predictions.py d'abord.")
        return 1

    alerts = df[df["failure_proba_24h"] >= ALERT_THRESHOLD]
    missed = df[(df["failure_proba_24h"] < ALERT_THRESHOLD) & (df["label_failure_next_24h"] == 1)]

    reviewed_at = datetime.now(UTC).isoformat()
    n_confirmed = n_false_alert = 0
    for row in alerts.itertuples():
        is_real_failure = bool(row.label_failure_next_24h)
        review_prediction(
            engine,
            row.machine_id,
            row.window_start,
            review_status="PANNE_CONFIRMEE" if is_real_failure else "FAUSSE_ALERTE",
            ground_truth=is_real_failure,
            reviewer_comment="revue simulee (verite terrain reelle deja connue)",
            reviewed_at=reviewed_at,
        )
        n_confirmed += is_real_failure
        n_false_alert += not is_real_failure

    reported_missed = missed.sample(frac=args.incident_non_predit_rate, random_state=args.seed)
    for row in reported_missed.itertuples():
        review_prediction(
            engine,
            row.machine_id,
            row.window_start,
            review_status="INCIDENT_NON_PREDIT",
            ground_truth=True,
            reviewer_comment="panne signalee via le canal manuel, jamais alertee par le modele",
            reviewed_at=reviewed_at,
        )

    total_real_failures = int(alerts["label_failure_next_24h"].sum()) + len(missed)
    covered_before = int(alerts["label_failure_next_24h"].sum())
    covered_after = covered_before + len(reported_missed)

    print(f"Alertes revues (biais de sélection, comme un vrai technicien) : {len(alerts)}")
    print(f"  -> {n_confirmed} PANNE_CONFIRMEE, {n_false_alert} FAUSSE_ALERTE")
    print(f"Pannes manquées par le modèle (faux négatifs) dans ce lot : {len(missed)}")
    print(
        f"Déclarées via 'incident non prédit' (taux {args.incident_non_predit_rate:.0%}) : "
        f"{len(reported_missed)}"
    )
    print(f"Couverture des vraies pannes dans les retours : {covered_before}/{total_real_failures}"
          f" ({covered_before / total_real_failures:.1%}) sans le bouton"
          f" -> {covered_after}/{total_real_failures} ({covered_after / total_real_failures:.1%}) avec")
    if args.incident_non_predit_rate == 0.0 and len(missed) > 0:
        print(
            f"BIAIS DE SÉLECTION MESURÉ : {len(missed)} vraies pannes restent invisibles dans les "
            "retours terrain tant que rien ne force leur remontée — c'est exactement ce que le "
            "modèle n'a pas vu, et donc ce qu'un réentraînement sur ces seuls retours ne corrigerait "
            "jamais."
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
