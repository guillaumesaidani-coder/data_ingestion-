#!/usr/bin/env python
"""Module 36 : peuple `predictions` avec un historique réel à réviser —
le flow horaire (`predict_flow.py`) ne score que la dernière fenêtre par
machine (15 lignes), pas assez pour simuler une boucle de retour
terrain. Ce script score le split `test` du Gold (19 944 lignes réelles,
avril-juin 2026, jamais vues à l'entraînement) avec le modèle de
production actuel — comme si ce modèle tournait en production depuis le
début de cette période et que chaque fenêtre avait déjà été scorée.

Usage : uv run --frozen python scripts/backfill_predictions.py [--limit N]
"""

import argparse
import sys
from datetime import UTC, datetime
from pathlib import Path

import joblib
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from indusense.config import get_engine, get_model_path, get_model_version, get_predictions_engine
from indusense.modeling.dataset import load_gold_dataset
from indusense.predictions_store import upsert_predictions
from indusense.scoring import score_features

CHUNK_SIZE = 2000


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, default=None, help="Limite le nombre de lignes (test = tout le split)")
    args = parser.parse_args()

    model = joblib.load(get_model_path())
    model_version = get_model_version()

    gold = load_gold_dataset(get_engine())
    test_df = gold.test_df.sort_values(["machine_id", "window_start"])
    if args.limit:
        test_df = test_df.head(args.limit)

    proba, payloads = score_features(model, test_df)
    scored_at = datetime.now(UTC).isoformat()

    rows = [
        {
            "machine_id": machine_id,
            "window_start": pd.Timestamp(window_start).isoformat(),
            "failure_proba_24h": float(p),
            "scored_at": scored_at,
            "model_version": model_version,
            "features_payload": payload,
        }
        for machine_id, window_start, p, payload in zip(
            test_df["machine_id"], test_df["window_start"], proba, payloads
        )
    ]

    engine = get_predictions_engine()
    total = 0
    for i in range(0, len(rows), CHUNK_SIZE):
        total = upsert_predictions(engine, rows[i : i + CHUNK_SIZE])
        print(f"  {min(i + CHUNK_SIZE, len(rows))}/{len(rows)} lignes envoyées -> {total} en base")

    n_alertes = sum(1 for p in proba if p >= 0.5)
    print(f"Backfill terminé : {len(rows)} fenêtres scorées (modèle {model_version})")
    print(f"  dont {n_alertes} alertes (proba >= 0.5) sur {len(rows)} ({n_alertes / len(rows):.1%})")
    print(f"  {total} lignes en base au total")


if __name__ == "__main__":
    main()
