#!/usr/bin/env python
"""Régénère `FEATURES` de `perf/locustfile.py` : une ligne réelle du Gold
(par défaut MACH-01, fenêtre 2026-04-14T01:00:00Z, split test), réduite
aux colonnes attendues par le modèle servi, dans l'ordre appris à
l'entraînement. Affiche le dict à recoller dans le locustfile — n'écrit
aucun fichier.

Usage : uv run --frozen python scripts/perf_sample_row.py [--machine MACH-01] [--window 2026-04-14T01:00:00Z]
"""

import argparse
import sys
from pathlib import Path

import joblib
import pandas as pd
from sqlalchemy import text

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from indusense.config import get_engine, get_model_path
from indusense.scoring import model_feature_cols


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--machine", default="MACH-01")
    parser.add_argument("--window", default="2026-04-14T01:00:00Z")
    args = parser.parse_args()

    df = pd.read_sql(
        text(
            "SELECT * FROM gold_machine_hourly_feature WHERE machine_id = :m AND window_start = :w"
        ),
        get_engine(),
        params={"m": args.machine, "w": args.window},
    )
    if len(df) != 1:
        sys.exit(f"{len(df)} ligne(s) Gold pour {args.machine} @ {args.window}, 1 attendue")

    row = df.iloc[0]
    cols = model_feature_cols(joblib.load(get_model_path()), list(df.columns))
    print(
        f"# {args.machine}, fenêtre {args.window}, split {row['split_set']} (gold_machine_hourly_feature)."
    )
    print("FEATURES = {")
    for c in cols:
        v = row[c]
        v = None if pd.isna(v) else v.item() if hasattr(v, "item") else v
        print(f"    {c!r}: {v!r},".replace("'", '"'))
    print("}")


if __name__ == "__main__":
    main()
