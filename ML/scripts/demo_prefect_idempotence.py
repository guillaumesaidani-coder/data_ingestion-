#!/usr/bin/env python
"""Preuve d'idempotence du module 29 : le flow est rejoué deux fois sur les
mêmes données, dans une base SQLite NEUVE créée pour l'occasion (jamais une
base déjà remplie, sinon la preuve serait truquée). Le comptage de lignes
est indépendant du flow lui-même (SELECT COUNT(*) direct) — on ne fait pas
confiance à la valeur que le flow prétend avoir écrite.

Usage : uv run --frozen python scripts/demo_prefect_idempotence.py
"""

import sys
import tempfile
from pathlib import Path

ML_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ML_DIR))
sys.path.insert(0, str(ML_DIR / "src"))

from flows.pipeline import DEFAULT_GOLD_CSV, indusense_pipeline
from indusense.predictions_store import count_predictions


def main() -> int:
    with tempfile.TemporaryDirectory(prefix="indusense-preuve-") as tmp:
        db_path = Path(tmp) / "predictions.db"

        result1 = indusense_pipeline(gold_csv=DEFAULT_GOLD_CSV, predictions_db=db_path)
        n1 = count_predictions(db_path)
        print(f"1er passage : rows_scored={result1['rows_scored']} rows_in_db={n1}")

        result2 = indusense_pipeline(gold_csv=DEFAULT_GOLD_CSV, predictions_db=db_path)
        n2 = count_predictions(db_path)
        print(f"2e  passage : rows_scored={result2['rows_scored']} rows_in_db={n2}")

        if n1 != n2:
            print(f"ECHEC idempotence : {n1} puis {n2} lignes ({db_path})", file=sys.stderr)
            return 1

        print(f"OK idempotence : {n2} lignes stables apres 2 passages ({db_path})")
        return 0


if __name__ == "__main__":
    sys.exit(main())
