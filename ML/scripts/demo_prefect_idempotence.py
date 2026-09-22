#!/usr/bin/env python
"""Preuve d'idempotence des modules 29/30 : le flow est rejoué deux fois
sur les mêmes données, dans une base SQLite NEUVE créée pour l'occasion
(jamais une base déjà remplie, sinon la preuve serait truquée). Le
comptage de lignes est indépendant du flow lui-même (SELECT COUNT(*)
direct) — on ne fait pas confiance à la valeur que le flow prétend avoir
écrite.

Usage : uv run --frozen python scripts/demo_prefect_idempotence.py
"""

import sys
import tempfile
from pathlib import Path

from sqlalchemy import create_engine

ML_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ML_DIR / "src"))

from indusense.flows.predict_flow import DEFAULT_GOLD_CSV, indusense_pipeline
from indusense.predictions_store import count_predictions


def main() -> int:
    with tempfile.TemporaryDirectory(prefix="indusense-preuve-") as tmp:
        engine = create_engine(f"sqlite:///{Path(tmp) / 'predictions.db'}")
        try:
            result1 = indusense_pipeline(gold_csv=DEFAULT_GOLD_CSV, predictions_engine=engine)
            n1 = count_predictions(engine)
            print(f"1er passage : rows_scored={result1['rows_scored']} rows_in_db={n1}")

            result2 = indusense_pipeline(gold_csv=DEFAULT_GOLD_CSV, predictions_engine=engine)
            n2 = count_predictions(engine)
            print(f"2e  passage : rows_scored={result2['rows_scored']} rows_in_db={n2}")
        finally:
            # Meme piege que le module documente pour sqlite3 brut, ici via le
            # pool de connexions SQLAlchemy : sans dispose(), le fichier reste
            # verrouille et le nettoyage du TemporaryDirectory echoue sous
            # Windows (PermissionError [WinError 32]).
            engine.dispose()

        if n1 != n2:
            print(f"ECHEC idempotence : {n1} puis {n2} lignes ({engine.url})", file=sys.stderr)
            return 1

        print(f"OK idempotence : {n2} lignes stables apres 2 passages ({engine.url})")
        return 0


if __name__ == "__main__":
    sys.exit(main())
