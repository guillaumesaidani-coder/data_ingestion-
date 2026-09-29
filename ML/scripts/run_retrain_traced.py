#!/usr/bin/env python
"""Réentraînement tracé : rejoue le cycle `indusense-retrain-cycle`
(feux verts -> arbitrage champion/challenger, règle d'or) et écrit
chaque opération dans logs/retrain_<horodatage>.log.

Avant le cycle, il vérifie que l'état n'a pas dérivé en silence. Il
compare d'abord le Gold en base au fichier data/gold/gold_dataset.csv,
puis contrôle le schéma de la table predictions et les retours
techniciens. Le champion (model.joblib) n'est jamais écrasé : son
empreinte est journalisée avant et après.

Usage : uv run --frozen python scripts/run_retrain_traced.py
"""

import hashlib
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
from sqlalchemy import text

ML_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ML_DIR / "src"))

from indusense.arbitration import CUTOFF, DEFAULT_CHALLENGER_PATH, DEFAULT_REPORT_CSV  # noqa: E402
from indusense.config import get_engine, get_model_path, get_predictions_engine, get_revision  # noqa: E402
from indusense.flows.retrain_flow import retrain_cycle  # noqa: E402
from indusense.modeling.dataset import TARGET, load_gold_dataset  # noqa: E402

GOLD_CSV = ML_DIR / "data" / "gold" / "gold_dataset.csv"
LOG_DIR = ML_DIR / "logs"


def _sha256(path: Path) -> str:
    if not path.exists():
        return "absent"
    return hashlib.sha256(path.read_bytes()).hexdigest()[:16]


def _setup_logging() -> Path:
    LOG_DIR.mkdir(exist_ok=True)
    log_path = LOG_DIR / f"retrain_{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}.log"
    fmt = logging.Formatter("%(asctime)s | %(levelname)-7s | %(name)s | %(message)s")
    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setFormatter(fmt)
    console = logging.StreamHandler(sys.stdout)
    console.setFormatter(fmt)
    # Les avertissements (pandas, sklearn) vont aussi dans le journal.
    logging.captureWarnings(True)
    for name in ("retrain", "prefect", "py.warnings"):
        logger = logging.getLogger(name)
        logger.setLevel(logging.INFO)
        logger.addHandler(file_handler)
    retrain_logger = logging.getLogger("retrain")
    retrain_logger.addHandler(console)
    # Prefect installe déjà un handler console sur la racine : sans ça,
    # chaque ligne s'affiche deux fois.
    retrain_logger.propagate = False
    return log_path


def _split_summary(df: pd.DataFrame) -> pd.DataFrame:
    return df.groupby("split_set").agg(n=(TARGET, "size"), pannes=(TARGET, "sum"))


def check_gold_consistency(log: logging.Logger, gold) -> bool:
    """Le Gold lu par le cycle (Postgres) doit être celui du CSV."""
    csv = pd.read_csv(GOLD_CSV, parse_dates=["window_start"])
    db = pd.concat([gold.trainval_df, gold.test_df])
    csv_summary = _split_summary(csv).astype(int)
    db_summary = _split_summary(db).astype(int)
    log.info("Gold CSV  (%s) :\n%s", GOLD_CSV.relative_to(ML_DIR), csv_summary)
    log.info("Gold base (gold_machine_hourly_feature) :\n%s", db_summary)
    same = csv_summary.equals(db_summary)
    log.info("Cohérence CSV/base : %s", "OK" if same else "ÉCART")
    return same


def check_predictions_state(log: logging.Logger) -> None:
    engine = get_predictions_engine()
    with engine.connect() as conn:
        cols = conn.execute(
            text("SELECT column_name FROM information_schema.columns WHERE table_name = 'predictions'")
        ).scalars().all()
        log.info("Table predictions : %d colonnes %s", len(cols), sorted(cols))
        rows = conn.execute(
            text("SELECT review_status, COUNT(*) FROM predictions GROUP BY review_status ORDER BY 1")
        ).all()
    for status, n in rows:
        log.info("  review_status=%s : %d", status, n)


def main() -> int:
    log_path = _setup_logging()
    log = logging.getLogger("retrain")
    log.info("=== Réentraînement tracé — journal %s ===", log_path.relative_to(ML_DIR))
    log.info("Révision du code : %s", get_revision())
    log.info("CUTOFF arbitrage : %s", CUTOFF)

    champion_path = Path(get_model_path())
    champion_before = _sha256(champion_path)
    challenger_before = _sha256(DEFAULT_CHALLENGER_PATH)
    log.info("Champion %s sha256=%s", champion_path, champion_before)
    log.info("Challenger précédent %s sha256=%s", DEFAULT_CHALLENGER_PATH, challenger_before)

    log.info("--- Étape 1 : cohérence de l'état ---")
    gold = load_gold_dataset(get_engine())
    if not check_gold_consistency(log, gold):
        log.error("Gold en base différent du CSV : réentraînement annulé (lancer make stack-etl ?)")
        return 1
    check_predictions_state(log)

    log.info("--- Étape 2 : cycle indusense-retrain-cycle (feux verts -> arbitrage) ---")
    result = retrain_cycle(CUTOFF)
    log.info("Statut du cycle : %s", result["status"])
    for name, (ok, detail) in result["gates"].items():
        log.info("  %s : %s -- %s", name, "OK" if ok else "SUSPENDU", detail)

    log.info("--- Étape 3 : état après le cycle ---")
    if result["status"] == "ARBITRE":
        last = pd.read_csv(DEFAULT_REPORT_CSV).iloc[-1]
        log.info("Dernière ligne %s :\n%s", DEFAULT_REPORT_CSV.relative_to(ML_DIR), last.to_string())
    champion_after = _sha256(champion_path)
    log.info(
        "Champion sha256=%s (%s)",
        champion_after,
        "inchangé" if champion_after == champion_before else "MODIFIÉ",
    )
    log.info("Challenger sha256=%s (avant : %s)", _sha256(DEFAULT_CHALLENGER_PATH), challenger_before)
    log.info("=== Fin — journal complet : %s ===", log_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
