#!/usr/bin/env python
"""Modules 31-34 : republie en métriques Prometheus les CSV écrits par
`evaluate_drift.py` — relit le disque toutes les 15 s, ne recalcule
jamais rien lui-même (une seule source de vérité : les scripts de
dérive, ce module ne fait que les exposer sur :9110/metrics).

Port 9110, pas 9109 : un exporteur du formateur (démarré 2026-09-03,
données de démonstration "sous la jauge") occupe déjà 9109 en
permanence sur cette machine partagée — collision réelle rencontrée en
testant, pas une supposition (`netstat` : PID persistant sur 9109).

Usage : uv run --frozen python scripts/export_drift_metrics.py
"""

import sys
import time
from pathlib import Path

import pandas as pd
from prometheus_client import Gauge, start_http_server

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

DRIFT_DIR = Path(__file__).resolve().parent.parent / "reports" / "drift"
SUIVI_CSV = DRIFT_DIR / "suivi_fenetres.csv"
PORT = 9110
RELOAD_SECONDS = 15

drift_psi = Gauge("indusense_drift_psi", "PSI par feature et référence", ["feature", "reference", "fenetre"])
drift_ks_pvalue = Gauge(
    "indusense_drift_ks_pvalue", "p-value KS par feature et référence", ["feature", "reference", "fenetre"]
)
drift_alerte = Gauge("indusense_drift_alerte", "1 si PSI > seuil, 0 sinon", ["fenetre", "reference"])
drift_rappel = Gauge("indusense_drift_rappel", "Rappel du modèle mesuré sur la fenêtre", ["fenetre", "reference"])
drift_precision = Gauge(
    "indusense_drift_precision", "Précision du modèle mesurée sur la fenêtre", ["fenetre", "reference"]
)
drift_roc_auc = Gauge("indusense_drift_roc_auc", "ROC-AUC mesuré sur la fenêtre", ["fenetre", "reference"])


def _reload() -> int:
    """Relit les CSV du disque, met à jour les gauges. Retourne le nombre
    de fichiers psi_f*.csv lus (0 si rien n'a encore été produit —
    export_drift_metrics.py peut démarrer avant le premier evaluate_drift.py)."""
    n = 0
    for psi_csv in sorted(DRIFT_DIR.glob("psi_f*_ref-*.csv")):
        table = pd.read_csv(psi_csv)
        for _, row in table.iterrows():
            labels = {
                "feature": row["feature"],
                "reference": row["reference"],
                "fenetre": str(row["fenetre"]),
            }
            drift_psi.labels(**labels).set(row["psi"])
            drift_ks_pvalue.labels(**labels).set(row["ks_pvalue"])
        n += 1

    if SUIVI_CSV.exists():
        history = pd.read_csv(SUIVI_CSV)
        latest = history.sort_values("horodatage").groupby(["fenetre", "reference"], as_index=False).last()
        for _, row in latest.iterrows():
            labels = {"fenetre": str(row["fenetre"]), "reference": row["reference"]}
            drift_alerte.labels(**labels).set(1 if row["alerte_psi"] else 0)
            drift_rappel.labels(**labels).set(row["recall"])
            drift_precision.labels(**labels).set(row["precision"])
            drift_roc_auc.labels(**labels).set(row["roc_auc"])

    return n


def main() -> None:
    start_http_server(PORT)
    print(f"Exporteur dérive démarré sur :{PORT}/metrics (relecture toutes les {RELOAD_SECONDS}s)")
    while True:
        n = _reload()
        print(f"{n} fichier(s) psi_f*.csv relu(s) depuis {DRIFT_DIR}")
        time.sleep(RELOAD_SECONDS)


if __name__ == "__main__":
    main()
