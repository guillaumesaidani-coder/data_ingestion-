#!/usr/bin/env python
"""Module 37 : arbitrage champion vs challenger — façade CLI, la logique
vit dans `indusense.arbitration` (module 38 : `indusense.flows.retrain_flow`
appelle la même fonction, derrière des feux verts Prefect plutôt qu'à la
demande).

Usage : uv run --frozen python scripts/arbitrate_challenger.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from indusense.arbitration import DEFAULT_CHALLENGER_PATH, DEFAULT_REPORT_CSV, persist_arbitration, run_arbitration


def main() -> int:
    result = run_arbitration()

    print(
        f"Entraînement augmenté : {result['n_train_augmente']} lignes avril-mai revues par la "
        f"boucle HITL ({result['n_pannes_augmentees']} pannes confirmées/déclarées)"
    )
    print(f"Arbitrage sur {result['n_arbitrage']} fenêtres, jamais vues par aucun des deux modèles")
    print(
        f"Challenger (mesuré sur l'arbitrage) : rappel={result['challenger_metrics']['recall_test']} "
        f"précision={result['challenger_metrics']['precision_test']} "
        f"ROC-AUC={result['challenger_metrics']['roc_auc_test']}"
    )
    print(f"Gain pur (champion faux, challenger juste)   : +{result['gains']}")
    print(f"Stabilité (les deux justes)                  : {result['stables']}")
    print(f"Régression (champion juste, challenger faux) : -{result['regressions']}")
    print(f"Angle mort (les deux faux)                   : {result['angles_morts']}")
    print(f"Bilan net : {result['gains'] - result['regressions']}")
    print(f"Décision : {result['decision']}")

    persist_arbitration(result)
    print(f"Challenger sauvegardé -> {DEFAULT_CHALLENGER_PATH} (non activé — bascule séparée, module 39)")
    print(f"Arbitrage journalisé -> {DEFAULT_REPORT_CSV}")

    return 0 if result["decision"] != "REJET_DU_MODELE_N" else 1


if __name__ == "__main__":
    sys.exit(main())
