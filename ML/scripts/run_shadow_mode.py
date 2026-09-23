#!/usr/bin/env python
"""Module 39 : mode fantôme — façade CLI, la logique vit dans
`indusense.shadow`. Ne fait rien si le dernier arbitrage (module 37-38)
n'a pas accepté de challenger.

Usage : uv run --frozen python scripts/run_shadow_mode.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from indusense.shadow import run_shadow_cycle


def main() -> int:
    result = run_shadow_cycle()
    print(f"Statut : {result['status']}")
    if result["status"] in ("AUCUN_ARBITRAGE", "PAS_ELIGIBLE"):
        print(result["detail"])
        return 1
    m = result["matrix"]
    print(
        f"Observation fantôme : gains=+{m['gains']} stables={m['stables']} "
        f"régressions=-{m['regressions']} angles_morts={m['angles_morts']}"
    )
    if result["status"] == "PROMU":
        print(f"Bascule effectuée -- ancien modèle sauvegardé -> {result['backup']}")
        return 0
    print("Bascule bloquée -- au moins une régression observée en fantôme, zéro tolérance à ce stade")
    return 1


if __name__ == "__main__":
    sys.exit(main())
