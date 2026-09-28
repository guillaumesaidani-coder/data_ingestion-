#!/usr/bin/env python
"""Certifie le modèle servi contre la base de connaissance
(indusense/knowledge/base_connaissance.yaml) : mesure sur train+validation
la part de chaque variable dans l'explication, le sens appris et les
variables vides, puis applique les contrôles (indusense.explain).

Écrit artifacts/models/explication_certification.json, lu par l'API au
démarrage : /ready refuse un modèle dont la certification est bloquante
ou porte sur un autre fichier modèle (model_version différente).

--demo-fuite entraîne en plus, pour la démonstration, le même pipeline
avec la variable de fuite future_incident_count_24h : il doit être
bloqué. Ce modèle n'est ni sauvegardé ni utilisé ailleurs.

Usage :
    uv run --frozen python scripts/certify_model.py [--demo-fuite]
Code retour 1 si le modèle servi n'est pas conforme.
"""

import argparse
import json
import sys
from pathlib import Path

import joblib

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from indusense.config import get_engine, get_model_path, get_model_version
from indusense.explain import certifier
from indusense.modeling.dataset import load_gold_dataset
from indusense.modeling.train import (
    B11_PARAMS,
    compute_scale_pos_weight,
    train_and_evaluate,
)

FUITE = "future_incident_count_24h"


def afficher(titre: str, rapport: dict) -> None:
    print(f"\n=== {titre} : {'CONFORME' if rapport['conforme'] else 'BLOQUÉ'}")
    print(f"écart d'additivité max : {rapport['ecart_additivite_max']:.2e}")
    for var, part in list(rapport["parts"].items())[:6]:
        print(f"  {part:6.1%}  {var}")
    for c in rapport["constats"]:
        cible = f" [{c['variable']}]" if c["variable"] else ""
        print(f"- {c['gravite'].upper()} {c['controle']}{cible} : {c['message']}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--demo-fuite", action="store_true")
    args = parser.parse_args()

    model_path = get_model_path()
    model = joblib.load(model_path)
    gold = load_gold_dataset(get_engine())

    rapport = certifier(model, gold.X_tv)
    rapport = {"model_version": get_model_version(model_path), **rapport}
    out = model_path.parent / "explication_certification.json"
    out.write_text(json.dumps(rapport, ensure_ascii=False, indent=2), encoding="utf-8")
    afficher(f"Modèle servi ({rapport['model_version']})", rapport)
    print(f"\n-> {out}")

    if args.demo_fuite:
        cols = [*gold.feature_cols, FUITE]
        params = {**B11_PARAMS, "scale_pos_weight": compute_scale_pos_weight(gold.y_tv)}
        fuite, metrics = train_and_evaluate(
            gold.trainval_df[cols], gold.y_tv, gold.test_df[cols], gold.y_test, params
        )
        afficher(
            f"Démo : modèle avec {FUITE} (PR-AUC test {metrics['pr_auc_test']})",
            certifier(fuite, gold.trainval_df[cols]),
        )

    return 0 if rapport["conforme"] else 1


if __name__ == "__main__":
    sys.exit(main())
