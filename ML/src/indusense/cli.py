"""Commandes `indusense train` / `indusense predict`.

Fine couche d'orchestration au-dessus de indusense.modeling : charge les
données, appelle train_and_evaluate() / un modèle sauvegardé, affiche ou
écrit le résultat. Aucune nouvelle logique métier — tout est délégué aux
modules déjà testés.
"""

import argparse
from pathlib import Path

import joblib
import pandas as pd

from indusense.config import get_engine
from indusense.modeling.dataset import load_gold_dataset
from indusense.modeling.train import (
    B11_PARAMS,
    compute_scale_pos_weight,
    train_and_evaluate,
)


def train(args: argparse.Namespace) -> None:
    gold = load_gold_dataset(get_engine())
    params = {**B11_PARAMS, "scale_pos_weight": compute_scale_pos_weight(gold.y_tv)}

    pipe, metrics = train_and_evaluate(gold.X_tv, gold.y_tv, gold.X_test, gold.y_test, params)

    print(
        f"PR-AUC train={metrics['pr_auc_train']}  PR-AUC test={metrics['pr_auc_test']}  "
        f"ROC-AUC test={metrics['roc_auc_test']}  F1 test={metrics['f1_test']}"
    )

    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(pipe, output_path)
        print(f"Modèle sauvegardé : {output_path}")


def predict(args: argparse.Namespace) -> None:
    pipe = joblib.load(args.model)
    X_new = pd.read_csv(args.input)

    proba = pipe.predict_proba(X_new)[:, 1]
    result = X_new.copy()
    result["failure_proba_24h"] = proba

    if args.output:
        result.to_csv(args.output, index=False)
        print(f"Prédictions écrites : {args.output}")
    else:
        print(result[["failure_proba_24h"]].to_csv(index=False))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="indusense", description="Maintenance prédictive InduSense"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    train_parser = subparsers.add_parser(
        "train", help="Entraîne et évalue le modèle b11-gkf sur le Gold dataset"
    )
    train_parser.add_argument(
        "-o", "--output", help="Chemin de sauvegarde du modèle entraîné (.joblib)"
    )
    train_parser.set_defaults(func=train)

    predict_parser = subparsers.add_parser(
        "predict", help="Score un CSV de features avec un modèle sauvegardé"
    )
    predict_parser.add_argument(
        "model", help="Chemin du modèle sauvegardé (.joblib, produit par `train -o`)"
    )
    predict_parser.add_argument(
        "input", help="CSV de features (mêmes colonnes que le Gold dataset)"
    )
    predict_parser.add_argument(
        "-o", "--output", help="Chemin du CSV de sortie (sinon affiché sur stdout)"
    )
    predict_parser.set_defaults(func=predict)

    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
