"""Commandes `indusense export-gold` / `etl` / `train` / `predict`.

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
from indusense.data import export_gold_dataset, hash_file
from indusense.modeling.dataset import load_gold_dataset
from indusense.modeling.tracking import log_training_run, write_metrics_and_params
from indusense.modeling.train import (
    B11_PARAMS,
    compute_scale_pos_weight,
    train_and_evaluate,
)


def export_gold(args: argparse.Namespace) -> None:
    out_path = export_gold_dataset(get_engine(), Path(args.output))
    md5 = hash_file(out_path)
    print(f"Export : {out_path}")
    print(f"gold_md5 : {md5}")
    print(f"Suite : dvc add {out_path}")


def etl(args: argparse.Namespace) -> None:
    # Import local : Prefect n'est chargé que pour cette commande.
    from indusense.flows.etl_flow import etl_flow

    result = etl_flow()
    for table, n_rows in result.items():
        print(f"{table} : {n_rows} lignes")


def train(args: argparse.Namespace) -> None:
    gold = load_gold_dataset(get_engine())
    params = {**B11_PARAMS, "scale_pos_weight": compute_scale_pos_weight(gold.y_tv)}

    pipe, metrics = train_and_evaluate(gold.X_tv, gold.y_tv, gold.X_test, gold.y_test, params)

    print(
        f"PR-AUC train={metrics['pr_auc_train']}  PR-AUC test={metrics['pr_auc_test']}  "
        f"ROC-AUC test={metrics['roc_auc_test']}  F1 test={metrics['f1_test']}"
    )

    gold_md5 = hash_file(Path(args.gold_csv)) if args.gold_csv else None
    if args.gold_csv and not gold_md5:
        print(f"Attention : {args.gold_csv} introuvable, run tracé sans gold_md5")

    run_id = log_training_run(
        run_name="indusense-train-cli",
        params=params,
        metrics=metrics,
        gold_md5=gold_md5,
        gold_dataset_path=args.gold_csv,
    )
    print(f"Run MLflow : {run_id}" + (f"  (gold_md5={gold_md5})" if gold_md5 else ""))

    if args.metrics_out:
        metrics_path, params_path = write_metrics_and_params(
            metrics, params, Path(args.metrics_out), gold_md5=gold_md5
        )
        print(f"Écrits : {metrics_path}, {params_path}")

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

    export_gold_parser = subparsers.add_parser(
        "export-gold", help="Exporte gold_machine_hourly_feature en CSV versionnable par DVC"
    )
    export_gold_parser.add_argument(
        "-o", "--output", default="data/gold/gold_dataset.csv", help="Chemin de sortie du CSV"
    )
    export_gold_parser.set_defaults(func=export_gold)

    etl_parser = subparsers.add_parser(
        "etl", help="Reconstruit Silver et Gold à partir du Bronze (flow Prefect indusense-etl)"
    )
    etl_parser.set_defaults(func=etl)

    train_parser = subparsers.add_parser(
        "train", help="Entraîne et évalue le modèle b11-gkf sur le Gold dataset"
    )
    train_parser.add_argument(
        "-o", "--output", help="Chemin de sauvegarde du modèle entraîné (.joblib)"
    )
    train_parser.add_argument(
        "--gold-csv",
        help="Export CSV du Gold dataset (produit par `export-gold`) : son hash MD5 est "
        "tracé dans MLflow pour lier ce run à une version de données précise",
    )
    train_parser.add_argument(
        "--metrics-out",
        help="Répertoire où écrire metrics.json / params.yaml (versionnables par Git)",
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
