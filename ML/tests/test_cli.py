"""Fine couche d'orchestration : on teste le parsing des arguments et le
câblage stdin/stdout/fichier de predict(), pas l'entraînement réel
(déjà couvert par test_modeling_train.py et vérifié manuellement via
`uv run indusense train`)."""

import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from indusense import cli


def test_train_subcommand_parses_and_routes():
    parser = cli.build_parser()
    args = parser.parse_args(["train", "-o", "out.joblib"])
    assert args.func is cli.train
    assert args.output == "out.joblib"


def test_train_subcommand_output_is_optional():
    parser = cli.build_parser()
    args = parser.parse_args(["train"])
    assert args.output is None


def test_etl_subcommand_parses_and_routes():
    parser = cli.build_parser()
    args = parser.parse_args(["etl"])
    assert args.func is cli.etl


def test_predict_subcommand_requires_model_and_input():
    parser = cli.build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["predict"])

    args = parser.parse_args(["predict", "model.joblib", "input.csv"])
    assert args.func is cli.predict
    assert args.model == "model.joblib"
    assert args.input == "input.csv"


class _StubPipe:
    def predict_proba(self, X):
        import numpy as np

        return np.tile([0.7, 0.3], (len(X), 1))


def test_predict_writes_csv_with_proba_column(tmp_path, monkeypatch):
    input_csv = tmp_path / "in.csv"
    pd.DataFrame({"f1": [1, 2, 3]}).to_csv(input_csv, index=False)
    output_csv = tmp_path / "out.csv"

    monkeypatch.setattr(cli.joblib, "load", lambda path: _StubPipe())

    args = cli.build_parser().parse_args(
        ["predict", "dummy_model.joblib", str(input_csv), "-o", str(output_csv)]
    )
    args.func(args)

    result = pd.read_csv(output_csv)
    assert list(result["failure_proba_24h"]) == [0.3, 0.3, 0.3]
    assert list(result["f1"]) == [1, 2, 3]
