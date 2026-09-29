"""generate_model_card.py : la card décrit le modèle servi et refuse de
s'écrire sur une certification qui ne lui correspond pas. Petit modèle
et données synthétiques, aucun Postgres : tourne en CI.
"""

import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import generate_model_card as gmc
from indusense.config import get_model_version
from indusense.modeling.dataset import TARGET


@pytest.fixture
def served_model(tmp_path, monkeypatch):
    X = pd.DataFrame({"incident_count_prev_24h": np.arange(40.0) % 3})
    y = (X["incident_count_prev_24h"] == 2).astype(int)
    pipe = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("model", XGBClassifier(n_estimators=5, max_depth=2, random_state=0)),
        ]
    ).fit(X, y)
    model_path = tmp_path / "model.joblib"
    joblib.dump(pipe, model_path)
    monkeypatch.setenv("MODEL_PATH", str(model_path))
    return model_path


def _write(path, payload):
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_card_refuses_missing_certification(served_model):
    with pytest.raises(SystemExit, match="certify_model.py"):
        gmc.load_served_model()


def test_card_refuses_certification_of_another_model(served_model):
    _write(served_model.parent / "explication_certification.json", {"model_version": "autre"})
    with pytest.raises(SystemExit, match="autre"):
        gmc.load_served_model()


def test_card_uses_certification_and_emissions_of_the_served_model(served_model):
    version = get_model_version(served_model)
    _write(served_model.parent / "explication_certification.json", {"model_version": version})
    _write(
        served_model.parent / "training_emissions.json",
        {"model_version": version, "emissions_g": 1.0},
    )

    _, _, got_version, certification, emissions = gmc.load_served_model()

    assert got_version == version == certification["model_version"]
    assert emissions["emissions_g"] == 1.0


def test_card_ignores_emissions_of_another_model(served_model):
    version = get_model_version(served_model)
    _write(served_model.parent / "explication_certification.json", {"model_version": version})
    _write(served_model.parent / "training_emissions.json", {"model_version": "autre"})

    assert gmc.load_served_model()[4] is None


def test_severity_precursor():
    df = pd.DataFrame(
        {
            gmc.SEVERITY_COL: [0, 0, 0, 0, 3, 3, 3, 3, 5, 5],
            TARGET: [0, 0, 0, 0, 1, 1, 1, 0, 1, 0],
        }
    )
    p = gmc.severity_precursor(df)
    assert p["rate_precursor"] == pytest.approx(0.75)
    assert p["rate_no_incident"] == 0
    assert p["share_failures"] == pytest.approx(0.75)
