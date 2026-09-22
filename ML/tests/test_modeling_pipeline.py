import sys
from pathlib import Path

from sklearn.impute import SimpleImputer
from xgboost import XGBClassifier

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from indusense.modeling.pipeline import build_xgb_pipeline


def test_build_xgb_pipeline_steps_and_params():
    params = {"n_estimators": 42, "max_depth": 3, "random_state": 42}
    pipe = build_xgb_pipeline(params)

    assert [name for name, _ in pipe.steps] == ["imputer", "model"]
    assert isinstance(pipe.named_steps["imputer"], SimpleImputer)
    assert pipe.named_steps["imputer"].strategy == "median"
    assert isinstance(pipe.named_steps["model"], XGBClassifier)
    assert pipe.named_steps["model"].get_params()["n_estimators"] == 42
    assert pipe.named_steps["model"].get_params()["max_depth"] == 3
