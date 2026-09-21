"""load_gold_dataset() extrait à l'identique de TP9.ipynb/TP11.ipynb (bloc
dupliqué mot pour mot dans les deux). Les mêmes colonnes de fuite que
tests/test_no_leakage_regression.py, ici verrouillées à la source unique
que TP9/TP11/generate_model_card.py partagent désormais.
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from indusense.modeling.dataset import LEAKAGE_COLS, TARGET, load_gold_dataset


def test_leakage_cols_is_the_known_correct_list():
    assert TARGET == "label_failure_next_24h"
    assert set(LEAKAGE_COLS) == {
        "machine_id",
        "ingestion_batch_id",
        "window_start",
        "window_end",
        "split_set",
        "label_failure_next_6h",
        "label_failure_next_12h",
        "label_failure_next_48h",
        "label_failure_next_24h",
        "feature_row_id",
        "future_incident_count_6h",
        "future_incident_count_12h",
        "future_incident_count_24h",
        "future_incident_count_48h",
    }


@pytest.mark.requires_local_infra
def test_load_gold_dataset_against_real_db(db_engine):
    gold = load_gold_dataset(db_engine)

    assert set(LEAKAGE_COLS).isdisjoint(gold.feature_cols)
    assert len(gold.X_tv) == 112_996
    assert len(gold.X_test) == 19_944
    assert list(gold.X_tv.columns) == gold.feature_cols
    assert gold.groups.nunique() == 15
    assert set(gold.y_tv.unique()) <= {True, False}
