"""Garde-fou : empêche la régression de fuite trouvée dans generate_model_card.py
(leakage_cols n'excluait pas future_incident_count_*, un équivalent quasi direct
du label -> PR-AUC/ROC-AUC = 1.0 en re-entraînement, confirmé par diagnostic).

Ne teste pas une hypothèse de comportement "idéal" : verrouille la liste que
TP9.ipynb et TP11.ipynb traitent eux-mêmes comme la fuite connue.
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import generate_model_card as gmc

pytestmark = pytest.mark.requires_local_infra

KNOWN_LEAKAGE_COLS = {
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


def test_generate_model_card_excludes_known_leakage_columns():
    # load_data() renvoie (X_tv, y_tv, X_test, y_test, feature_cols, n_machines)
    feature_cols = gmc.load_data()[4]
    leaked = KNOWN_LEAKAGE_COLS & set(feature_cols)
    assert not leaked, f"Colonnes de fuite réintroduites dans feature_cols : {leaked}"
