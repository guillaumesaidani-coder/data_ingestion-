"""operators.py extrait de TP4.ipynb §4 : fixtures synthétiques, sans
base. La preuve de non-régression (mêmes opérateurs et mêmes badge_hash
que la table `operator` produite par TP4) est dans
test_operators_match_tp4_table, marqué requires_local_infra.
"""

import hashlib
import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from indusense.processing.operators import OPERATOR_COLUMNS, badge_hash, build_operators

ML_DIR = Path(__file__).resolve().parent.parent


def _bronze(*rows) -> pd.DataFrame:
    """(operator_name, operator_badge, parse_ok) -> lignes Bronze incidents."""
    return pd.DataFrame(rows, columns=["operator_name", "operator_badge", "parse_ok"])


def test_badge_hash_is_tp4_algorithm():
    expected = hashlib.sha256(b"alice martin|b-042").hexdigest()[:16]
    assert badge_hash(" Alice Martin ", " B-042 ") == expected
    assert len(expected) == 16


def test_build_operators_normalizes_key_and_hashes_badge():
    ops = build_operators(_bronze((" Alice MARTIN ", "B-042", True)))

    assert list(ops.columns) == OPERATOR_COLUMNS
    assert ops.to_dict("records") == [
        {"operator_key": "alice martin", "badge_hash": badge_hash("Alice Martin", "B-042")}
    ]


def test_build_operators_one_row_per_key_first_wins():
    ops = build_operators(
        _bronze(("Alice", "B1", True), ("alice ", "B2", True), ("Bob", "B3", True))
    )

    assert ops["operator_key"].tolist() == ["alice", "bob"]
    assert ops.loc[0, "badge_hash"] == badge_hash("Alice", "B1")


def test_build_operators_ignores_rejected_rows():
    ops = build_operators(_bronze(("Alice", "B1", True), ("Alcie", "B1", False)))

    assert ops["operator_key"].tolist() == ["alice"]


@pytest.mark.parametrize("name", [None, "", "   "])
def test_build_operators_skips_empty_name(name):
    """operator_key NULL échapperait à la contrainte UNIQUE : pas d'opérateur."""
    assert build_operators(_bronze((name, "B1", True))).empty


def test_build_operators_missing_badge_gives_null_hash():
    ops = build_operators(_bronze(("Alice", None, True)))

    assert ops.loc[0, "operator_key"] == "alice"
    assert ops.loc[0, "badge_hash"] is None


@pytest.mark.requires_local_infra
def test_operators_match_tp4_table(db_engine):
    """releves_incidents.csv revalidé par le paquet -> exactement les
    opérateurs et badge_hash de la table `operator` (TP4 §4)."""
    from sqlalchemy import text

    from indusense.processing.bronze_validation import validate_bronze

    raw = pd.read_csv(ML_DIR / "releves_incidents.csv", dtype=str, encoding="utf-8")
    ours = build_operators(validate_bronze(raw, "incidents"))
    theirs = pd.read_sql(text("SELECT operator_key, badge_hash FROM operator"), db_engine)

    def _sorted(df):
        return df.sort_values("operator_key").reset_index(drop=True)

    assert len(ours) == 15
    pd.testing.assert_frame_equal(_sorted(ours), _sorted(theirs), check_dtype=False)
