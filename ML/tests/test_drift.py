"""drift.py : PSI + KS. Vérifie les deux propriétés dont dépend tout le
reste du sous-système (drift_windows/evaluate_drift/drift_spec) — un PSI
quasi nul sur une distribution inchangée, un PSI élevé sur une dérive
franche, et les bins figés sur la référence seule.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from indusense.drift import drift_table, ks_pvalue, psi, reference_bin_edges

rng = np.random.default_rng(42)


def test_psi_near_zero_on_identical_distribution():
    reference = pd.Series(rng.normal(50, 5, 2000))
    current = pd.Series(rng.normal(50, 5, 2000))
    assert psi(reference, current) < 0.05


def test_psi_high_on_shifted_distribution():
    reference = pd.Series(rng.normal(50, 5, 2000))
    current = pd.Series(rng.normal(58, 5, 2000))  # +8, meme ordre que le scenario capteur
    assert psi(reference, current) > 0.25


def test_ks_pvalue_low_on_shifted_distribution():
    reference = pd.Series(rng.normal(50, 5, 2000))
    current = pd.Series(rng.normal(58, 5, 2000))
    assert ks_pvalue(reference, current) < 0.01


def test_ks_pvalue_high_on_identical_distribution():
    reference = pd.Series(rng.normal(50, 5, 2000))
    current = pd.Series(rng.normal(50, 5, 2000))
    assert ks_pvalue(reference, current) > 0.05


def test_bin_edges_are_frozen_on_reference_only():
    reference = pd.Series(rng.normal(50, 5, 2000))
    edges_before = reference_bin_edges(reference)
    # une fenetre courante tres differente ne doit jamais changer les bornes
    _ = psi(reference, pd.Series(rng.normal(200, 50, 500)))
    edges_after = reference_bin_edges(reference)
    np.testing.assert_array_equal(edges_before, edges_after)


def test_psi_is_not_symmetric_free_of_nans_with_missing_values():
    reference = pd.Series([1.0, 2.0, 3.0, np.nan, 4.0, 5.0] * 50)
    current = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0, np.nan] * 50)
    value = psi(reference, current)
    assert np.isfinite(value)


def test_drift_table_has_one_row_per_feature():
    reference = pd.DataFrame({"temp": rng.normal(50, 5, 500), "pressure": rng.normal(10, 1, 500)})
    current = pd.DataFrame({"temp": rng.normal(50, 5, 500), "pressure": rng.normal(10, 1, 500)})
    table = drift_table(reference, current, ["temp", "pressure"])
    assert list(table["feature"]) == ["temp", "pressure"]
    assert {"psi", "ks_pvalue"}.issubset(table.columns)
