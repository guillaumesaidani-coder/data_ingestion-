"""arbitration_matrix()/arbitration_decision() (module 37) : la règle
d'or de la feuille de route (§5) — jamais valider un candidat sur un
score global seul, toujours refuser une régression non compensée.
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from arbitrate_challenger import arbitration_decision, arbitration_matrix


def test_matrix_classifies_all_four_cases():
    y_true = np.array([1, 1, 1, 0])
    pred_champion = np.array([0, 1, 1, 0])  # rate le 1er, juste sur les 3 autres
    pred_challenger = np.array([1, 1, 0, 0])  # corrige le 1er, rate le 3e

    matrix = arbitration_matrix(y_true, pred_champion, pred_challenger)
    assert matrix == {"gains": 1, "stables": 2, "regressions": 1, "angles_morts": 0}


def test_direct_acceptance_requires_zero_regression():
    assert arbitration_decision(gains=5, regressions=0) == "ACCEPTATION_DIRECTE"
    assert (
        arbitration_decision(gains=0, regressions=0) == "REJET_DU_MODELE_N"
    )  # aucun gain, rien a valider


def test_derogation_requires_gains_at_least_triple_regressions():
    assert arbitration_decision(gains=4, regressions=1) == "ACCEPTATION_SOUS_DEROGATION"
    assert (
        arbitration_decision(gains=3, regressions=1) == "REJET_DU_MODELE_N"
    )  # pas strictement > 3x


def test_reject_when_regressions_too_high_even_with_net_positive_gain():
    # Cas reel observe (module 37, juin 2026) : gains=9, regressions=7 -- bilan net +2,
    # mais 7 regressions sur des pannes n'est jamais acceptable sans compensation massive.
    assert arbitration_decision(gains=9, regressions=7) == "REJET_DU_MODELE_N"


def test_reject_when_more_than_one_regression_even_above_ratio():
    # gains > 3x regressions mais regressions > 1 : la derogation exige aussi <= 1 regression
    assert arbitration_decision(gains=10, regressions=2) == "REJET_DU_MODELE_N"
