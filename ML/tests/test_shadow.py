"""Mode fantôme (module 39) : zéro tolérance à la régression, bascule
réelle mais toujours réversible (sauvegarde avant écrasement).
"""

import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from indusense.shadow import (
    latest_arbitration,
    promote_challenger,
    run_shadow_cycle,
    shadow_confirms,
)


def test_shadow_confirms_true_without_any_regression():
    assert shadow_confirms({"gains": 3, "regressions": 0, "stables": 10, "angles_morts": 1})


def test_shadow_confirms_false_on_single_regression():
    # L'arbitrage (module 37) tolere jusqu'a 1 regression sous derogation ;
    # le mode fantome, dernier filet avant la vraie production, n'en tolere aucune.
    assert not shadow_confirms({"gains": 10, "regressions": 1, "stables": 5, "angles_morts": 0})


def test_shadow_confirms_false_when_no_gain_either():
    assert not shadow_confirms({"gains": 0, "regressions": 0, "stables": 20, "angles_morts": 0})


def test_promote_challenger_backs_up_before_overwriting(tmp_path):
    model_path = tmp_path / "model.joblib"
    challenger_path = tmp_path / "challenger.joblib"
    model_path.write_bytes(b"ancien-modele")
    challenger_path.write_bytes(b"nouveau-modele")

    backup_path = promote_challenger(challenger_path=challenger_path, model_path=model_path)

    assert backup_path.read_bytes() == b"ancien-modele"  # l'ancien modele est preserve
    assert model_path.read_bytes() == b"nouveau-modele"  # le pointeur sert desormais le challenger


def test_latest_arbitration_returns_none_without_log(tmp_path):
    assert latest_arbitration(tmp_path / "does_not_exist.csv") is None


def test_latest_arbitration_returns_last_row(tmp_path):
    log = tmp_path / "arbitration_log.csv"
    pd.DataFrame(
        [
            {"decision": "REJET_DU_MODELE_N", "gains": 1},
            {"decision": "ACCEPTATION_DIRECTE", "gains": 9},
        ]
    ).to_csv(log, index=False)
    assert latest_arbitration(log)["decision"] == "ACCEPTATION_DIRECTE"


def test_shadow_cycle_reports_no_arbitration_when_log_missing(tmp_path):
    result = run_shadow_cycle(arbitration_log=tmp_path / "missing.csv")
    assert result["status"] == "AUCUN_ARBITRAGE"


def test_shadow_cycle_blocks_when_last_decision_was_rejected(tmp_path):
    log = tmp_path / "arbitration_log.csv"
    pd.DataFrame([{"decision": "REJET_DU_MODELE_N"}]).to_csv(log, index=False)

    result = run_shadow_cycle(arbitration_log=log)
    assert result["status"] == "PAS_ELIGIBLE"
    assert "REJET_DU_MODELE_N" in result["detail"]


@pytest.mark.requires_local_infra
def test_shadow_cycle_promotes_a_genuinely_accepted_challenger(tmp_path, monkeypatch):
    """Le seul arbitrage réel de ce projet (modules 37-38) a conclu
    REJET_DU_MODELE_N -- honnête, pas arrangé, donc jamais promu en
    vrai. Ce test prouve séparément que le CHEMIN MÉCANIQUE (arbitrage
    accepté -> observation fantôme -> confirmation -> bascule de
    fichier réelle) fonctionne de bout en bout : `run_shadow_window` est
    remplacé par une matrice fixe (gains>0, 0 régression) pour isoler ce
    câblage de la performance ML elle-même, déjà testée ailleurs
    (test_arbitration.py, ce fichier). Nécessite Postgres local (Gold
    réel, pour que `run_shadow_cycle` puisse construire sa fenêtre)."""
    import joblib

    import indusense.shadow as shadow_module

    arbitration_log = tmp_path / "arbitration_log.csv"
    pd.DataFrame([{"decision": "ACCEPTATION_DIRECTE", "gains": 5, "regressions": 0}]).to_csv(
        arbitration_log, index=False
    )

    model_path = tmp_path / "model.joblib"
    challenger_path = tmp_path / "challenger.joblib"
    joblib.dump("modele-champion-factice", model_path)
    joblib.dump("modele-challenger-factice", challenger_path)

    monkeypatch.setattr(shadow_module, "get_model_path", lambda: model_path)
    monkeypatch.setattr(shadow_module, "DEFAULT_CHALLENGER_PATH", challenger_path)
    monkeypatch.setattr(
        shadow_module,
        "run_shadow_window",
        lambda champion, challenger, X, y_true: {
            "gains": 5,
            "stables": 100,
            "regressions": 0,
            "angles_morts": 2,
        },
    )

    result = run_shadow_cycle(
        arbitration_log=arbitration_log,
        shadow_log=tmp_path / "shadow_log.csv",
    )

    assert result["status"] == "PROMU"
    backup_path = Path(result["backup"])
    assert backup_path.exists()
    assert joblib.load(backup_path) == "modele-champion-factice"  # l'ancien modele est preserve
    assert joblib.load(model_path) == "modele-challenger-factice"  # le pointeur sert le challenger
    assert (tmp_path / "shadow_log.csv").exists()
