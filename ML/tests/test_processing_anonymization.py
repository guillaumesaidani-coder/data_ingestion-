"""anon() extrait à l'identique de TP1.ipynb (SHA-256 salé, cf. justification
"Anonymisation — choix technique" dans le notebook : non réversible,
déterministe, légère).
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from indusense.processing.anonymization import SALT, anon


def test_anon_output_format():
    result = anon("Lucas Bernard")
    assert result.startswith("OP_ANON_")
    assert len(result) == len("OP_ANON_") + 8
    assert result[len("OP_ANON_"):].isupper()


def test_anon_is_deterministic():
    assert anon("Lucas Bernard") == anon("Lucas Bernard")


def test_anon_is_not_reversible_trivially():
    # Pas de round-trip possible : la sortie ne contient aucune trace du nom en clair.
    assert "Lucas" not in anon("Lucas Bernard")
    assert "Bernard" not in anon("Lucas Bernard")


def test_anon_distinguishes_different_names():
    assert anon("Lucas Bernard") != anon("Hugo Thomas")


def test_anon_matches_known_value_for_current_salt():
    # Valeur figée (pas recalculée) : un changement de SALT ou de la fonction de
    # hash casse ce test, signalant une rupture de compatibilité avec l'existant
    # (ex. releves_incidents_anonymised.csv déjà exporté avec ce SALT).
    assert SALT == "incidents_v1"
    assert anon("Lucas Bernard") == "OP_ANON_2C59BBEB"
