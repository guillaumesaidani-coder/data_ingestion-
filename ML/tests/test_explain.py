"""indusense.explain : décomposition exacte des scores, explications et
contrôles du modèle contre la base de connaissance. Données synthétiques,
aucun Postgres ni vrai modèle : tourne en CI.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from indusense.explain import (
    BIAIS,
    attribuer_derive,
    certifier,
    charger_base,
    contributions,
    controler_modele,
    est_conforme,
    expliquer,
    logit,
    parts_par_famille,
)
from indusense.modeling.dataset import LEAKAGE_COLS
from indusense.processing.gold_features import GOLD_COLS

VARS = ["incident_count_prev_24h", "temp_mean_24h", "pressure_mean_24h", "temp_std_1h"]


def _data(n=400, seed=0):
    rng = np.random.default_rng(seed)
    X = pd.DataFrame(
        {
            "incident_count_prev_24h": rng.poisson(1.0, n).astype(float),
            "temp_mean_24h": rng.normal(50, 5, n),
            "pressure_mean_24h": rng.normal(200, 10, n),
            "temp_std_1h": np.nan,  # vide, comme dans le vrai Gold
        }
    )
    y = ((X["incident_count_prev_24h"] >= 2) & (X["temp_mean_24h"] > 48)).astype(int)
    return X, y


def _pipe(X, y):
    pipe = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("model", XGBClassifier(n_estimators=30, max_depth=3, random_state=0)),
        ]
    )
    return pipe.fit(X, y)


@pytest.fixture(scope="module")
def modele():
    X, y = _data()
    return _pipe(X, y), X


# ------------------------------------------------------------ base / code


def test_base_decrit_exactement_les_variables_du_gold():
    base = charger_base()
    attendues = set(GOLD_COLS) - set(LEAKAGE_COLS)
    assert set(base["variables"]) == attendues


def test_base_couvre_toutes_les_exclusions_du_code():
    assert set(LEAKAGE_COLS) <= set(charger_base()["exclusions"])


def test_base_est_bien_formee():
    base = charger_base()
    for var, info in base["variables"].items():
        assert info["famille"] in base["familles"], var
        assert info["sens"] in ("hausse", "baisse", "indetermine"), var
        if "plage" in info:
            assert info["plage"][0] < info["plage"][1], var
    for var, info in base["exclusions"].items():
        assert info["categorie"] in ("identifiant", "technique", "cible", "fuite"), var


# ------------------------------------------------------------ décomposition


def test_somme_des_contributions_egale_le_logit(modele):
    pipe, X = modele
    contrib = contributions(pipe, X)
    assert np.abs(contrib.sum(axis=1).to_numpy() - logit(pipe, X)).max() < 1e-4


def test_variable_vide_ne_contribue_pas(modele):
    pipe, X = modele
    assert (contributions(pipe, X)["temp_std_1h"] == 0).all()


def test_refuse_un_modele_non_decomposable():
    X, y = _data()
    pipe = Pipeline(
        [("imputer", SimpleImputer(strategy="median")), ("model", LogisticRegression())]
    )
    pipe.fit(X.drop(columns="temp_std_1h"), y)
    with pytest.raises(TypeError):
        contributions(pipe, X)


# ------------------------------------------------------------ explication


def test_explication_separe_hausse_et_baisse(modele):
    pipe, X = modele
    ligne = X.iloc[[0]].assign(incident_count_prev_24h=4.0, temp_mean_24h=60.0)
    e = expliquer(pipe, ligne)
    assert e["facteurs_hausse"] and all(f["contribution"] > 0 for f in e["facteurs_hausse"])
    assert all(f["contribution"] < 0 for f in e["facteurs_baisse"])
    assert "Température moyenne sur 24 h" in " ".join(
        f["texte"] for f in e["facteurs_hausse"] + e["facteurs_baisse"]
    )
    assert e["decision"].startswith(("Alerte", "Pas d'alerte"))


def test_explication_signale_valeur_manquante_et_hors_plage(modele):
    pipe, X = modele
    ligne = X.iloc[[0]].assign(pressure_mean_24h=np.nan, temp_mean_24h=950.0)
    avert = " ".join(expliquer(pipe, ligne)["avertissements"])
    assert "Pression moyenne sur 24 h : valeur manquante" in avert
    assert "hors de la plage physique [-20, 200]" in avert
    # Variable vide à l'entraînement : jamais vue par le modèle, pas d'avertissement.
    assert "Écart-type de température sur 1 h" not in avert


# ------------------------------------------------------------ contrôles


def test_variable_exclue_bloque():
    constats = controler_modele(["temp_mean_24h", "future_incident_count_24h"])
    assert not est_conforme(constats)
    assert constats[0]["controle"] == "exclusion"


def test_variable_non_decrite_avertit_sans_bloquer():
    constats = controler_modele(["temp_mean_24h", "nouvelle_colonne_gold"])
    assert est_conforme(constats)
    assert [c["controle"] for c in constats] == ["couverture"]


def test_concentration_sur_une_variable_bloque():
    parts = {"incident_count_prev_24h": 0.7, "temp_mean_24h": 0.3}
    constats = controler_modele(list(parts), parts=parts)
    assert not est_conforme(constats)
    assert {"concentration", "concentration_famille"} <= {c["controle"] for c in constats}


def test_sens_contraire_avertit_sauf_variable_negligeable():
    parts = {"temp_mean_24h": 0.5, "temp_max_24h": 0.0001, "pressure_mean_24h": 0.4999}
    sens = {"temp_mean_24h": -0.4, "temp_max_24h": -0.9, "pressure_mean_24h": -0.9}
    constats = controler_modele(list(parts), parts=parts, sens_observes=sens)
    # pressure : sens indéterminé ; temp_max : ne pèse rien.
    assert [(c["controle"], c["variable"]) for c in constats if c["controle"] == "sens"] == [
        ("sens", "temp_mean_24h")
    ]


def test_parts_par_famille_somme_et_trie():
    parts = {"incident_count_prev_24h": 0.5, "incident_count_prev_7d": 0.2, "temp_mean_24h": 0.3}
    familles = parts_par_famille(parts)
    assert list(familles) == ["incidents", "temperature"]
    assert familles["incidents"] == pytest.approx(0.7)


def test_certifier_detecte_la_variable_vide_et_mesure_les_parts(modele):
    pipe, X = modele
    rapport = certifier(pipe, X)
    assert rapport["conforme"] is False  # incident_count porte l'essentiel ici
    assert abs(sum(rapport["parts"].values()) - 1) < 1e-6
    assert rapport["ecart_additivite_max"] < 1e-4
    vides = [c["variable"] for c in rapport["constats"] if c["controle"] == "informativite"]
    assert "temp_std_1h" in vides


# ------------------------------------------------------------ dérive


def test_attribution_de_derive_est_exacte(modele):
    pipe, X = modele
    courant = X.assign(incident_count_prev_24h=X["incident_count_prev_24h"] + 1)
    table = attribuer_derive(pipe, X, courant)
    ecart_logit = logit(pipe, courant).mean() - logit(pipe, X).mean()
    assert table["ecart"].sum() == pytest.approx(ecart_logit, abs=1e-4)
    assert table.index[0] == "incident_count_prev_24h"
    assert BIAIS not in table.index
