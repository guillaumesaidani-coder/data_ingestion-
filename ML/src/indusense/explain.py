"""Explicabilité du modèle de panne : une explication par score, et un
contrôle automatique du modèle contre la base de connaissance
(indusense/knowledge/base_connaissance.yaml).

Un seul module sert le script de certification, l'API (?explain=true,
garde-fou sur /ready) et les tests : aucune explication recodée ailleurs.

Décomposition exacte : pour XGBoost, `pred_contribs=True` donne les
valeurs SHAP exactes (TreeSHAP, calculé par XGBoost lui-même, aucune
bibliothèque ajoutée) : logit(score) = biais + somme des contributions.
Le biais est la sortie moyenne du modèle sur l'entraînement. Toute autre
forme de modèle est refusée (TypeError) plutôt qu'expliquée faussement.

Limites, à dire : une contribution décrit ce que le modèle a appris, pas
une cause physique ; elle est relative au biais (la machine « moyenne »
de l'entraînement) ; des variables corrélées (moyennes 6/12/24 h) se
partagent un effet de façon instable. La médiane citée dans les textes
est un repère (celle qui remplace une valeur manquante), pas la
référence du calcul.
"""

from functools import cache
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
import yaml
from scipy.stats import spearmanr
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

BASE_PATH = Path(__file__).resolve().parent / "knowledge" / "base_connaissance.yaml"
BIAIS = "biais"

BLOQUANT, AVERTISSEMENT, INFORMATION = "bloquant", "avertissement", "information"


@cache
def charger_base(path: Path = BASE_PATH) -> dict:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


# --------------------------------------------------------------- décomposition


def _etapes(model) -> tuple[SimpleImputer, xgb.XGBClassifier]:
    """Imputer + XGBoost (indusense.modeling.pipeline) : la seule forme
    de modèle dont la décomposition est exacte ici."""
    if not isinstance(model, Pipeline):
        raise TypeError(f"Explication exacte impossible pour {type(model).__name__}")
    imputer = model.named_steps.get("imputer")
    clf = model.named_steps.get("model")
    if not isinstance(imputer, SimpleImputer) or not isinstance(clf, xgb.XGBClassifier):
        raise TypeError(
            "Explication exacte réservée au pipeline imputer + XGBClassifier, "
            f"reçu {[type(s).__name__ for s in model.named_steps.values()]}"
        )
    return imputer, clf


def variables_du_modele(model) -> list[str]:
    imputer, _ = _etapes(model)
    return list(imputer.feature_names_in_)


def contributions(model, X: pd.DataFrame) -> pd.DataFrame:
    """Contribution de chaque variable au logit de chaque ligne, plus la
    colonne `biais` : leur somme sur une ligne est exactement le logit du
    score. Une variable écartée par l'imputer (vide à l'entraînement)
    reçoit 0 : le modèle ne la voit jamais."""
    imputer, clf = _etapes(model)
    cols = list(imputer.feature_names_in_)
    gardees = list(imputer.get_feature_names_out())
    Xt = imputer.transform(X.reindex(columns=cols))
    booster = clf.get_booster()
    raw = booster.predict(xgb.DMatrix(Xt, feature_names=booster.feature_names), pred_contribs=True)
    out = pd.DataFrame(0.0, index=X.index, columns=[*cols, BIAIS])
    out[gardees] = raw[:, :-1]
    out[BIAIS] = raw[:, -1]
    return out


def logit(model, X: pd.DataFrame) -> np.ndarray:
    imputer, clf = _etapes(model)
    Xt = imputer.transform(X.reindex(columns=list(imputer.feature_names_in_)))
    booster = clf.get_booster()
    return booster.predict(xgb.DMatrix(Xt, feature_names=booster.feature_names), output_margin=True)


# ------------------------------------------------------------------- textes


def _fmt(v: float) -> str:
    if float(v).is_integer():
        return f"{int(v)}"
    return f"{v:.3g}".replace(".", ",")


def _valeur(v: float, unite: str | None) -> str:
    if not unite:
        return _fmt(v)
    return f"{_fmt(v)}{unite}" if unite.startswith("/") else f"{_fmt(v)} {unite}"


def expliquer(model, ligne: pd.DataFrame, base: dict | None = None, k: int = 3) -> dict:
    """Pourquoi ce score : la décision et la règle appliquée, les k
    facteurs qui montent le risque et les k qui le baissent (libellés et
    unités de la base), et les avertissements qui fragilisent l'explication
    (valeur manquante remplacée, valeur hors de la plage physique)."""
    base = base or charger_base()
    imputer, _ = _etapes(model)
    cols = list(imputer.feature_names_in_)
    gardees = set(imputer.get_feature_names_out())
    medianes = dict(zip(cols, imputer.statistics_, strict=True))
    brute = ligne.reindex(columns=cols).iloc[0]

    contrib = contributions(model, ligne.iloc[[0]]).iloc[0]
    marge = float(contrib.sum())
    proba = float(model.predict_proba(ligne.reindex(columns=cols).iloc[[0]])[:, 1][0])

    seuil = base["regles"]["seuil_alerte"]["valeur"]
    if proba >= seuil:
        decision = (
            f"Alerte : probabilité de panne sous 24 h {_fmt(proba)} ≥ seuil {_fmt(seuil)}. "
            "À confirmer par un technicien, aucune action automatique."
        )
    else:
        decision = (
            f"Pas d'alerte : probabilité de panne sous 24 h {_fmt(proba)} < seuil {_fmt(seuil)}."
        )

    def facteur(var: str) -> dict:
        info = base["variables"].get(var, {})
        unite = info.get("unite")
        valeur = brute[var]
        c = float(contrib[var])
        vu = "manquante" if pd.isna(valeur) else _valeur(valeur, unite)
        texte = (
            f"{info.get('libelle', var)} : {vu} (médiane d'entraînement "
            f"{_valeur(medianes[var], unite)}), {'augmente' if c > 0 else 'diminue'} le risque"
        )
        return {
            "variable": var,
            "valeur": None if pd.isna(valeur) else float(valeur),
            "contribution": round(c, 4),
            "texte": texte,
        }

    variables = contrib.drop(BIAIS)
    hausse = [facteur(v) for v in variables[variables > 0].nlargest(k).index]
    baisse = [facteur(v) for v in variables[variables < 0].nsmallest(k).index]

    avertissements = []
    for var in cols:
        info = base["variables"].get(var, {})
        valeur = brute[var]
        if pd.isna(valeur):
            if var in gardees:
                avertissements.append(
                    f"{info.get('libelle', var)} : valeur manquante, remplacée par la "
                    f"médiane d'entraînement ({_valeur(medianes[var], info.get('unite'))})"
                )
            continue
        plage = info.get("plage")
        if plage and not plage[0] <= valeur <= plage[1]:
            avertissements.append(
                f"{info.get('libelle', var)} : {_valeur(valeur, info.get('unite'))} hors de la "
                f"plage physique [{_fmt(plage[0])}, {_fmt(plage[1])}] (contrat DAT §5.4)"
            )

    return {
        "decision": decision,
        "facteurs_hausse": hausse,
        "facteurs_baisse": baisse,
        "avertissements": avertissements,
        "logit": round(marge, 6),
        "biais": round(float(contrib[BIAIS]), 6),
    }


# ----------------------------------------------------------------- contrôles


def _constat(controle: str, gravite: str, message: str, variable: str | None = None) -> dict:
    return {"controle": controle, "gravite": gravite, "variable": variable, "message": message}


def controler_modele(
    variables: list[str],
    base: dict | None = None,
    parts: dict[str, float] | None = None,
    sens_observes: dict[str, float] | None = None,
    taux_manquants: dict[str, float] | None = None,
) -> list[dict]:
    """Confronte le modèle à la base. Sans données (au démarrage de
    l'API), seuls exclusion et couverture s'évaluent ; avec les mesures
    d'entraînement (certifier), s'y ajoutent sens, concentration par
    variable et par famille, et variables vides. Pas de contrôle de
    modalités : le modèle n'a aucune variable catégorielle."""
    base = base or charger_base()
    decrites, exclues = base["variables"], base["exclusions"]
    seuils = base["controles"]
    constats = []

    for var in variables:
        if var in exclues:
            e = exclues[var]
            constats.append(
                _constat(
                    "exclusion",
                    BLOQUANT,
                    f"Variable exclue ({e['categorie']}) : {e['raison']}",
                    var,
                )
            )
        elif var not in decrites:
            constats.append(
                _constat(
                    "couverture",
                    AVERTISSEMENT,
                    "Variable absente de la base : ni libellé, ni sens attendu",
                    var,
                )
            )

    for var, taux in (taux_manquants or {}).items():
        if taux >= 1.0:
            constats.append(
                _constat(
                    "informativite",
                    AVERTISSEMENT,
                    "Variable vide dans tout l'entraînement : écartée par l'imputer, construite pour rien",
                    var,
                )
            )

    if parts:
        for var, part in parts.items():
            if part > seuils["part_max_une_variable"]:
                constats.append(
                    _constat(
                        "concentration",
                        BLOQUANT,
                        f"Porte {part:.0%} de l'explication (max {seuils['part_max_une_variable']:.0%}) : fuite probable",
                        var,
                    )
                )
        familles: dict[str, float] = {}
        for var, part in parts.items():
            fam = decrites.get(var, {}).get("famille", "hors base")
            familles[fam] = familles.get(fam, 0.0) + part
        for fam, part in familles.items():
            if part > seuils["part_max_une_famille"]:
                constats.append(
                    _constat(
                        "concentration_famille",
                        AVERTISSEMENT,
                        f"La famille « {fam} » porte {part:.0%} de l'explication (max {seuils['part_max_une_famille']:.0%})",
                    )
                )

    for var, rho in (sens_observes or {}).items():
        attendu = decrites.get(var, {}).get("sens")
        if attendu not in ("hausse", "baisse") or abs(rho) < seuils["correlation_sens_min"]:
            continue
        # Le sens d'une variable qui ne pèse rien est du bruit, pas un constat.
        if parts and parts.get(var, 0.0) < seuils["part_negligeable"]:
            continue
        if (attendu == "hausse") != (rho > 0):
            constats.append(
                _constat(
                    "sens",
                    AVERTISSEMENT,
                    f"Sens attendu : {attendu} du risque ; appris : {'hausse' if rho > 0 else 'baisse'} "
                    f"(corrélation valeur / contribution {rho:+.2f})",
                    var,
                )
            )
    return constats


def est_conforme(constats: list[dict]) -> bool:
    return not any(c["gravite"] == BLOQUANT for c in constats)


def certifier(model, X: pd.DataFrame, base: dict | None = None, max_lignes: int = 50_000) -> dict:
    """Mesure sur l'entraînement ce que les contrôles demandent (part de
    chaque variable dans l'explication, sens appris, variables vides) et
    rend le verdict. La part = moyenne des |contributions|, rapportée au
    total ; le sens appris = corrélation de Spearman entre la valeur d'une
    variable et sa contribution (un arbre n'a pas de coefficient)."""
    base = base or charger_base()
    cols = variables_du_modele(model)
    taux_manquants = X.reindex(columns=cols).isna().mean().to_dict()
    if len(X) > max_lignes:
        X = X.sample(max_lignes, random_state=42)

    contrib = contributions(model, X)
    ecart = float(np.abs(contrib.sum(axis=1).to_numpy() - logit(model, X)).max())
    absolu = contrib[cols].abs().mean()
    parts = (absolu / absolu.sum()).to_dict()

    sens = {}
    for var in cols:
        valeurs = X[var]
        ok = valeurs.notna()
        if ok.sum() > 10 and valeurs[ok].nunique() > 1 and contrib.loc[ok, var].nunique() > 1:
            sens[var] = float(spearmanr(valeurs[ok], contrib.loc[ok, var]).statistic)

    constats = controler_modele(cols, base, parts, sens, taux_manquants)
    negligeables = sorted(
        v
        for v, p in parts.items()
        if p < base["controles"]["part_negligeable"] and taux_manquants.get(v, 0) < 1
    )
    if negligeables:
        constats.append(
            _constat(
                "informativite",
                INFORMATION,
                f"{len(negligeables)} variable(s) pèsent moins de "
                f"{base['controles']['part_negligeable']:.1%} chacune : {', '.join(negligeables)}",
            )
        )
    return {
        "conforme": est_conforme(constats),
        "constats": constats,
        "parts": {k: round(v, 5) for k, v in sorted(parts.items(), key=lambda kv: -kv[1])},
        "sens_observes": {k: round(v, 4) for k, v in sens.items()},
        "ecart_additivite_max": ecart,
        "n_lignes": len(X),
    }


# ------------------------------------------------------------------- dérive


def attribuer_derive(model, reference: pd.DataFrame, courant: pd.DataFrame) -> pd.DataFrame:
    """Décompose l'écart de logit moyen entre deux périodes, variable par
    variable (exact : le biais s'annule). Le PSI dit ce qui a bougé ; cette
    table dit ce qui a pesé sur le score."""
    ref = contributions(model, reference).mean()
    cur = contributions(model, courant).mean()
    table = pd.DataFrame({"reference": ref, "courant": cur}).drop(index=BIAIS)
    table["ecart"] = table["courant"] - table["reference"]
    total = table["ecart"].sum()
    table["part_de_l_ecart"] = table["ecart"] / total if total else 0.0
    return table.sort_values("ecart", key=np.abs, ascending=False)
