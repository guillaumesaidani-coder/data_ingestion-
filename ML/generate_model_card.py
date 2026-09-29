"""Génère la model card (format Hugging Face) du modèle servi.

La card décrit le fichier que l'API sert (`artifacts/models/model.joblib`),
pas un modèle ré-entraîné à part : elle recharge ce fichier, le mesure sur
le Gold (PostgreSQL), et reprend son explication dans
`explication_certification.json` (scripts/certify_model.py). Elle refuse
de s'écrire si cette certification manque ou porte sur un autre
`model_version` : une card fausse est pire qu'une card absente.

Ordre après tout changement de modèle (entraînement ou bascule) :
    uv run --frozen python scripts/certify_model.py
    uv run --frozen python generate_model_card.py
(`make model-card` enchaîne les deux.)

Le livrable est le fichier `.md` — ce script n'est que l'outil qui le produit.
"""

from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

import joblib
from huggingface_hub import ModelCard, ModelCardData
from huggingface_hub.repocard_data import EvalResult

from indusense.config import get_engine, get_model_path, get_model_version
from indusense.explain import charger_base, parts_par_famille
from indusense.modeling.dataset import TARGET, load_gold_dataset
from indusense.modeling.heterogeneity import (
    leave_one_group_out_proba,
    load_machine_reference,
    pr_auc_by,
)
from indusense.modeling.train import B11_PARAMS, evaluate

ARTIFACTS_DIR = Path("artifacts")
ARTIFACTS_DIR.mkdir(exist_ok=True)

SEVERITY_COL = "incident_max_severity_prev_24h"
PRECURSOR_SEVERITY = 3


def load_data():
    engine = get_engine()
    gold = load_gold_dataset(engine)
    machines = load_machine_reference(engine)
    n_machines = gold.trainval_df["machine_id"].nunique()

    return gold, machines, n_machines


def load_served_model():
    """Le modèle servi, sa version et sa certification. Échoue si la
    certification manque ou décrit un autre fichier."""
    model_path = get_model_path()
    model = joblib.load(model_path)
    version = get_model_version(model_path)

    cert_path = model_path.parent / "explication_certification.json"
    if not cert_path.exists():
        raise SystemExit(f"{cert_path} absent : lancer scripts/certify_model.py d'abord")
    certification = json.loads(cert_path.read_text(encoding="utf-8"))
    if certification.get("model_version") != version:
        raise SystemExit(
            f"Certification faite pour {certification.get('model_version')}, modèle servi "
            f"{version} : relancer scripts/certify_model.py d'abord"
        )

    emissions_path = model_path.parent / "training_emissions.json"
    emissions = None
    if emissions_path.exists():
        emissions = json.loads(emissions_path.read_text(encoding="utf-8"))
        if emissions.get("model_version") != version:
            emissions = None

    return model_path, model, version, certification, emissions


def severity_precursor(df):
    """Ce que le modèle exploite surtout (cf. certification) : un incident
    de sévérité 3 dans les 24 h précédentes. Mesuré sur train+validation,
    les données dont il a appris."""
    y = df[TARGET].astype(int)
    sev = df[SEVERITY_COL]
    precursor = sev == PRECURSOR_SEVERITY
    return {
        "rate_precursor": y[precursor].mean(),
        "rate_no_incident": y[sev == 0].mean(),
        "share_failures": y[precursor].sum() / y.sum(),
    }


def measure_heterogeneity(gold, machines):
    """PR-AUC par machine en validation croisée « une machine cachée »
    (GroupKFold à un fold par machine) puis « un type de presse caché »,
    sur train+validation, avec les hyperparamètres b11."""
    df = gold.trainval_df.merge(machines, on="machine_id", how="left")
    if df["machine_type"].isna().any():
        missing = sorted(df.loc[df["machine_type"].isna(), "machine_id"].unique())
        raise ValueError(f"Machines absentes de la table machine : {missing}")

    by_machine = pr_auc_by(
        df, TARGET,
        leave_one_group_out_proba(df, gold.feature_cols, TARGET, "machine_id", B11_PARAMS),
        "machine_id",
    ).dropna()
    by_machine_type_held_out = pr_auc_by(
        df, TARGET,
        leave_one_group_out_proba(df, gold.feature_cols, TARGET, "machine_type", B11_PARAMS),
        "machine_id",
    ).dropna()
    machine_type = df.groupby("machine_id")["machine_type"].first()
    return {
        "by_machine": by_machine,
        "mean": by_machine.mean(),
        "std": by_machine.std(),
        "type_held_out_mean": by_machine_type_held_out.mean(),
        "type_held_out_drop": (by_machine - by_machine_type_held_out).dropna(),
        "by_type": by_machine.groupby(machine_type).mean(),
    }


def model_examination(certification, base):
    """Section « Examen du modèle » : reprise telle quelle de la
    certification, la même que l'API lit pour /ready."""
    parts = certification["parts"]
    libelles = {f: d["libelle"] for f, d in base["familles"].items()}
    familles = " · ".join(
        f"{libelles.get(f, f)} {p:.0%}" for f, p in parts_par_famille(parts, base).items() if p >= 0.01
    )
    top = "\n".join(f"  - `{v}` : {p:.1%}" for v, p in list(parts.items())[:6])
    constats = "\n".join(
        f"  - {c['gravite']} ({c['controle']}{', `' + c['variable'] + '`' if c['variable'] else ''}) : {c['message']}"
        for c in certification["constats"]
        if c["gravite"] != "information"
    )
    return (
        f"Certification du modèle `{certification['model_version']}` "
        f"(`scripts/certify_model.py`, {certification['n_lignes']:,} lignes d'entraînement) : "
        f"**{'conforme' if certification['conforme'] else 'BLOQUÉ'}**. Part de l'explication = "
        "moyenne des |contributions SHAP| (TreeSHAP exact d'XGBoost), rapportée au total.\n\n"
        f"- Par famille : {familles}.\n"
        f"- Variables les plus lourdes :\n{top}\n"
        f"- Constats de la certification :\n{constats or '  - aucun'}\n\n"
        "Une contribution décrit ce que le modèle a appris, pas une cause physique."
    )


def build_card(gold, n_machines, model_path, model, version, certification, emissions,
               metrics, het, precursor):
    base = charger_base()
    X_tv, X_test, y_test = gold.X_tv, gold.X_test, gold.y_test
    xgb_params = model.named_steps["model"].get_params()
    spw = xgb_params["scale_pos_weight"]
    n_features = len(model.named_steps["imputer"].feature_names_in_)

    bm = het["by_machine"]
    worst, best = bm.idxmin(), bm.idxmax()
    drop = het["type_held_out_drop"]
    worst_drop = drop.idxmax()
    by_type = " · ".join(f"{t} {v:.2f}" for t, v in het["by_type"].items())

    parts = certification["parts"]
    top_name, top_share = next(iter(parts.items()))
    familles = parts_par_famille(parts, base)
    incidents_share = familles.get("incidents", 0.0)

    card_data = ModelCardData(
        model_name="indusense-xgb-maintenance-b11-gkf",
        model_version=version,
        certification="conforme" if certification["conforme"] else "bloque",
        license="other",
        library_name="xgboost",
        tags=["tabular-classification", "predictive-maintenance", "xgboost", "manufacturing", "time-series-features"],
        eval_results=[
            EvalResult(
                task_type="binary-classification",
                dataset_type="indusense-gold-machine-hourly",
                dataset_name="Gold machine hourly features — indusense_db",
                metric_type="pr_auc",
                metric_value=metrics["pr_auc_test"],
                metric_name="PR-AUC (average precision, test chronologique)",
            ),
        ],
    )

    if emissions:
        speeds = f"{emissions['duration_s']:.1f}s pour l'entraînement de ce modèle sur {len(X_tv):,} lignes (poste de travail local, CPU)"
        hours = f"{emissions['duration_s']/3600:.4f} h (entraînement de ce modèle, `indusense train`)"
        region = emissions["country"]
        co2 = f"{emissions['emissions_g']:.4f} gCO2eq ({emissions['energy_wh']:.3f} Wh) — mesure CodeCarbon de l'entraînement de ce modèle"
    else:
        speeds = hours = region = "Non mesuré pour ce modèle (voir Carbon Emitted)"
        co2 = (
            f"Non mesuré pour `{version}` : la mesure CodeCarbon est prise par `indusense train` "
            "(fichier `training_emissions.json`), absente pour ce modèle entraîné avant cet ajout."
        )

    template_kwargs = dict(
        model_id="indusense-xgb-maintenance-b11-gkf",
        model_summary=(
            "Classifieur XGBoost qui estime le risque de panne (incident de sévérité ≥ 4) d'une "
            "machine dans les 24 h à venir. En pratique, il s'appuie surtout sur un précurseur : "
            f"un incident de sévérité {PRECURSOR_SEVERITY} déclaré dans les 24 h précédentes, qui "
            f"précède {precursor['share_failures']:.0%} des pannes de l'entraînement. L'historique "
            f"d'incidents porte {incidents_share:.0%} de sa décision ; la télémétrie pèse peu. "
            f"Modèle servi : `{version}`."
        ),
        model_description=(
            f"Fichier servi : `{model_path.as_posix()}` (versionné par DVC), `model_version` "
            f"`{version}` = SHA-256 tronqué du fichier, le même que l'API, les prédictions et la "
            "certification. Métriques mesurées en scorant ce fichier, sans ré-entraînement.\n\n"
            f"Entraîné sur {len(X_tv):,} observations horaires ({n_machines} machines, "
            f"{n_features} features). Cible : `{TARGET}`, vraie si un incident de sévérité ≥ 4 "
            "survient sur la machine dans les 24 h suivantes. Taux de panne observé à "
            f"l'entraînement : {precursor['rate_precursor']:.1%} après un incident de sévérité "
            f"{PRECURSOR_SEVERITY} dans les 24 h, {precursor['rate_no_incident']:.1%} sans aucun "
            "incident. Recette évaluée en validation croisée GroupKFold (une machine exclue par "
            "fold) ; hyperparamètres optimisés par Optuna (TPE, 30 essais) — voir `TP11.ipynb`."
        ),
        developers="Guillaume Saïdani",
        model_type="XGBoost (gradient boosting), classification binaire, pipeline scikit-learn (imputation médiane + XGBClassifier)",
        language="n/a (données tabulaires, pas de NLP)",
        license="Usage interne — données propriétaires (télémétrie machine, non publiques)",
        base_model="Aucun — entraîné from scratch",
        repo=(
            f"ML/ (ce dépôt) — `indusense train -o {model_path.as_posix()}` ; recherche "
            "d'hyperparamètres : TP11.ipynb ; certification : `scripts/certify_model.py`."
        ),
        direct_use=(
            "Scorer un enregistrement horaire machine et obtenir un risque de panne à 24 h. "
            f"Le score monte surtout après un incident de sévérité {PRECURSOR_SEVERITY} déclaré : "
            "une panne sans ce précurseur est rarement détectée. Seuil de décision par défaut "
            "0.5 (celui évalué ici) — à ajuster selon l'arbitrage rappel/précision métier."
        ),
        downstream_use=(
            "Alimentation d'un tableau de bord de maintenance prédictive classant les "
            "machines par risque décroissant, à l'usage d'un planificateur de maintenance "
            "humain — pas d'arrêt automatique de machine."
        ),
        out_of_scope_use=(
            "- Arrêt automatique ou décision de maintenance sans validation humaine.\n"
            "- Toute machine d'un type de presse absent de l'entraînement, sans "
            f"ré-entraînement : PR-AUC moyenne {het['mean']:.2f} quand seule la machine est "
            f"inconnue, {het['type_held_out_mean']:.2f} quand tout son type l'est (cf. "
            "limites). Une nouvelle machine d'un type déjà couvert reste dans le périmètre.\n"
            "- Détection d'une dégradation visible seulement dans les capteurs : le modèle "
            "réagit surtout aux incidents déclarés.\n"
            "- Interprétation de `predict_proba` comme une probabilité calibrée — "
            "aucune calibration (Platt/isotonic) n'a été appliquée.\n"
            "- Usage réglementaire ou de certification sécurité — aucune validation de ce type."
        ),
        bias_risks_limitations=(
            "- **Dépendance à la saisie des incidents** : "
            f"{precursor['share_failures']:.0%} des pannes de l'entraînement suivent un incident "
            f"de sévérité {PRECURSOR_SEVERITY} dans les 24 h. Un incident non saisi, ou saisi "
            "avec une autre sévérité, fait chuter le risque ; les pannes sans ce précurseur "
            "passent le plus souvent inaperçues.\n"
            "- **Performance hétérogène par machine** : PR-AUC en validation croisée "
            f"GroupKFold (une machine par fold) de {bm[worst]:.2f} ({worst}) à "
            f"{bm[best]:.2f} ({best}) — écart-type ±{het['std']:.2f} autour d'une moyenne "
            f"de {het['mean']:.2f}. Un score agrégé unique masque des machines où le "
            "modèle est nettement moins fiable.\n"
            "- **Types de presse différents, invisibles pour le modèle** : ni `machine_id` "
            "ni le type (`machine.model`) ne sont des entrées. Moyenne par type (une machine "
            f"cachée) : {by_type}. Quand tout un type est caché à l'entraînement, la "
            f"PR-AUC moyenne passe de {het['mean']:.2f} à {het['type_held_out_mean']:.2f} "
            f"(pire cas {worst_drop} : −{drop[worst_drop]:.2f}).\n"
            f"- **Sur-ajustement structurel** : PR-AUC train = {metrics['pr_auc_train']:.3f} "
            f"contre {het['mean']:.2f} en CV — `{top_name}` porte à elle seule {top_share:.0%} "
            "de l'explication (SHAP, certification). Persiste malgré une régularisation poussée "
            "(`reg_lambda`, `min_child_weight` élevés) ; qualifié de structurel, pas résolu par "
            "les hyperparamètres seuls.\n"
            "- **Historique de fuite de données** : une version antérieure (B7, TP8) "
            "incluait `feature_row_id`, un identifiant séquentiel corrélé à l'ordre "
            "temporel, qui gonflait le PR-AUC de +0.111. Corrigé depuis TP8b — retiré "
            "explicitement des colonnes de fuite — mais signale la fragilité du pipeline "
            "de features aux fuites indirectes.\n"
            f"- **Rappel au seuil par défaut** : {metrics['recall_test']:.1%} de "
            f"rappel, {metrics['fn']} pannes non détectées sur {metrics['fn']+metrics['tp']} "
            "au seuil 0.5 — un seuil plus bas augmenterait le rappel au prix de plus de "
            "fausses alertes.\n"
            "- **Tentative de normalisation par machine infructueuse** : une normalisation "
            "z-score par machine a dégradé la généralisation en GroupKFold (TP10) — "
            "confirme que XGBoost est déjà insensible à l'échelle des features, ne pas "
            "réintroduire cette étape."
        ),
        bias_recommendations=(
            "Ne jamais utiliser en décision automatique. Veiller à la saisie des incidents et "
            "de leur sévérité : c'est la première entrée du modèle. Suivre la performance par "
            f"machine et par type de presse, pas seulement l'agrégat — une machine comme {worst} "
            "justifie une vigilance humaine renforcée plutôt qu'une confiance dans le score. "
            "Ré-entraîner avant de scorer un nouveau type de presse. Calibrer les probabilités "
            "(Platt/isotonic) avant tout usage nécessitant une probabilité réelle. Comparer le "
            f"modèle à la règle « sévérité max 24 h = {PRECURSOR_SEVERITY} » : il ne vaut que "
            "s'il fait mieux qu'elle."
        ),
        get_started_code=(
            "```python\n"
            "# Le fichier servi par l'API (dvc pull pour le récupérer)\n"
            "import joblib\n"
            f"model = joblib.load('{model_path.as_posix()}')  # imputer + XGBClassifier\n"
            "proba = model.predict_proba(X_new)[:, 1]  # risque de panne à 24h\n"
            "```"
        ),
        training_data=(
            f"Table `gold_machine_hourly_feature` (PostgreSQL, base indusense_db) — "
            f"{len(X_tv):,} observations horaires (entraînement + validation), "
            f"{n_machines} machines, {n_features} features (rolling 6h/12h/24h "
            "télémétrie + historique incidents agrégé 24h/7j). Split chronologique "
            "(quantiles 0.70/0.85 sur `window_start`), pas de KFold aléatoire (fuite "
            "temporelle sinon)."
        ),
        preprocessing="Imputation des valeurs manquantes par médiane (`SimpleImputer`), aucune normalisation (XGBoost invariant à l'échelle — confirmé empiriquement, TP12).",
        training_regime=f"fp32, XGBoost gradient boosting, hyperparamètres Optuna (TPE, 30 essais, objectif GroupKFold(5)), scale_pos_weight={spw} (déséquilibre de classes)",
        speeds_sizes_times=speeds,
        testing_data=f"Même table, partition test chronologiquement postérieure — {len(X_test):,} lignes, {int(y_test.sum())} pannes positives.",
        testing_factors="Évaluation agrégée toutes machines confondues pour la métrique principale ; performance par machine disponible via validation croisée GroupKFold (hétérogénéité "
                        f"{bm.min():.2f}-{bm.max():.2f}) et par type de presse (voir limites).",
        testing_metrics="PR-AUC (average precision — préférée à l'accuracy vu le déséquilibre de classe), ROC-AUC, F1, matrice de confusion au seuil 0.5.",
        results=(
            f"Modèle `{version}` — "
            f"PR-AUC train={metrics['pr_auc_train']:.4f} · PR-AUC test={metrics['pr_auc_test']:.4f} · "
            f"ROC-AUC test={metrics['roc_auc_test']:.4f} · F1 test={metrics['f1_test']:.4f} · "
            f"TP={metrics['tp']} TN={metrics['tn']} FP={metrics['fp']} FN={metrics['fn']} · "
            f"Précision={metrics['precision_test']:.1%} · Rappel={metrics['recall_test']:.1%}"
        ),
        results_summary=(
            f"PR-AUC test {metrics['pr_auc_test']:.2f} sur un problème fortement déséquilibré "
            f"(scale_pos_weight={spw:.0f}) — signal réel, porté surtout par l'historique "
            "d'incidents, et hétérogène selon les machines (voir limites). Écart train/CV "
            "important : le modèle généralise moins bien qu'il ne le suggère sur ses propres "
            "données d'entraînement."
        ),
        model_examination=model_examination(certification, base),
        hardware_type="Intel Core i7-12700H (CPU) — entraînement XGBoost, pas de GPU requis",
        hours_used=hours,
        cloud_provider="Aucun — poste de travail local",
        cloud_region=region,
        co2_emitted=co2,
        model_specs=(
            f"XGBoost : n_estimators={xgb_params['n_estimators']}, max_depth={xgb_params['max_depth']}, "
            f"learning_rate={xgb_params['learning_rate']:.4f}, objectif binaire pondéré "
            f"(scale_pos_weight={spw})"
        ),
        compute_infrastructure=(
            "Poste de travail local, connexion PostgreSQL directe. Modèle versionné par DVC "
            f"(`{model_path.as_posix()}.dvc`), identifié par `model_version` `{version}`."
        ),
        hardware_requirements="CPU suffisant — pas de dépendance GPU",
        software="xgboost 3.3.0, scikit-learn 1.9.0, optuna 4.9.0, mlflow 3.14.0, Python 3.13 (ML/.venv)",
        model_card_authors="Guillaume Saïdani",
        model_card_contact="guillaume.saidani@ext.aelion.fr",
    )

    return ModelCard.from_template(card_data, **template_kwargs)


def main():
    print("[1/4] Modèle servi et sa certification...")
    model_path, model, version, certification, emissions = load_served_model()
    print(f"  {model_path} : model_version={version}, certification "
          f"{'conforme' if certification['conforme'] else 'BLOQUÉE'}")

    print("[2/4] Chargement des données et mesure du modèle servi (sans ré-entraînement)...")
    gold, machines, n_machines = load_data()
    metrics = evaluate(model, gold.X_tv, gold.y_tv, gold.X_test, gold.y_test)
    precursor = severity_precursor(gold.trainval_df)
    print(f"  PR-AUC test={metrics['pr_auc_test']}  ROC-AUC={metrics['roc_auc_test']}  F1={metrics['f1_test']}")
    print(f"  Pannes précédées d'une sévérité {PRECURSOR_SEVERITY} : {precursor['share_failures']:.1%}")

    print("[3/4] Hétérogénéité : une machine cachée, puis un type de presse caché (~2 min)...")
    het = measure_heterogeneity(gold, machines)
    print(f"  PR-AUC CV machine={het['mean']:.4f} ±{het['std']:.4f}  type caché={het['type_held_out_mean']:.4f}")

    print("[4/4] Génération de la model card (template Hugging Face officiel)...")
    shown_path = model_path.relative_to(Path.cwd()) if model_path.is_relative_to(Path.cwd()) else model_path
    card = build_card(gold, n_machines, shown_path, model, version, certification, emissions,
                      metrics, het, precursor)
    card.validate()
    card.save(ARTIFACTS_DIR / "model_card.md")
    print(f"  Model card sauvegardée : {ARTIFACTS_DIR / 'model_card.md'} ({len(str(card))} caractères)")


if __name__ == "__main__":
    main()
