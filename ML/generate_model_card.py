"""Génère la model card (format Hugging Face) du modèle de maintenance prédictive B11-GKF.

Ce script recharge les données, ré-entraîne B11-GKF avec les hyperparamètres figés
(Optuna, voir TP11.ipynb), recalcule les métriques de test en direct (jamais copiées
d'un ancien run), interroge MLflow pour l'artefact persisté (voir DL/TP9.ipynb), puis
écrit `artifacts/model_card.md` via le template officiel Hugging Face.

Le livrable est le fichier `.md` — ce script n'est que l'outil qui le produit.

Usage:
    uv run python generate_model_card.py
"""

from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from codecarbon import EmissionsTracker
import mlflow

from huggingface_hub import ModelCard, ModelCardData
from huggingface_hub.repocard_data import EvalResult

from indusense.config import get_engine
from indusense.modeling.dataset import TARGET, load_gold_dataset
from indusense.modeling.heterogeneity import (
    leave_one_group_out_proba,
    load_machine_reference,
    pr_auc_by,
)
from indusense.modeling.train import B11_PARAMS, train_and_evaluate

ARTIFACTS_DIR = Path("artifacts")
ARTIFACTS_DIR.mkdir(exist_ok=True)

LINEAGE = [
    {"model": "B5 (TP7)",       "note": "Baseline LogReg/RF/XGBoost, comparaison initiale"},
    {"model": "B7 (TP8)",       "note": "Optuna, PR-AUC val 0.8617 — FUITE via feature_row_id (ordre temporel)"},
    {"model": "B7c/B8 (TP8b)",  "note": "Fuite corrigée — PR-AUC val 0.7504, split temporel"},
    {"model": "B8-ES (TP8c)",   "note": "Early stopping — rejeté, pas d'amélioration"},
    {"model": "B9-GKF (TP9)",   "note": "GroupKFold introduit — PR-AUC CV 0.7792"},
    {"model": "B11-GKF (TP11)", "note": "Optuna re-tuné sur objectif GroupKFold — RETENU"},
    {"model": "B12-GKF (TP12)", "note": "Feature stacking — rejeté, moins bon que B11"},
]


def load_data():
    engine = get_engine()
    gold = load_gold_dataset(engine)
    machines = load_machine_reference(engine)
    n_machines = gold.trainval_df["machine_id"].nunique()

    return gold, machines, n_machines


def refit_and_measure(X_tv, y_tv, X_test, y_test, spw_global):
    best_params = {**B11_PARAMS, "scale_pos_weight": spw_global}

    tracker = EmissionsTracker(
        project_name="ml_b11_gkf_refit",
        output_dir=str(ARTIFACTS_DIR),
        measure_power_secs=1,
        log_level="error",
        save_to_file=True,
    )
    tracker.start()
    pipe, metrics = train_and_evaluate(X_tv, y_tv, X_test, y_test, best_params)
    emissions_kg = tracker.stop()
    carbon_data = tracker.final_emissions_data

    return pipe, best_params, metrics, emissions_kg, carbon_data


def top_feature_share(pipe):
    """Feature la plus importante du modèle réentraîné et sa part de
    l'importance totale (l'imputer écarte les colonnes entièrement vides,
    d'où get_feature_names_out plutôt que feature_names_in_)."""
    importances = pipe.named_steps["model"].feature_importances_
    names = pipe.named_steps["imputer"].get_feature_names_out()
    i = int(importances.argmax())
    return names[i], float(importances[i] / importances.sum())


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


def find_mlflow_artifact():
    """Interroge MLflow pour l'artefact B11-GKF loggé par DL/TP9.ipynb — jamais de run_id codé en dur."""
    mlflow.set_tracking_uri(f"sqlite:///{Path('mlflow_tp7.db').resolve().as_posix()}")
    client = mlflow.MlflowClient()
    experiment = client.get_experiment_by_name("TP7_maintenance_predictive")
    if experiment is None:
        return None, None

    runs = client.search_runs(
        experiment.experiment_id,
        filter_string="tags.mlflow.runName = 'XGBoost_GroupKFold_B11'",
        order_by=["start_time DESC"],
        max_results=1,
    )
    if not runs:
        return None, None

    run_id = runs[0].info.run_id
    return run_id, f"runs:/{run_id}/xgboost_b11_gkf"


def build_card(X_tv, X_test, y_test, feature_cols, n_machines, spw_global,
               best_params, metrics, emissions_kg, carbon_data, mlflow_model_uri,
               het, top_feature):
    bm = het["by_machine"]
    worst, best = bm.idxmin(), bm.idxmax()
    drop = het["type_held_out_drop"]
    worst_drop = drop.idxmax()
    by_type = " · ".join(f"{t} {v:.2f}" for t, v in het["by_type"].items())
    top_name, top_share = top_feature

    card_data = ModelCardData(
        model_name="indusense-xgb-maintenance-b11-gkf",
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

    repo_line = (
        "ML/ (ce dépôt) — TP11.ipynb (recherche HP) ; artefact persisté : "
        + (f"MLflow `{mlflow_model_uri}` (store `ML/mlflow_tp7.db`, voir DL/TP9.ipynb)"
           if mlflow_model_uri else "aucun — exécuter DL/TP9.ipynb pour en créer un")
    )

    if mlflow_model_uri:
        get_started_code = (
            "```python\n"
            "# Artefact persisté dans MLflow (voir DL/TP9.ipynb) — rechargement direct\n"
            "import mlflow.xgboost\n"
            "mlflow.set_tracking_uri('sqlite:///mlflow_tp7.db')\n"
            f"model = mlflow.xgboost.load_model('{mlflow_model_uri}')\n\n"
            "from sklearn.impute import SimpleImputer\n"
            "X_imputed = SimpleImputer(strategy='median').fit(X_train_val).transform(X_new)\n"
            "proba = model.predict_proba(X_imputed)[:, 1]  # risque de panne à 24h\n"
            "```"
        )
    else:
        get_started_code = (
            "```python\n"
            "# Aucun artefact persisté : ré-entraînement avec hyperparamètres figés (Optuna, TP11)\n"
            "from sklearn.pipeline import Pipeline\n"
            "from sklearn.impute import SimpleImputer\n"
            "from xgboost import XGBClassifier\n\n"
            "best_params = " + json.dumps(best_params) + "\n"
            "pipe = Pipeline([('imputer', SimpleImputer(strategy='median')),\n"
            "                  ('model', XGBClassifier(**best_params))])\n"
            "pipe.fit(X_train_val, y_train_val)\n"
            "proba = pipe.predict_proba(X_new)[:, 1]  # risque de panne à 24h\n"
            "```"
        )

    template_kwargs = dict(
        model_id="indusense-xgb-maintenance-b11-gkf",
        model_summary=(
            "Classifieur XGBoost prédisant le risque de panne d'une machine industrielle "
            "dans les 24h à venir, à partir de features télémétrie (température, pression, "
            "tension, vitesse de rotation) et d'historique d'incidents agrégés sur des "
            "fenêtres glissantes 6h/12h/24h."
        ),
        model_description=(
            f"Entraîné sur {len(X_tv):,} observations horaires ({n_machines} machines, "
            f"{len(feature_cols)} features), évalué en validation croisée GroupKFold (une "
            "machine exclue par fold, pour ne jamais évaluer sur une machine vue à "
            "l'entraînement) et sur un jeu de test chronologiquement postérieur. "
            "Hyperparamètres optimisés par Optuna (TPE, 30 essais) sur l'objectif "
            "GroupKFold — voir `TP11.ipynb`."
        ),
        developers="Guillaume Saïdani",
        model_type="XGBoost (gradient boosting), classification binaire, pipeline scikit-learn (imputation médiane + XGBClassifier)",
        language="n/a (données tabulaires, pas de NLP)",
        license="Usage interne — données propriétaires (télémétrie machine, non publiques)",
        base_model="Aucun — entraîné from scratch",
        repo=repo_line,
        direct_use=(
            "Scorer un enregistrement horaire machine (features télémétrie + historique "
            "incidents agrégé) et obtenir une probabilité de panne à 24h. Seuil de "
            "décision par défaut 0.5 (celui évalué ici) — à ajuster selon l'arbitrage "
            "rappel/précision métier avant tout déploiement (cf. TP8 §8 pour une démarche "
            "de calibration de seuil sur une version antérieure)."
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
            "- Interprétation de `predict_proba` comme une probabilité calibrée — "
            "aucune calibration (Platt/isotonic) n'a été appliquée.\n"
            "- Usage réglementaire ou de certification sécurité — aucune validation de ce type."
        ),
        bias_risks_limitations=(
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
            f"contre {het['mean']:.2f} en CV — {top_share:.0%} de l'importance sur une seule "
            f"feature (`{top_name}`). Persiste malgré une "
            "régularisation poussée (`reg_lambda`, `min_child_weight` élevés) ; qualifié de "
            "structurel, pas résolu par les hyperparamètres seuls.\n"
            "- **Historique de fuite de données** : une version antérieure (B7, TP8) "
            "incluait `feature_row_id`, un identifiant séquentiel corrélé à l'ordre "
            "temporel, qui gonflait le PR-AUC de +0.111. Corrigé depuis TP8b — retiré "
            "explicitement des colonnes de fuite — mais signale la fragilité du pipeline "
            "de features aux fuites indirectes.\n"
            f"- **Rappel modéré au seuil par défaut** : {metrics['recall_test']:.1%} de "
            f"rappel, {metrics['fn']} pannes non détectées sur {metrics['fn']+metrics['tp']} "
            "au seuil 0.5 — un seuil plus bas augmenterait le rappel au prix de plus de "
            "fausses alertes (arbitrage non refait ici pour B11).\n"
            "- **Tentative de normalisation par machine infructueuse** : une normalisation "
            "z-score par machine a dégradé la généralisation en GroupKFold (TP10) — "
            "confirme que XGBoost est déjà insensible à l'échelle des features, ne pas "
            "réintroduire cette étape."
        ),
        bias_recommendations=(
            "Ne jamais utiliser en décision automatique. Suivre la performance par machine "
            f"et par type de presse, pas seulement l'agrégat — une machine comme {worst} "
            "justifie une vigilance humaine renforcée plutôt qu'une confiance dans le score. "
            "Ré-entraîner avant de scorer un nouveau type de presse. Calibrer "
            "les probabilités (Platt/isotonic) avant tout usage nécessitant un score "
            "interprétable comme une probabilité réelle. Recalibrer le seuil de décision "
            "selon le coût métier faux négatif vs faux positif avant déploiement."
        ),
        get_started_code=get_started_code,
        training_data=(
            f"Table `gold_machine_hourly_feature` (PostgreSQL, base indusense_db) — "
            f"{len(X_tv):,} observations horaires (entraînement + validation), "
            f"{n_machines} machines, {len(feature_cols)} features (rolling 6h/12h/24h "
            "télémétrie + historique incidents agrégé 24h/7j). Split chronologique "
            "(quantiles 0.70/0.85 sur `window_start`), pas de KFold aléatoire (fuite "
            "temporelle sinon)."
        ),
        preprocessing="Imputation des valeurs manquantes par médiane (`SimpleImputer`), aucune normalisation (XGBoost invariant à l'échelle — confirmé empiriquement, TP12).",
        training_regime=f"fp32, XGBoost gradient boosting, hyperparamètres Optuna (TPE, 30 essais, objectif GroupKFold(5)), scale_pos_weight={spw_global} (déséquilibre de classes)",
        speeds_sizes_times=f"{carbon_data.duration:.1f}s pour un ré-entraînement complet sur {len(X_tv):,} lignes (poste de travail local, CPU)",
        testing_data=f"Même table, partition test chronologiquement postérieure — {len(X_test):,} lignes, {int(y_test.sum())} pannes positives.",
        testing_factors="Évaluation agrégée toutes machines confondues pour la métrique principale ; performance par machine disponible via validation croisée GroupKFold (hétérogénéité "
                        f"{bm.min():.2f}-{bm.max():.2f}) et par type de presse (voir limites).",
        testing_metrics="PR-AUC (average precision — préférée à l'accuracy vu le déséquilibre de classe), ROC-AUC, F1, matrice de confusion au seuil 0.5.",
        results=(
            f"PR-AUC train={metrics['pr_auc_train']:.4f} · PR-AUC test={metrics['pr_auc_test']:.4f} · "
            f"ROC-AUC test={metrics['roc_auc_test']:.4f} · F1 test={metrics['f1_test']:.4f} · "
            f"TP={metrics['tp']} TN={metrics['tn']} FP={metrics['fp']} FN={metrics['fn']} · "
            f"Précision={metrics['precision_test']:.1%} · Rappel={metrics['recall_test']:.1%}"
        ),
        results_summary=(
            f"PR-AUC test {metrics['pr_auc_test']:.2f} sur un problème fortement déséquilibré "
            f"(scale_pos_weight={spw_global:.0f}) — signal réel mais hétérogène selon les "
            "machines (voir limites). Écart train/CV important : le modèle généralise moins "
            "bien qu'il ne le suggère sur ses propres données d'entraînement."
        ),
        model_examination="Non réalisé — SHAP (TreeExplainer) recommandé en prochaine étape pour identifier les features dominantes par machine (cf. b7_optimisation_explicabilite.md, hors périmètre ici).",
        hardware_type="Intel Core i7-12700H (CPU) — entraînement XGBoost, pas de GPU requis",
        hours_used=f"{carbon_data.duration/3600:.4f} h (ré-entraînement complet mesuré ci-dessus)",
        cloud_provider="Aucun — poste de travail local",
        cloud_region=f"{carbon_data.country_name} ({carbon_data.country_iso_code})",
        co2_emitted=f"{emissions_kg*1000:.4f} gCO2eq ({carbon_data.energy_consumed*1000:.3f} Wh) — mesure CodeCarbon de ce ré-entraînement",
        model_specs=f"XGBoost : n_estimators=287, max_depth=9, learning_rate={best_params['learning_rate']:.4f}, objectif binaire pondéré (scale_pos_weight={spw_global})",
        compute_infrastructure=(
            "Poste de travail local, connexion PostgreSQL directe. "
            + (f"Artefact tracké et versionné dans MLflow (voir DL/TP9.ipynb) : {mlflow_model_uri}."
               if mlflow_model_uri else
               "Aucune pipeline CLI industrialisée pour ce modèle (contrairement au DL, cf. DL/scripts/run_vision_pipeline.py).")
        ),
        hardware_requirements="CPU suffisant — pas de dépendance GPU",
        software="xgboost 3.3.0, scikit-learn 1.9.0, optuna 4.9.0, mlflow 3.14.0, Python 3.13 (ML/.venv)",
        model_card_authors="Guillaume Saïdani",
        model_card_contact="guillaume.saidani@ext.aelion.fr",
    )

    return ModelCard.from_template(card_data, **template_kwargs)


def main():
    print("[1/5] Chargement des données (PostgreSQL, requête identique à TP11)...")
    gold, machines, n_machines = load_data()
    X_tv, y_tv, X_test, y_test = gold.X_tv, gold.y_tv, gold.X_test, gold.y_test
    feature_cols = gold.feature_cols
    spw_global = round((y_tv == 0).sum() / (y_tv == 1).sum(), 2)
    print(f"  Train+val={len(X_tv):,}  Test={len(X_test):,}  Features={len(feature_cols)}  Machines={n_machines}")

    print("[2/5] Ré-entraînement (hyperparamètres figés, Optuna TP11) + mesure CodeCarbon...")
    pipe, best_params, metrics, emissions_kg, carbon_data = refit_and_measure(X_tv, y_tv, X_test, y_test, spw_global)
    print(f"  PR-AUC test={metrics['pr_auc_test']}  ROC-AUC={metrics['roc_auc_test']}  F1={metrics['f1_test']}")
    print(f"  Émissions={emissions_kg*1000:.4f} gCO2eq")

    print("[3/5] Hétérogénéité : une machine cachée, puis un type de presse caché (~2 min)...")
    het = measure_heterogeneity(gold, machines)
    print(f"  PR-AUC CV machine={het['mean']:.4f} ±{het['std']:.4f}  type caché={het['type_held_out_mean']:.4f}")

    print("[4/5] Recherche de l'artefact MLflow (loggé par DL/TP9.ipynb)...")
    _, mlflow_model_uri = find_mlflow_artifact()
    print(f"  {'Trouvé : ' + mlflow_model_uri if mlflow_model_uri else 'Aucun artefact trouvé — exécuter DL/TP9.ipynb'}")

    print("[5/5] Génération de la model card (template Hugging Face officiel)...")
    card = build_card(X_tv, X_test, y_test, feature_cols, n_machines, spw_global,
                       best_params, metrics, emissions_kg, carbon_data, mlflow_model_uri,
                       het, top_feature_share(pipe))
    card.validate()
    card.save(ARTIFACTS_DIR / "model_card.md")
    print(f"  Model card sauvegardée : {ARTIFACTS_DIR / 'model_card.md'} ({len(str(card))} caractères)")


if __name__ == "__main__":
    main()
