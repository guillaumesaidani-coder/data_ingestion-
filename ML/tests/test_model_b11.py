"""Vérifie que le modèle documenté comme final (indusense-xgb-maintenance-b11-gkf,
artifacts/model_card.md) est bien celui tracé dans MLflow et que les métriques
qu'il annonce sont internement cohérentes et rechargeables.

Aucun réentraînement, aucune nouvelle logique de scoring : on relit ce qui a
déjà été produit par TP9/TP11.
"""

import pytest
import xgboost as xgb
from conftest import B11_MODEL_ARTIFACT_DIR, B11_RUN_ID

# Relit mlflow_tp7.db, base MLflow locale non versionnée dans Git (binaire,
# change à chaque run) : absente d'un checkout CI brut.
pytestmark = pytest.mark.requires_local_infra

DOCUMENTED_METRICS = {
    "pr_auc_test": 0.8799,
    "roc_auc_test": 0.9949,
    "f1_test": 0.7586,
}

# n_features tel que déclaré dans le model card et les params MLflow du run b11.
DOCUMENTED_N_FEATURES = 69


def test_b11_run_exists_with_expected_lineage(mlflow_client):
    run = mlflow_client.get_run(B11_RUN_ID)
    assert run.data.tags.get("lineage") == "B11-GKF"


def test_b11_logged_metrics_match_model_card(mlflow_client):
    run = mlflow_client.get_run(B11_RUN_ID)
    for key, expected in DOCUMENTED_METRICS.items():
        assert run.data.metrics.get(key) == expected, key


def test_b11_confusion_matrix_is_internally_consistent(mlflow_client):
    """tp/tn/fp/fn loggés doivent reproduire precision/recall/f1 loggés,
    et sommer au nombre de lignes du split test (19 944, cf. test ETL gold)."""
    m = mlflow_client.get_run(B11_RUN_ID).data.metrics
    tp, tn, fp, fn = m["tp"], m["tn"], m["fp"], m["fn"]

    assert tp + tn + fp + fn == 19_944

    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    f1 = 2 * precision * recall / (precision + recall)

    assert round(precision, 4) == m["precision_test"]
    assert round(recall, 4) == m["recall_test"]
    assert round(f1, 4) == m["f1_test"]


def test_b11_model_artifact_is_loadable():
    assert B11_MODEL_ARTIFACT_DIR.exists(), (
        f"Artefact modèle absent à {B11_MODEL_ARTIFACT_DIR} : "
        "le model card référence un runs:/.../xgboost_b11_gkf non matérialisé."
    )
    booster = xgb.Booster()
    booster.load_model(str(B11_MODEL_ARTIFACT_DIR / "model.ubj"))
    assert booster.num_features() > 0


@pytest.mark.xfail(
    reason=(
        "Ecart connu et documente, pas un bug : le booster persiste (67 features) "
        "ne correspond pas au nombre declare dans le model card (69). Garde ce test "
        "actif (xfail, pas skip) pour qu'un XPASS signale si l'ecart est un jour "
        "reconcilie."
    ),
    strict=False,
)
def test_b11_model_feature_count_matches_documentation():
    """Ce test matérialise un écart réel entre le model card (69 features déclarées)
    et le nombre de features réellement encodé dans le booster persisté.
    Un échec ici n'est pas un bug du test : c'est la preuve qu'un notebook
    aujourd'hui (colonnes gold_machine_hourly_feature actuelles, en particulier
    les type_*_count_prev_24h ajoutées après le run b11) ne reproduirait pas le
    modèle documenté sans reconstruire l'exact jeu de colonnes utilisé au moment
    du run fa336bc0880b48bc849a4a6b1e6f412d — jeu de colonnes qui n'est nulle part
    persisté (le booster ne porte pas de feature_names)."""
    booster = xgb.Booster()
    booster.load_model(str(B11_MODEL_ARTIFACT_DIR / "model.ubj"))
    assert booster.num_features() == DOCUMENTED_N_FEATURES
