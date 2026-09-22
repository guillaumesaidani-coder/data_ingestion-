"""Mode fantôme (module 39) : le challenger déjà arbitré (module 37-38)
score en silence à côté du champion, sur une fenêtre d'observation, sans
jamais alerter personne — seul le champion sert. La bascule
(`promote_challenger`) ne s'exécute que si l'observation confirme
l'arbitrage.

Écrit dans un journal séparé (`SHADOW_LOG_CSV`), pas dans `predictions` :
le mode fantôme n'a jamais servi de vraie décision, mélanger ses lignes
à celles réellement vues par un technicien romprait le contrat
`predictions` = ce qui a été réellement servi (feuille de route §1).

Notre dataset s'arrête au 2026-06-08 (dernière fenêtre Gold disponible) :
il n'existe pas de période plus fraîche que celle déjà utilisée pour
l'arbitrage (modules 37-38) pour une vraie fenêtre fantôme. Ce module
rejoue donc la fenêtre d'arbitrage elle-même pour prouver le mécanisme
(scoring silencieux, comparaison, bascule conditionnée) — la décision
stratégique elle-même a déjà été tranchée par l'arbitrage réel."""

import shutil
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

from indusense.arbitration import CUTOFF, DEFAULT_CHALLENGER_PATH, arbitration_matrix
from indusense.config import ML_DIR, get_engine, get_model_path
from indusense.modeling.dataset import TARGET, load_gold_dataset

ACCEPTED_DECISIONS = {"ACCEPTATION_DIRECTE", "ACCEPTATION_SOUS_DEROGATION"}
DEFAULT_ARBITRATION_LOG = ML_DIR / "reports" / "hitl" / "arbitration_log.csv"
SHADOW_LOG_CSV = ML_DIR / "reports" / "hitl" / "shadow_log.csv"


def latest_arbitration(log_csv: Path = DEFAULT_ARBITRATION_LOG) -> dict | None:
    if not log_csv.exists():
        return None
    log = pd.read_csv(log_csv)
    if log.empty:
        return None
    return log.iloc[-1].to_dict()


def run_shadow_window(champion, challenger, X: pd.DataFrame, y_true) -> dict:
    """Score les deux modèles sur la même fenêtre, sans rien publier.
    Retourne la matrice gain/stabilité/régression/angle mort (module 37) —
    même grille de lecture que l'arbitrage, appliquée à l'observation
    fantôme plutôt qu'au premier jugement."""
    champion_pred = (champion.predict_proba(X)[:, 1] >= 0.5).astype(int)
    challenger_pred = (challenger.predict_proba(X)[:, 1] >= 0.5).astype(int)
    return arbitration_matrix(y_true.to_numpy(), champion_pred, challenger_pred)


def shadow_confirms(matrix: dict) -> bool:
    """Zéro tolérance en fantôme : c'est la dernière porte avant la
    production réelle. Contrairement à l'arbitrage (qui admet jusqu'à 1
    régression sous dérogation), une seule régression observée en
    conditions d'exploitation bloque la bascule."""
    return matrix["regressions"] == 0 and matrix["gains"] > 0


def promote_challenger(challenger_path: Path | None = None, model_path: Path | None = None) -> Path:
    """Bascule réelle : sauvegarde le modèle actif (jamais écrasé sans
    filet), puis le remplace par le challenger. Retourne le chemin de la
    sauvegarde -- un rollback est une simple copie en sens inverse.

    Les valeurs par défaut sont résolues à l'appel (pas dans la
    signature) : sinon un test qui monkeypatch DEFAULT_CHALLENGER_PATH
    n'aurait aucun effet -- les valeurs par défaut d'une signature sont
    figées à la définition de la fonction, pas à son appel."""
    challenger_path = Path(challenger_path) if challenger_path else DEFAULT_CHALLENGER_PATH
    model_path = Path(model_path) if model_path else get_model_path()
    backup_path = model_path.with_name(
        f"{model_path.stem}.backup-{datetime.now(UTC):%Y%m%dT%H%M%S}{model_path.suffix}"
    )
    shutil.copy2(model_path, backup_path)
    shutil.copy2(challenger_path, model_path)
    return backup_path


def run_shadow_cycle(
    cutoff: pd.Timestamp = CUTOFF,
    arbitration_log: Path = DEFAULT_ARBITRATION_LOG,
    shadow_log: Path = SHADOW_LOG_CSV,
) -> dict:
    import joblib

    arbitration = latest_arbitration(arbitration_log)
    if arbitration is None:
        return {"status": "AUCUN_ARBITRAGE", "detail": "lancer arbitrate_challenger.py d'abord"}
    if arbitration["decision"] not in ACCEPTED_DECISIONS:
        return {
            "status": "PAS_ELIGIBLE",
            "detail": f"dernière décision d'arbitrage = {arbitration['decision']}, "
            "le mode fantôme n'observe que les challengers acceptés",
        }

    gold = load_gold_dataset(get_engine())
    window = gold.test_df[gold.test_df["window_start"] >= cutoff]
    X, y_true = window[gold.feature_cols], window[TARGET].astype(int)

    champion = joblib.load(get_model_path())
    challenger = joblib.load(DEFAULT_CHALLENGER_PATH)
    matrix = run_shadow_window(champion, challenger, X, y_true)
    confirmed = shadow_confirms(matrix)

    shadow_log.parent.mkdir(parents=True, exist_ok=True)
    row = pd.DataFrame(
        [
            {
                "horodatage": datetime.now(UTC).isoformat(),
                "n_observation": len(window),
                **matrix,
                "confirme": confirmed,
            }
        ]
    )
    row.to_csv(shadow_log, mode="a", header=not shadow_log.exists(), index=False)

    if not confirmed:
        return {"status": "BLOQUE", "matrix": matrix}

    backup_path = promote_challenger()
    return {"status": "PROMU", "matrix": matrix, "backup": str(backup_path)}
