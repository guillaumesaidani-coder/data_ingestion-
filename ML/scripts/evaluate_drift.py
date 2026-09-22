#!/usr/bin/env python
"""Modules 31-34 : rejoue une fenêtre de dérive contre une référence gelée
avec le vrai modèle de production (b11-gkf, `artifacts/models/model.joblib`
— pas un modèle de dérive séparé, voir pipeline_proof.md/M30 sur la reprise).

Règle figée (reports/drift/drift_spec.md) : PSI calculé sur les bins de
la référence, alerte si PSI > 0.25 sur au moins une feature, persistant
si la fenêtre précédente évaluée contre la même référence l'était déjà.
Le KS est calculé pour confirmation humaine, jamais décisionnel seul.

Chaque exécution ajoute une ligne à reports/drift/suivi_fenetres.csv (le
CSV qu'export_drift_metrics.py republie en métriques Prometheus) et écrit
le détail PSI/KS par feature dans reports/drift/psi_f<N>_ref-<ref>.csv.

Usage : uv run --frozen python scripts/evaluate_drift.py --fenetre 2
        uv run --frozen python scripts/evaluate_drift.py --fenetre 4 --reference haute_charge
"""

import argparse
import sys
from datetime import UTC, datetime
from pathlib import Path

import joblib
import pandas as pd
from sklearn.metrics import precision_score, recall_score, roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from indusense.config import get_model_path
from indusense.drift import drift_table
from indusense.modeling.dataset import LEAKAGE_COLS, TARGET

DRIFT_DIR = Path(__file__).resolve().parent.parent / "reports" / "drift"
SUIVI_CSV = DRIFT_DIR / "suivi_fenetres.csv"

DRIFT_FEATURES = [
    "temp_mean_24h",
    "temp_std_24h",
    "pressure_mean_24h",
    "pressure_std_24h",
    "voltage_mean_24h",
    "rotation_mean_24h",
    "pieces_produced_sum_24h",
    "incident_count_prev_24h",
]

WINDOWS = {
    1: ("temoin", "window_f1_temoin.csv"),
    2: ("capteur", "window_f2_capteur.csv"),
    3: ("concept_drift", "window_f3_concept_drift.csv"),
    4: ("campagne", "window_f4_campagne.csv"),
}

PSI_ALERT_THRESHOLD = 0.25
RECALL_FLOOR = 0.70  # ~23% sous le rappel de reference du modele (0.9106, artifacts/models/metrics.json)


def _reference_path(reference: str) -> Path:
    return DRIFT_DIR / f"reference_{reference}.csv"


def _previous_alert(fenetre: int, reference: str) -> bool | None:
    """Dernière évaluation connue de cette fenêtre contre cette référence
    (avant l'exécution courante) — sert à décider la persistance."""
    if not SUIVI_CSV.exists():
        return None
    history = pd.read_csv(SUIVI_CSV)
    match = history[(history["fenetre"] == fenetre) & (history["reference"] == reference)]
    if match.empty:
        return None
    return bool(match.iloc[-1]["alerte_psi"])


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fenetre", type=int, required=True, choices=sorted(WINDOWS))
    parser.add_argument("--reference", choices=["normale", "haute_charge"], default="normale")
    args = parser.parse_args()

    label, filename = WINDOWS[args.fenetre]
    window_df = pd.read_csv(DRIFT_DIR / filename)
    reference_df = pd.read_csv(_reference_path(args.reference))

    table = drift_table(reference_df, window_df, DRIFT_FEATURES)
    max_row = table.loc[table["psi"].idxmax()]
    alerte_psi = bool(max_row["psi"] > PSI_ALERT_THRESHOLD)

    model = joblib.load(get_model_path())
    feature_cols = [c for c in window_df.columns if c not in LEAKAGE_COLS]
    X = window_df[feature_cols]
    y_true = window_df[TARGET]
    y_prob = model.predict_proba(X)[:, 1]
    y_pred = (y_prob >= 0.5).astype(int)
    recall = float(recall_score(y_true, y_pred, zero_division=0))
    precision = float(precision_score(y_true, y_pred, zero_division=0))
    roc_auc = float(roc_auc_score(y_true, y_prob)) if y_true.nunique() > 1 else float("nan")

    persistant = _previous_alert(args.fenetre, args.reference) is True and alerte_psi

    print(f"Fenêtre {args.fenetre} ({label}) vs référence '{args.reference}'")
    print(table.to_string(index=False))
    print(f"PSI max : {max_row['feature']} = {max_row['psi']:.4f} (seuil {PSI_ALERT_THRESHOLD})")
    print(f"Alerte PSI : {'OUI' if alerte_psi else 'non'}" + (" — persistante (2 fenêtres)" if persistant else ""))
    print(
        f"Rappel modèle : {recall:.4f} (plancher {RECALL_FLOOR}) | "
        f"Précision : {precision:.4f} | ROC-AUC : {roc_auc:.4f}"
    )
    if recall < RECALL_FLOOR:
        print("KPI second rideau : rappel sous le plancher -> réentraînement à envisager (protocole m21)")

    DRIFT_DIR.mkdir(parents=True, exist_ok=True)
    row = pd.DataFrame(
        [
            {
                "horodatage": datetime.now(UTC).isoformat(),
                "fenetre": args.fenetre,
                "label": label,
                "reference": args.reference,
                "psi_max_feature": max_row["feature"],
                "psi_max": round(float(max_row["psi"]), 4),
                "ks_pvalue_min": round(float(table["ks_pvalue"].min()), 6),
                "alerte_psi": alerte_psi,
                "persistant": persistant,
                "recall": round(recall, 4),
                "precision": round(precision, 4),
                "roc_auc": round(roc_auc, 4),
                "sous_plancher_rappel": recall < RECALL_FLOOR,
            }
        ]
    )
    row.to_csv(SUIVI_CSV, mode="a", header=not SUIVI_CSV.exists(), index=False)
    table.assign(fenetre=args.fenetre, reference=args.reference).to_csv(
        DRIFT_DIR / f"psi_f{args.fenetre}_ref-{args.reference}.csv", index=False
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
