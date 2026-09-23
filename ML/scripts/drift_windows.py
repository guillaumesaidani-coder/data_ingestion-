#!/usr/bin/env python
"""Modules 31-34 (dérive) : fige deux références et construit 4 fenêtres
d'évaluation à partir du vrai Gold (`gold_machine_hourly_feature`, 15
machines, juin 2025 -> juin 2026) — pas du jeu de données jouet du guide.

Deux mois réels servent de référence :
- "normale"      : 2026-03 (température moyenne la plus basse observée,
  profil calme)
- "haute_charge" : 2025-09 (température moyenne la plus haute observée,
  à égalité avec octobre) — notre jeu de données synthétique n'a pas de
  vrai changement de régime saisonnier marqué comme celui du guide ;
  c'est le choix le plus honnête disponible dans nos vraies données.

Quatre fenêtres, dont deux avec une perturbation synthétique clairement
documentée (jamais présentée comme un fait mesuré) :
- F1 témoin        : 2026-04, réel, sans modification.
- F2 capteur menteur : 2026-04, copie avec temp_mean_* décalé de +8°C
  (biais constant = offset de calibration, l'écart-type n'est pas touché).
- F3 concept drift : 2026-05, réel, `label_failure_next_24h` mélangé
  (permutation aléatoire, seed fixe) — simule une relation X->y qui a
  changé sans toucher aux features.
- F4 campagne planifiée : 2025-10, réel, même saison chaude que la
  référence haute_charge mais mois différent (contre-épreuve non triviale).

Usage : uv run --frozen python scripts/drift_windows.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from indusense.config import get_engine
from indusense.modeling.dataset import LEAKAGE_COLS, TARGET

OUT_DIR = Path(__file__).resolve().parent.parent / "reports" / "drift"

# Sous-ensemble surveillé pour PSI/KS — représentatif de chaque capteur,
# pas les ~70 colonnes du Gold (bruit, pas de valeur ajoutée en plus).
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

REFERENCE_NORMALE_MONTH = "2026-03"
REFERENCE_HAUTE_CHARGE_MONTH = "2025-09"
F1_MONTH = "2026-04"
F3_MONTH = "2026-05"
F4_MONTH = "2025-10"

SEED = 42


def _load_month(df: pd.DataFrame, month: str) -> pd.DataFrame:
    window = df[df["window_start"].dt.strftime("%Y-%m") == month].copy()
    if window.empty:
        raise ValueError(f"Aucune ligne Gold pour {month}")
    return window


def _inject_sensor_bias(window: pd.DataFrame, delta_celsius: float = 8.0) -> pd.DataFrame:
    """F2 — biais constant de calibration : décale les moyennes de
    température, laisse l'écart-type intact (un offset ne change pas la
    dispersion), laisse les autres capteurs inchangés."""
    perturbed = window.copy()
    for col in perturbed.columns:
        if col.startswith("temp_mean_"):
            perturbed[col] = perturbed[col] + delta_celsius
    return perturbed


def _shuffle_labels(window: pd.DataFrame, seed: int = SEED) -> pd.DataFrame:
    """F3 — permute le label observé sans toucher aux features : simule
    une relation X->y qui a changé (concept drift), invisible au PSI qui
    ne regarde que la distribution des features."""
    perturbed = window.copy()
    rng = np.random.default_rng(seed)
    perturbed[TARGET] = rng.permutation(perturbed[TARGET].to_numpy())
    return perturbed


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    engine = get_engine()
    df = pd.read_sql(
        "SELECT * FROM gold_machine_hourly_feature ORDER BY machine_id, window_start", engine
    )
    df["window_start"] = pd.to_datetime(df["window_start"])
    feature_cols = [c for c in df.columns if c not in LEAKAGE_COLS]
    keep_cols = ["machine_id", "window_start", TARGET, *feature_cols]

    reference_normale = _load_month(df, REFERENCE_NORMALE_MONTH)[keep_cols]
    reference_haute_charge = _load_month(df, REFERENCE_HAUTE_CHARGE_MONTH)[keep_cols]
    f1 = _load_month(df, F1_MONTH)[keep_cols]
    f2 = _inject_sensor_bias(f1)
    f3_base = _load_month(df, F3_MONTH)[keep_cols]
    f3 = _shuffle_labels(f3_base)
    f4 = _load_month(df, F4_MONTH)[keep_cols]

    reference_normale.to_csv(OUT_DIR / "reference_normale.csv", index=False)
    reference_haute_charge.to_csv(OUT_DIR / "reference_haute_charge.csv", index=False)
    f1.to_csv(OUT_DIR / "window_f1_temoin.csv", index=False)
    f2.to_csv(OUT_DIR / "window_f2_capteur.csv", index=False)
    f3.to_csv(OUT_DIR / "window_f3_concept_drift.csv", index=False)
    f4.to_csv(OUT_DIR / "window_f4_campagne.csv", index=False)

    print(f"reference_normale       ({REFERENCE_NORMALE_MONTH})      : {len(reference_normale)} lignes")
    print(f"reference_haute_charge  ({REFERENCE_HAUTE_CHARGE_MONTH}) : {len(reference_haute_charge)} lignes")
    print(f"F1 témoin               ({F1_MONTH})      : {len(f1)} lignes")
    print(f"F2 capteur menteur      (F1 + 8°C synth.) : {len(f2)} lignes")
    print(f"F3 concept drift        ({F3_MONTH}, labels mélangés synth.) : {len(f3)} lignes")
    print(f"F4 campagne planifiée   ({F4_MONTH})      : {len(f4)} lignes")
    print(f"Écrit dans {OUT_DIR}")


if __name__ == "__main__":
    main()
