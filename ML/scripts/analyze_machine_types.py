#!/usr/bin/env python
"""Les 15 presses ne sont pas identiques : 4 modèles (InduPress-X1, X2,
X3, Z1, colonne `machine.model`), des taux de panne qui varient d'un
facteur 3, des niveaux de tension très différents. Le modèle b11-gkf ne
reçoit pourtant ni `machine_id` ni le type. Ce script mesure ce que ça
lui coûte :

1. profil par type (taux de panne, niveaux capteurs, incidents) ;
2. part de variance de chaque feature expliquée par le type (eta²),
   rapprochée de l'importance de la feature dans le modèle de prod ;
3. PR-AUC du modèle de prod sur le test, par type ;
4. validation croisée sur train+validation, hyperparamètres b11 :
   - « machine cachée » : une machine par fold (son type reste vu) ;
   - « type caché »     : un type par fold (cas d'un nouveau modèle de presse) ;
   - « machine cachée + type + âge » : les mêmes folds, en ajoutant le
     type (one-hot) et l'âge de la machine aux features, ensemble puis
     séparément (« + âge » seul, « + type » seul) pour isoler chacun.

Environ 65 entraînements, 6 à 7 minutes sur CPU.

Usage :
    uv run --frozen python scripts/analyze_machine_types.py
    # sans Postgres : Gold exporté par DVC + référentiel du seed SQL
    uv run --frozen python scripts/analyze_machine_types.py \\
        --gold-csv data/gold/gold_dataset.csv --machine-sql machine.sql
"""

import argparse
import json
import re
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

ML_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ML_DIR / "src"))

from indusense.config import get_engine
from indusense.modeling.dataset import LEAKAGE_COLS, TARGET
from indusense.modeling.heterogeneity import (
    leave_one_group_out_proba,
    load_machine_reference,
    pr_auc_by,
)
from indusense.modeling.train import B11_PARAMS
from indusense.scoring import model_feature_cols

# ('MACH-01', '2021-05-12', 770, 48, 'InduPress-X2', ...) dans machine.sql
SEED_ROW = re.compile(r"\('(MACH-\d+)',\s*'([\d-]+)',\s*\d+,\s*\d+,\s*'([^']+)'")


def machines_from_seed(path: Path) -> pd.DataFrame:
    rows = SEED_ROW.findall(path.read_text(encoding="utf-8"))
    df = pd.DataFrame(rows, columns=["machine_id", "commissioning_date", "machine_type"])
    df["commissioning_date"] = pd.to_datetime(df["commissioning_date"])
    return df


def load_inputs(gold_csv: Path | None, machine_sql: Path | None):
    if gold_csv:
        gold = pd.read_csv(gold_csv, parse_dates=["window_start"])
    else:
        gold = pd.read_sql("SELECT * FROM gold_machine_hourly_feature", get_engine(),
                           parse_dates=["window_start"])
    machines = machines_from_seed(machine_sql) if machine_sql else load_machine_reference(get_engine())
    df = gold.merge(machines, on="machine_id", how="left")
    if df["machine_type"].isna().any():
        missing = sorted(df.loc[df["machine_type"].isna(), "machine_id"].unique())
        raise SystemExit(f"Machines absentes du référentiel : {missing}")
    return df


def eta2(df: pd.DataFrame, col: str) -> float:
    """Part de la variance de `col` expliquée par le type de presse."""
    x = df[[col, "machine_type"]].dropna()
    total = ((x[col] - x[col].mean()) ** 2).sum()
    if total == 0:
        return np.nan
    g = x.groupby("machine_type")[col]
    return float(((g.mean() - x[col].mean()) ** 2 * g.size()).sum() / total)


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--gold-csv", type=Path, help="Gold exporté (sinon Postgres)")
    parser.add_argument("--machine-sql", type=Path, help="seed machine.sql (sinon table machine)")
    parser.add_argument("--model", type=Path, default=ML_DIR / "artifacts/models/model.joblib")
    parser.add_argument("--out", type=Path, default=ML_DIR / "artifacts/machine_types_analysis.json")
    args = parser.parse_args()

    df = load_inputs(args.gold_csv, args.machine_sql)
    prod = joblib.load(args.model)
    feats = model_feature_cols(prod, [c for c in df.columns if c not in LEAKAGE_COLS])
    report = {}

    # 1. Profil par type
    profile = df.groupby("machine_type").agg(
        machines=("machine_id", "nunique"), lignes=("machine_id", "size"),
        taux_panne=(TARGET, "mean"), temp_c=("temp_mean_1h", "mean"),
        pression_bar=("pressure_mean_1h", "mean"), rotation=("rotation_mean_1h", "mean"),
        tension_v=("voltage_mean_1h", "mean"), incidents_24h=("incident_count_prev_24h", "mean"),
    )
    print("== 1. Profil par type ==\n", profile.round(3).to_string(), "\n")
    report["profil"] = profile.round(4).reset_index().to_dict("records")

    # 2. eta² vs importance dans le modèle de prod
    e = pd.Series({c: eta2(df, c) for c in feats})
    imp = pd.Series(prod.named_steps["model"].feature_importances_,
                    index=prod.named_steps["imputer"].get_feature_names_out())
    imp = imp / imp.sum()
    top = pd.DataFrame({"importance": imp, "eta2_type": e}).sort_values("importance", ascending=False)
    print("== 2. Features les plus importantes et dépendance au type ==\n",
          top.head(8).round(3).to_string())
    print(f"features dont le type explique > 10 % de la variance : {int((e > 0.10).sum())}/{len(e)} "
          f"— part de l'importance du modèle sur ces features : {imp[e[e > 0.10].index].sum():.1%}\n")
    report["top_features"] = top.head(10).round(4).reset_index(names="feature").to_dict("records")
    report["eta2_top"] = e.sort_values(ascending=False).head(10).round(4).to_dict()

    # 3. Modèle de prod sur le test, par type
    test = df[df["split_set"] == "test"]
    test_proba = pd.Series(prod.predict_proba(test.reindex(columns=feats))[:, 1], index=test.index)
    by_type_test = pr_auc_by(test, TARGET, test_proba, "machine_type")
    print("== 3. PR-AUC test du modèle de prod par type ==\n", by_type_test.round(3).to_string(),
          f"\n global : {average_precision_score(test[TARGET], test_proba):.3f}\n")
    report["test_pr_auc_by_type"] = by_type_test.round(4).to_dict()

    # 4. Validations croisées
    tv = df[df["split_set"].isin(["train", "validation"])].copy()
    window_start = tv["window_start"]
    if window_start.dt.tz is not None:
        window_start = window_start.dt.tz_localize(None)
    tv["machine_age_days"] = (window_start - tv["commissioning_date"]).dt.days
    type_cols = pd.get_dummies(tv["machine_type"], prefix="type", dtype=float)
    tv = pd.concat([tv, type_cols], axis=1)
    feats_plus = feats + ["machine_age_days", *type_cols.columns]

    variants = {
        "machine_cachee": (feats, "machine_id"),
        "type_cache": (feats, "machine_type"),
        "machine_cachee_age": (feats + ["machine_age_days"], "machine_id"),
        "machine_cachee_type": (feats + list(type_cols.columns), "machine_id"),
        "machine_cachee_type_age": (feats_plus, "machine_id"),
    }
    cv = pd.DataFrame({"machine_type": tv.groupby("machine_id")["machine_type"].first()})
    for name, (cols, group) in variants.items():
        print(f"  validation croisée « {name} »...", flush=True)
        proba = leave_one_group_out_proba(tv, cols, TARGET, group, B11_PARAMS)
        cv[name] = pr_auc_by(tv, TARGET, proba, "machine_id")
    scores = cv[list(variants)]
    print("\n== 4. PR-AUC par machine ==\n", cv.round(3).to_string())
    print("\n moyenne par type :\n", scores.groupby(cv["machine_type"]).mean().round(3).to_string())
    print("\n moyenne 15 machines :", scores.mean().round(3).to_dict())
    print(" écart-type          :", scores.std().round(3).to_dict())
    report["cv_by_machine"] = cv.round(4).reset_index().to_dict("records")
    report["cv_mean"] = scores.mean().round(4).to_dict()
    report["cv_std"] = scores.std().round(4).to_dict()

    args.out.write_text(json.dumps(report, indent=1, ensure_ascii=False, default=str), encoding="utf-8")
    print(f"\nRapport : {args.out}")


if __name__ == "__main__":
    main()
