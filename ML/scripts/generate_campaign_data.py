#!/usr/bin/env python
"""Campagne 2 : trois mois de lots terrain SIMULÉS (9 juin → 7 septembre
2026, 13 lots hebdomadaires) où un nouveau mode de panne apparaît.
Pédagogique : faire évoluer le champion pour une bonne raison — le monde
a changé — sans assouplir la règle d'or de l'arbitrage.

Deux mécanismes, par machine :

1. Le connu continue : la machine rejoue un bloc contigu de 13 semaines
   de son propre historique (telemetry.csv, releves_incidents.csv,
   maintenance de machine.sql), décalé dans le temps, capteurs bruités.
   Bloc contigu : les chaînes « incident de sévérité 3 puis panne 24 h
   plus tard » restent intactes, comme les défauts de qualité d'origine
   (doublons horaires, valeurs manquantes).
2. Le nouveau : des fuites hydrauliques. La pression baisse pendant
   12 à 18 h, puis panne (sévérité 4 ou 5, type baisse de pression),
   SANS incident de sévérité 3 avant : le champion, qui s'appuie sur les
   incidents déclarés, ne les voit pas venir. Apparition progressive
   (rare en juin, installée à partir d'août), plus fréquente sur les
   presses les plus anciennes.

Sorties dans data/campagne_2/ : lot_NN_<date>_{telemetry,incidents,
maintenance}.csv, au format des fichiers terrain d'origine (prêts pour
`indusense.ingest.ingest_terrain_batch`), et verite_terrain_fuites.csv
— la liste des fuites injectées, pour l'enseignant, jamais lue par le
pipeline. Déterministe (graine fixe) ; chaque lot est vérifié avec la
validation Bronze avant d'être écrit.

Source des incidents : releves_incidents.csv (le seul dont les
sévérités et les machines correspondent au Gold ; le CSV anonymisé est
perturbé), noms remplacés par leur pseudonyme `OP_ANON_…`. Ce fichier
n'existe que sur le poste d'origine : les lots générés sont versionnés
par DVC (data/campagne_2.dvc), c'est ainsi qu'on les récupère ailleurs.

Usage :
    uv run --frozen python scripts/generate_campaign_data.py
    uv run --frozen python scripts/generate_campaign_data.py --intensite 1.5
"""

import argparse
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ML_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ML_DIR / "src"))

from indusense.processing.anonymization import anon
from indusense.processing.bronze_validation import validate_bronze
from indusense.processing.gold_features import TYPE_COLS

SEED = 20260609
PERIOD_START = pd.Timestamp("2026-06-09 00:00:00")
N_WEEKS = 13
BLOCK = pd.Timedelta(weeks=N_WEEKS)
# Début du bloc source tiré dans cet intervalle : le bloc de 13 semaines
# reste dans l'historique (1er juin 2025 → 8 juin 2026).
SOURCE_START_MIN = pd.Timestamp("2025-06-02")
SOURCE_START_MAX = pd.Timestamp("2026-03-09")
NOISE_FACTOR = 0.25  # bruit ajouté = 25 % de la variation horaire typique du capteur

# Fuites hydrauliques
LEAK_WEEKLY_RATE = 0.30  # probabilité par machine et par semaine, une fois installée
LEAK_RAMP_WEEKS = 8  # montée progressive : semaine 1 → 1/8 du taux, semaine 8 → taux plein
OLD_MACHINE_BEFORE = pd.Timestamp("2022-01-01")  # presses anciennes : fuites 1,5 × plus fréquentes
LEAK_DURATION_H = (12, 18)
LEAK_DROP_BAR = (6.0, 12.0)  # × --intensite
QUIET_BEFORE_H, QUIET_AFTER_H = 48, 24  # aucune autre panne ni sévérité ≥ 3 autour d'une fuite
LEAK_COMMENT = "fuite circuit hydraulique"
LEAK_COMPONENT = "circuit hydraulique"
LEAK_REPAIR = "Remplacement flexible + purge circuit hydraulique"

OUT_DIR = ML_DIR / "data" / "campagne_2"
INCIDENT_COLS = ["incident_id", "date", "time", "operator_name", "machine_id", "severity",
                 "operator_badge", "comment", "shift", *TYPE_COLS]
MAINT_COLS = ["maintenance_id", "machine_id", "maintenance_at", "maintenance_type",
              "action_type", "component", "description", "related_incident_id", "duration_hours"]
SIGNALS = ["temperature_c", "pressure_bar", "voltage_mean_v", "rotation_mean_rpm"]
MAINT_ROW = re.compile(
    r"\((\d+), '(MACH-\d+)', '([^']+)', '(\w+)', '(\w+)', '([^']*)', '([^']*)', "
    r"(NULL|'[^']*'), ([\d.]+)\)"
)


def load_history():
    tel = pd.read_csv(ML_DIR / "telemetry.csv", parse_dates=["timestamp"])
    inc = pd.read_csv(ML_DIR / "releves_incidents.csv")
    inc["ts"] = pd.to_datetime(inc["date"] + " " + inc["time"])
    # Les lots sont versionnés (DVC) : aucun vrai nom n'en sort, seulement
    # le pseudonyme du CSV anonymisé du projet (même fonction, même sel).
    inc["operator_name"] = inc["operator_name"].map(anon)
    rows = MAINT_ROW.findall((ML_DIR / "machine.sql").read_text(encoding="utf-8"))
    maint = pd.DataFrame(rows, columns=MAINT_COLS)
    maint["ts"] = pd.to_datetime(maint["maintenance_at"].str.replace("+00", "", regex=False))
    maint["related_incident_id"] = maint["related_incident_id"].replace("NULL", None).str.strip("'")
    commissioning = dict(re.findall(r"\('(MACH-\d+)', '(\d{4}-\d{2}-\d{2})', \d+, \d+, '", (ML_DIR / "machine.sql").read_text(encoding="utf-8")))
    return tel, inc, maint, {m: pd.Timestamp(d) for m, d in commissioning.items()}


def shift_of(hour: int) -> str:
    return "matin" if 6 <= hour < 14 else "apres-midi" if 14 <= hour < 22 else "nuit"


def replay_known(rng, tel, inc, maint):
    """Mécanisme 1 : chaque machine rejoue un bloc contigu de son historique."""
    noise_std = tel.sort_values("timestamp").groupby("machine_id")[SIGNALS].agg(
        lambda s: s.diff().std() * NOISE_FACTOR)
    days = (SOURCE_START_MAX - SOURCE_START_MIN).days
    out_tel, out_inc, out_maint, sources = [], [], [], {}
    for m in sorted(tel["machine_id"].unique()):
        src = SOURCE_START_MIN + pd.Timedelta(days=int(rng.integers(0, days + 1)))
        delta = PERIOD_START - src
        sources[m] = src
        t = tel[(tel.machine_id == m) & (tel.timestamp >= src) & (tel.timestamp < src + BLOCK)].copy()
        t["timestamp"] = t["timestamp"] + delta
        for c in SIGNALS:
            t[c] = (t[c] + rng.normal(0, noise_std.loc[m, c], len(t))).round(3)
        out_tel.append(t)
        i = inc[(inc.machine_id == m) & (inc.ts >= src) & (inc.ts < src + BLOCK)].copy()
        i["ts"] = i["ts"] + delta
        out_inc.append(i)
        k = maint[(maint.machine_id == m) & (maint.ts >= src) & (maint.ts < src + BLOCK)].copy()
        k["ts"] = k["ts"] + delta
        out_maint.append(k)
    return pd.concat(out_tel), pd.concat(out_inc), pd.concat(out_maint), sources


def inject_leaks(rng, tel, inc, maint, commissioning, intensite):
    """Mécanisme 2 : fuites hydrauliques sans précurseur déclaré."""
    roster = inc[["operator_name", "operator_badge"]].drop_duplicates().to_dict("records")
    leaks, new_inc, new_maint = [], [], []
    for m in sorted(tel["machine_id"].unique()):
        weight = 1.5 if commissioning[m] < OLD_MACHINE_BEFORE else 1.0
        for w in range(N_WEEKS):
            p = LEAK_WEEKLY_RATE * weight * min(1.0, (w + 1) / LEAK_RAMP_WEEKS)
            if rng.random() >= p:
                continue
            week_start = PERIOD_START + pd.Timedelta(weeks=w)
            for _ in range(50):  # un instant « calme » sur cette machine, cette semaine
                T = week_start + pd.Timedelta(hours=int(rng.integers(0, 168)),
                                              minutes=int(rng.integers(0, 60)))
                if T < PERIOD_START + pd.Timedelta(hours=QUIET_BEFORE_H):
                    continue
                near = inc[(inc.machine_id == m) & (inc.severity >= 3)
                           & (inc.ts > T - pd.Timedelta(hours=QUIET_BEFORE_H))
                           & (inc.ts < T + pd.Timedelta(hours=QUIET_AFTER_H))]
                near_leak = [x for x in leaks if x["machine_id"] == m
                             and abs((x["panne_at"] - T).total_seconds()) < 72 * 3600]
                if near.empty and not near_leak:
                    break
            else:
                continue
            duration = int(rng.integers(LEAK_DURATION_H[0], LEAK_DURATION_H[1] + 1))
            drop = float(rng.uniform(*LEAK_DROP_BAR)) * intensite
            hours = (T - tel["timestamp"]).dt.total_seconds() / 3600
            sel = (tel.machine_id == m) & (hours >= 0) & (hours <= duration)
            progress = 1 - hours[sel] / duration  # 0 au début de la fuite, 1 à la panne
            tel.loc[sel, "pressure_bar"] = (tel.loc[sel, "pressure_bar"] - drop * progress**1.5).round(3)
            op = roster[int(rng.integers(0, len(roster)))]
            severity = 5 if rng.random() < 0.2 else 4
            new_inc.append({"ts": T, "machine_id": m, "severity": severity, "comment": LEAK_COMMENT,
                            "shift": shift_of(T.hour), **op,
                            **{c: int(c == "type_baisse_pression") for c in TYPE_COLS}})
            repair_at = T + pd.Timedelta(minutes=int(rng.integers(30, 180)))
            new_maint.append({"ts": repair_at, "machine_id": m, "maintenance_type": "reactive",
                              "action_type": "intervention_corrective", "component": LEAK_COMPONENT,
                              "description": LEAK_REPAIR, "leak_ts": T,
                              "duration_hours": round(float(rng.uniform(2.0, 4.0)), 2)})
            leaks.append({"machine_id": m, "debut_fuite": T - pd.Timedelta(hours=duration),
                          "panne_at": T, "duree_h": duration, "baisse_bar": round(drop, 2),
                          "severite": severity, "semaine": w + 1})
    return pd.DataFrame(leaks), pd.DataFrame(new_inc), pd.DataFrame(new_maint)


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--intensite", type=float, default=1.0,
                        help="multiplie la baisse de pression des fuites (1.0 = 6 à 12 bar)")
    parser.add_argument("--out", type=Path, default=OUT_DIR)
    args = parser.parse_args()
    rng = np.random.default_rng(SEED)

    tel_h, inc_h, maint_h, commissioning = load_history()
    tel, inc, maint, sources = replay_known(rng, tel_h, inc_h, maint_h)
    leaks, leak_inc, leak_maint = inject_leaks(rng, tel, inc, maint, commissioning, args.intensite)

    # Incidents : numérotation chronologique à la suite de l'historique
    inc = pd.concat([inc.assign(leak_ts=pd.NaT, old_id=inc["incident_id"]),
                     leak_inc.assign(leak_ts=leak_inc["ts"], old_id=None)], ignore_index=True)
    inc = inc.sort_values(["ts", "machine_id"]).reset_index(drop=True)
    first = int(inc_h["incident_id"].str[4:].astype(int).max()) + 1
    inc["incident_id"] = [f"INC-{n:06d}" for n in range(first, first + len(inc))]
    inc["date"], inc["time"] = inc["ts"].dt.strftime("%Y-%m-%d"), inc["ts"].dt.strftime("%H:%M")
    # Les copies gardent le lien à « leur » incident (même bloc) ; une fuite, au sien.
    by_old = dict(zip(inc["old_id"].dropna(), inc.loc[inc["old_id"].notna(), "incident_id"]))
    by_leak = dict(zip(zip(inc.loc[inc.leak_ts.notna(), "machine_id"], inc.loc[inc.leak_ts.notna(), "leak_ts"]),
                       inc.loc[inc.leak_ts.notna(), "incident_id"]))
    maint = pd.concat([maint.assign(related_incident_id=maint["related_incident_id"].map(by_old)),
                       leak_maint.assign(related_incident_id=[by_leak[(m, t)] for m, t in
                                                              zip(leak_maint.machine_id, leak_maint.leak_ts)])],
                      ignore_index=True).sort_values(["ts", "machine_id"]).reset_index(drop=True)
    first_m = int(maint_h["maintenance_id"].astype(int).max()) + 1
    maint["maintenance_id"] = range(first_m, first_m + len(maint))
    maint["maintenance_at"] = maint["ts"].dt.strftime("%Y-%m-%d %H:%M:%S+00")
    tel = tel.sort_values(["timestamp", "machine_id"], kind="stable")
    tel["timestamp"] = tel["timestamp"].dt.strftime("%Y-%m-%d %H:%M:%S")

    args.out.mkdir(parents=True, exist_ok=True)
    report = []
    for w in range(N_WEEKS):
        a = PERIOD_START + pd.Timedelta(weeks=w)
        b = a + pd.Timedelta(weeks=1)
        stem = f"lot_{w + 1:02d}_{a:%Y-%m-%d}"
        ts_tel = pd.to_datetime(tel["timestamp"])
        parts = {
            "telemetry": tel[(ts_tel >= a) & (ts_tel < b)],
            "incidents": inc[(inc.ts >= a) & (inc.ts < b)][INCIDENT_COLS],
            "maintenance": maint[(maint.ts >= a) & (maint.ts < b)][MAINT_COLS],
        }
        row = {"lot": stem}
        for source, df in parts.items():
            path = args.out / f"{stem}_{source}.csv"
            df.to_csv(path, index=False)
            checked = validate_bronze(pd.read_csv(path, dtype=str), source)
            row[source] = len(df)
            row[f"{source}_rejets"] = int((~checked["parse_ok"]).sum())
        row["fuites"] = int(((leaks.panne_at >= a) & (leaks.panne_at < b)).sum()) if len(leaks) else 0
        report.append(row)
    leaks.to_csv(args.out / "verite_terrain_fuites.csv", index=False)

    report = pd.DataFrame(report).set_index("lot")
    print(report.to_string())
    failures = inc[inc.severity >= 4]
    print(f"\n{len(leaks)} fuites injectées sur {leaks.machine_id.nunique() if len(leaks) else 0} machines "
          f"(intensité {args.intensite}) ; pannes (sévérité ≥ 4) : {len(failures)}, dont "
          f"{int(failures.comment.eq(LEAK_COMMENT).sum())} fuites")
    print("blocs source :", {m: f"{s:%Y-%m-%d}" for m, s in sources.items()})
    print(f"écrit dans {args.out}")


if __name__ == "__main__":
    main()
