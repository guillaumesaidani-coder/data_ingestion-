"""Silver -> Gold : feature engineering (extrait de TP6.ipynb).

Chaque fonction correspond à une section du notebook (mêmes noms de
colonnes, même ordre de calcul). `build_gold_features()` les enchaîne
dans l'ordre exact du notebook. L'écriture en base (TRUNCATE + to_sql,
gestion d'ingestion_batch) reste dans TP6.ipynb : ce module ne fait que
la transformation, pas l'I/O.
"""
import numpy as np
import pandas as pd

SIGNALS = {
    "temp":     "temperature_c",
    "pressure": "pressure_bar",
    "voltage":  "voltage_mean_v",
    "rotation": "rotation_mean_rpm",
}

PREFIX_MAP = {
    "temperature_c":     "temp",
    "pressure_bar":      "pressure",
    "voltage_mean_v":    "voltage",
    "rotation_mean_rpm": "rotation",
}

TYPE_COLS = [
    "type_surchauffe", "type_baisse_pression", "type_vibration", "type_bruit_mecanique",
    "type_surconsommation", "type_blocage_mecanique", "type_alarme_capteur",
    "type_arret_urgence", "type_defaut_qualite",
]

HORIZONS = [
    ("label_failure_next_6h",  "future_incident_count_6h",  6),
    ("label_failure_next_12h", "future_incident_count_12h", 12),
    ("label_failure_next_24h", "future_incident_count_24h", 24),
    ("label_failure_next_48h", "future_incident_count_48h", 48),
]


def pivot_to_wide(df_sensors: pd.DataFrame) -> pd.DataFrame:
    """1 ligne / capteur / heure -> 1 ligne / machine / heure."""
    df_wide = df_sensors.pivot_table(
        index=["machine_id", "observed_at"],
        columns="sensor_type", values="sensor_value", aggfunc="mean",
    ).reset_index()
    df_wide.columns.name = None
    for col in ["temperature_c", "pressure_bar", "voltage_mean_v", "rotation_mean_rpm", "pieces_produced"]:
        if col not in df_wide.columns:
            df_wide[col] = float("nan")
    return df_wide.sort_values(["machine_id", "observed_at"]).reset_index(drop=True)


def compute_rolling_features(df_wide: pd.DataFrame) -> pd.DataFrame:
    """Agrégats 1h bruts, rolling 6h/12h/24h (mean/max/std, closed='left'
    donc jamais la valeur courante), tendances delta_1h/delta_3h/trend_6h."""
    chunks = []
    for _, grp in df_wide.groupby("machine_id"):
        grp = grp.sort_values("observed_at").copy()
        idx = grp.set_index("observed_at")

        grp["temp_mean_1h"]           = grp["temperature_c"]
        grp["temp_max_1h"]            = grp["temperature_c"]
        grp["temp_std_1h"]            = float("nan")
        grp["pressure_mean_1h"]       = grp["pressure_bar"]
        grp["pressure_max_1h"]        = grp["pressure_bar"]
        grp["pressure_std_1h"]        = float("nan")
        grp["voltage_mean_1h"]        = grp["voltage_mean_v"]
        grp["rotation_mean_1h"]       = grp["rotation_mean_rpm"]
        grp["pieces_produced_sum_1h"] = pd.to_numeric(grp["pieces_produced"], errors="coerce")

        for w in ["6h", "12h", "24h"]:
            roll = idx[list(SIGNALS.values())].rolling(window=w, closed="left", min_periods=1)
            for col, pfx in PREFIX_MAP.items():
                grp[f"{pfx}_mean_{w}"] = roll[col].mean().values
                grp[f"{pfx}_std_{w}"]  = roll[col].std().values
                grp[f"{pfx}_max_{w}"]  = roll[col].max().values

        grp["pieces_produced_sum_24h"] = (
            idx["pieces_produced"].rolling(window="24h", closed="left", min_periods=1).sum().values
        )

        for col, pfx in PREFIX_MAP.items():
            s = grp[col]
            grp[f"{pfx}_delta_1h"] = s.diff(1)
            grp[f"{pfx}_delta_3h"] = s.diff(3)
            grp[f"{pfx}_trend_6h"] = s.diff(6)

        chunks.append(grp)

    return pd.concat(chunks, ignore_index=True)


def compute_capacity_utilization(df: pd.DataFrame, df_machine: pd.DataFrame) -> pd.DataFrame:
    """`capacity_utilization_pct` : jointure statique, aucune fuite (pas de dépendance temporelle)."""
    df = df.merge(df_machine, on="machine_id", how="left")
    df["capacity_utilization_pct"] = (
        df["pieces_produced_sum_1h"] / df["max_hourly_capacity_pieces"] * 100
    )
    return df.drop(columns=["max_hourly_capacity_pieces"])


def compute_rolling_zscore(df: pd.DataFrame) -> pd.DataFrame:
    """Z-score glissant 24h : (valeur courante - mean_24h) / std_24h."""
    df = df.copy()
    df["temp_zscore_24h"]     = (df["temperature_c"] - df["temp_mean_24h"]) / df["temp_std_24h"].replace(0, float("nan"))
    df["pressure_zscore_24h"] = (df["pressure_bar"]   - df["pressure_mean_24h"]) / df["pressure_std_24h"].replace(0, float("nan"))
    return df


def assign_time_split(df: pd.DataFrame) -> pd.DataFrame:
    """Split chronologique 70/15/15 sur les timestamps distincts (pas sur les lignes)."""
    df = df.copy()
    dates  = df["observed_at"].sort_values().unique()
    n      = len(dates)
    cut_70 = dates[int(n * 0.70)]
    cut_85 = dates[int(n * 0.85)]

    df["split_set"] = np.where(
        df["observed_at"] < cut_70, "train",
        np.where(df["observed_at"] < cut_85, "validation", "test"),
    )
    return df


def compute_machine_zscore(df: pd.DataFrame) -> pd.DataFrame:
    """Z-score machine : baseline (mean/std) calculée sur le train set uniquement,
    appliquée telle quelle à train/val/test (anti-leakage)."""
    train_stats = (
        df[df["split_set"] == "train"]
        .groupby("machine_id")[["temperature_c", "pressure_bar"]]
        .agg(["mean", "std"])
    )
    train_stats.columns = ["temp_train_mean", "temp_train_std", "pressure_train_mean", "pressure_train_std"]

    df = df.merge(train_stats.reset_index(), on="machine_id", how="left")
    df["temp_zscore_machine"]     = (df["temperature_c"] - df["temp_train_mean"]) / df["temp_train_std"].replace(0, float("nan"))
    df["pressure_zscore_machine"] = (df["pressure_bar"]  - df["pressure_train_mean"]) / df["pressure_train_std"].replace(0, float("nan"))
    return df.drop(columns=["temp_train_mean", "temp_train_std", "pressure_train_mean", "pressure_train_std"])


def compute_incident_lookback_features(df: pd.DataFrame, df_inc: pd.DataFrame) -> pd.DataFrame:
    """Comptage incidents 24h/7j, sévérité max 24h, heures depuis dernier incident,
    comptage par type sur 24h — intra-machine, strictement passé (< h)."""
    df = df.copy()
    inc_cols = ["incident_count_prev_24h", "incident_max_severity_prev_24h",
                "incident_count_prev_7d", "hours_since_last_incident"] + \
               [f"{t}_count_prev_24h" for t in TYPE_COLS]
    for col in inc_cols:
        df[col] = 0
    df["hours_since_last_incident"] = float("nan")

    for machine in df["machine_id"].unique():
        mask_f = df["machine_id"] == machine
        hours  = df.loc[mask_f, "observed_at"]
        evts   = df_inc[df_inc["machine_id"] == machine].sort_values("occurred_at")
        if evts.empty:
            continue
        occ = evts["occurred_at"].values
        for idx, h in hours.items():
            w24  = evts[(evts["occurred_at"] >= h - pd.Timedelta(hours=24)) & (evts["occurred_at"] < h)]
            w7d  = evts[(evts["occurred_at"] >= h - pd.Timedelta(days=7))  & (evts["occurred_at"] < h)]
            past = occ[occ < np.datetime64(h)]
            df.at[idx, "incident_count_prev_24h"]        = len(w24)
            df.at[idx, "incident_max_severity_prev_24h"] = int(w24["severity"].max()) if len(w24) else 0
            df.at[idx, "incident_count_prev_7d"]         = len(w7d)
            df.at[idx, "hours_since_last_incident"]      = (
                (np.datetime64(h) - past[-1]) / np.timedelta64(1, "h") if len(past) else float("nan")
            )
            for t in TYPE_COLS:
                df.at[idx, f"{t}_count_prev_24h"] = int(w24[t].sum()) if len(w24) else 0

    return df


def compute_maintenance_lookback_features(df: pd.DataFrame, df_maint: pd.DataFrame) -> pd.DataFrame:
    """`days_since_last_maintenance` et `maintenance_count_prev_30d` — intra-machine, strictement passé."""
    df = df.copy()
    for col in ["days_since_last_maintenance", "maintenance_count_prev_30d"]:
        df[col] = 0
    df["days_since_last_maintenance"] = float("nan")

    for machine in df["machine_id"].unique():
        mask_f = df["machine_id"] == machine
        hours  = df.loc[mask_f, "observed_at"]
        evts   = df_maint[df_maint["machine_id"] == machine].sort_values("performed_at")
        if evts.empty:
            continue
        occ = evts["performed_at"].values
        for idx, h in hours.items():
            w30d = evts[(evts["performed_at"] >= h - pd.Timedelta(days=30)) & (evts["performed_at"] < h)]
            past = occ[occ < np.datetime64(h)]
            df.at[idx, "maintenance_count_prev_30d"] = len(w30d)
            df.at[idx, "days_since_last_maintenance"] = (
                (np.datetime64(h) - past[-1]) / np.timedelta64(1, "D") if len(past) else float("nan")
            )

    return df


def compute_multi_horizon_labels(df: pd.DataFrame, df_inc: pd.DataFrame) -> pd.DataFrame:
    """Lookahead inversé : pour chaque label event à t, marque True toutes les
    heures dans [t-H, t) pour chaque horizon H, et incrémente le compte brut."""
    df = df.copy()
    for label_col, count_col, _ in HORIZONS:
        df[label_col] = False
        df[count_col] = 0

    label_events = df_inc[df_inc["is_label_event"]]
    for _, evt in label_events.iterrows():
        for label_col, count_col, h in HORIZONS:
            mask = (
                (df["machine_id"] == evt["machine_id"]) &
                (df["observed_at"] >= evt["occurred_at"] - pd.Timedelta(hours=h)) &
                (df["observed_at"] <  evt["occurred_at"])
            )
            df.loc[mask, label_col] = True
            df.loc[mask, count_col] += 1

    return df


def build_gold_features(
    df_sensors: pd.DataFrame,
    df_inc: pd.DataFrame,
    df_maint: pd.DataFrame,
    df_machine: pd.DataFrame,
) -> pd.DataFrame:
    """Enchaîne les étapes dans l'ordre exact de TP6.ipynb (sections 2 à 6).
    Ne fait pas l'I/O (batch d'ingestion, écriture PostgreSQL) : voir TP6.ipynb.
    """
    df = pivot_to_wide(df_sensors)
    df = compute_rolling_features(df)
    df = compute_capacity_utilization(df, df_machine)
    df = compute_rolling_zscore(df)
    df = assign_time_split(df)
    df = compute_machine_zscore(df)
    df = compute_incident_lookback_features(df, df_inc)
    df = compute_maintenance_lookback_features(df, df_maint)
    df = compute_multi_horizon_labels(df, df_inc)
    return df
