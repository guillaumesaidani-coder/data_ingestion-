import numpy as np
import pandas as pd

NUMERIC_COLS = [
    "temperature_c",
    "pressure_bar",
    "voltage_mean_v",
    "rotation_mean_rpm",
]

IQR_FACTOR = 3  # seuil outlier : Q1 - 3*IQR  /  Q3 + 3*IQR

SILVER_COLUMNS = [
    "machine_id",
    "timestamp",
    "year",
    "month",
    "day",
    "hour",
    "day_of_week",
    "is_weekend",
    "temp_c",
    "pressure_bar",
    "voltage_v",
    "rotation_rpm",
    "pieces_produced",
    "is_stopped",
]


def build_silver_telemetry(bronze: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, int]]:
    """Bronze telemetry -> Silver : typage, tri, colonnes temporelles,
    flag d'arrêt, détection outliers IQR (facteur 3) -> NaN -> interpolation
    linéaire par machine, arrondi, renommage.

    Retourne (silver_df, outlier_report) où outlier_report compte, par
    colonne source, le nombre de valeurs traitées comme outliers.
    """
    df = bronze.copy()

    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df["machine_id"] = df["machine_id"].astype("category")

    df = df.sort_values(["machine_id", "timestamp"]).reset_index(drop=True)

    df["year"] = df["timestamp"].dt.year
    df["month"] = df["timestamp"].dt.month
    df["day"] = df["timestamp"].dt.day
    df["hour"] = df["timestamp"].dt.hour
    df["day_of_week"] = df["timestamp"].dt.dayofweek  # 0 = lundi
    df["is_weekend"] = df["day_of_week"].isin([5, 6])

    df["is_stopped"] = df["pieces_produced"] == 0

    outlier_report: dict[str, int] = {}
    for col in NUMERIC_COLS:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        low = Q1 - IQR_FACTOR * IQR
        high = Q3 + IQR_FACTOR * IQR

        mask = (df[col] < low) | (df[col] > high)
        outlier_report[col] = int(mask.sum())
        df.loc[mask, col] = np.nan

    df[NUMERIC_COLS] = df.groupby("machine_id", observed=True)[NUMERIC_COLS].transform(
        lambda g: g.interpolate(method="linear", limit_direction="both")
    )

    df[NUMERIC_COLS] = df[NUMERIC_COLS].round(2)

    df = df.rename(
        columns={
            "temperature_c": "temp_c",
            "voltage_mean_v": "voltage_v",
            "rotation_mean_rpm": "rotation_rpm",
        }
    )

    return df[SILVER_COLUMNS], outlier_report
