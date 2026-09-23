"""Contrôle qualité des capteurs (module 38, feuille de route §6, feu
vert 3 : « aucun capteur n'a envoyé de valeurs absurdes, ex. température
de 9 999 °C »). Bornes physiques réutilisées du contrat DAT (§5.4,
`DAT_InduSense_Sprint3_v1.0.pdf`) plutôt qu'inventées : `temperature
entre -20 et 200`, `pressure_bar strictement positive et au plus 400` —
même contrat que celui documenté pour l'API, appliqué ici aux agrégats
du Gold.
"""

import pandas as pd
from pandera.errors import SchemaErrors
from pandera.pandas import Check, Column, DataFrameSchema

SENSOR_QUALITY_SCHEMA = DataFrameSchema(
    {
        "temp_mean_24h": Column(float, Check.in_range(-20, 200), nullable=True),
        "pressure_mean_24h": Column(
            float, Check.in_range(0, 400, include_min=False), nullable=True
        ),
        "voltage_mean_24h": Column(float, Check.in_range(0, 500), nullable=True),
        "rotation_mean_24h": Column(float, Check.in_range(0, 5000), nullable=True),
    },
    strict=False,  # ne juge que ces 4 colonnes, le Gold en porte ~70 au total
)


def check_sensor_quality(df: pd.DataFrame) -> tuple[bool, list[str]]:
    """Valide les bornes physiques sur un DataFrame de features Gold.
    `lazy=True` : collecte toutes les violations plutôt que de s'arrêter
    à la première — utile pour un feu vert Prefect qui doit expliquer
    précisément pourquoi il bloque, pas juste qu'il bloque."""
    try:
        SENSOR_QUALITY_SCHEMA.validate(df, lazy=True)
        return True, []
    except SchemaErrors as exc:
        messages = [
            f"{row['column']} : {row['check']} — {row['failure_case']}"
            for row in exc.failure_cases.to_dict(orient="records")
        ]
        return False, messages
