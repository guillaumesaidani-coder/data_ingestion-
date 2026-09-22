"""Snapshot versionnable du Gold dataset.

Le trou identifié plus tôt : rien ne lie un run MLflow à l'état exact de
`gold_machine_hourly_feature` (Postgres) qui l'a produit — le schéma a
dérivé sans laisser de trace (67 vs 69 vs 78 colonnes selon le moment).
export_gold_dataset() fige un instantané en fichier, versionnable par DVC
(`dvc add`) exactement comme telemetry.csv ; hash_file() donne l'empreinte
à faire suivre au run d'entraînement pour retrouver quelle version de
données l'a produit.
"""

import hashlib
from pathlib import Path

import pandas as pd
from sqlalchemy.engine import Engine

GOLD_QUERY = "SELECT * FROM gold_machine_hourly_feature ORDER BY machine_id, window_start"


def export_gold_dataset(engine: Engine, out_path: Path) -> Path:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.read_sql(GOLD_QUERY, engine)
    df.to_csv(out_path, index=False)
    return out_path


def hash_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    """MD5 du contenu du fichier (même famille de hash que les pointeurs .dvc)."""
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            h.update(chunk)
    return h.hexdigest()
