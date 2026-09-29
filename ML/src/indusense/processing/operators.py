"""Référentiel `operator` à partir d'un lot d'incidents (extrait de
TP4.ipynb, section 4).

Chaque incident nomme son opérateur (`operator_name`, `operator_badge`).
Le Silver retrouve `operator_id` par une jointure gauche sur
`operator_key` (`silver_events.build_silver_incidents`) : un opérateur
absent du référentiel y donnerait un `operator_id` NULL, sans erreur.
D'où ce module, appelé à l'ingestion de chaque lot d'incidents
(`indusense.ingest`), comme TP4 le fait pour le fichier complet.

`badge_hash` reprend TP4 à l'identique (sha256 sans sel, 16 caractères)
pour ne pas changer le hash des opérateurs déjà en base. Ce n'est PAS
`processing.anonymization.anon()` (salé, `OP_ANON_…`), qui sert à
produire le CSV anonymisé.

Décisions (docs/02_architecture/cadrage_operateurs_ingest.md §5) :
- Seules les lignes `parse_ok=True` créent un opérateur (TP4 prend
  toutes les lignes) : un nom issu d'une ligne rejetée n'est pas fiable.
  Sans effet sur releves_incidents.csv (0 ligne rejetée).
- Nom vide : pas d'opérateur. `operator_key` NULL échappe à la contrainte
  UNIQUE (NULL ≠ NULL), chaque rejeu insérerait une ligne de plus.
- Badge vide : `badge_hash` NULL, qui n'écrase pas le hash déjà connu
  (`UPSERT_OPERATOR_SQL`). Un badge renseigné et différent remplace
  l'ancien, comme TP4 (badge réattribué ; l'historique est hors
  périmètre).
"""

import hashlib

import pandas as pd

OPERATOR_COLUMNS = ["operator_key", "badge_hash"]

# SQLite >= 3.24 et PostgreSQL : même syntaxe, pas de branche par moteur.
UPSERT_OPERATOR_SQL = """
    INSERT INTO operator (operator_key, badge_hash) VALUES (:operator_key, :badge_hash)
    ON CONFLICT (operator_key)
    DO UPDATE SET badge_hash = COALESCE(EXCLUDED.badge_hash, operator.badge_hash)
"""


def operator_key(names: pd.Series) -> pd.Series:
    """Nom normalisé : la clé de jointure entre incidents et `operator`."""
    return names.str.strip().str.lower()


def badge_hash(name: str, badge: str) -> str:
    """Badge pseudonymisé, algorithme de TP4 §4."""
    raw = f"{str(name).strip().lower()}|{str(badge).strip().lower()}"
    return hashlib.sha256(raw.encode()).hexdigest()[:16]


def build_operators(bronze_incidents: pd.DataFrame) -> pd.DataFrame:
    """Lignes Bronze incidents -> un opérateur par `operator_key`
    (le premier rencontré gagne, comme le `drop_duplicates` de TP4)."""
    df = bronze_incidents[bronze_incidents["parse_ok"].astype(bool)].copy()
    df["operator_key"] = operator_key(df["operator_name"].astype("string"))
    df = df[df["operator_key"].fillna("") != ""]

    has_badge = df["operator_badge"].fillna("").astype(str).str.strip() != ""
    df["badge_hash"] = [
        badge_hash(name, badge) if ok else None
        for name, badge, ok in zip(
            df["operator_name"], df["operator_badge"], has_badge, strict=True
        )
    ]
    df = df.drop_duplicates(subset="operator_key", keep="first")
    return df[OPERATOR_COLUMNS].astype(object).reset_index(drop=True)
