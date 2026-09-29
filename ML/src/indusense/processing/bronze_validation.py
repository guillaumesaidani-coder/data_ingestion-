"""Validation Bronze des lots terrain (extrait de TP4.ipynb, sections 2,
3 et 10).

Le Bronze reste un miroir du fichier source (docs/03_donnees/BRONZE.md) : une ligne
invalide n'est ni corrigée ni supprimée, elle est écrite avec
`parse_ok=False` et la raison du rejet (`parse_ok_reason`), puis tracée
dans `data_quality_issue`. Le Silver ne lit que `parse_ok=True`.

Deux familles de règles :
- règles *ligne* (Pydantic) : types, champs obligatoires, plages
  plausibles, flags binaires ;
- règle *entre lignes* (pandas) : doublon horaire en télémétrie — plus
  d'un relevé pour la même machine à la même heure.

Décisions (cadrage, rupture ① point 2) :
- Doublon horaire : **toutes** les copies sont rejetées, comme TP4. Les
  copies d'un même doublon n'ont pas les mêmes valeurs (double
  transmission capteur, ex. MACH-01 2025-06-01 00:00 : 46.348 °C puis
  46.332 °C) : rien ne permet de dire laquelle est la bonne, on n'en
  choisit pas une au hasard. La règle ne voit qu'un lot à la fois ; un
  doublon réparti sur deux lots est départagé plus loin par le Silver
  (`is_duplicate`, le premier arrivé gagne) — compromis assumé, le
  Bronze ne réécrit jamais une ligne déjà archivée.
- Plages : le Bronze rejette l'*impossible* (capteur fou, erreur de
  saisie), pas le *hors contrat*. Ses plages doivent donc englober
  celles du contrat DAT §5.4, que le feu vert n°3 contrôle sur le Gold
  (`indusense.data_quality`) — sinon une valeur conforme au contrat
  serait jetée à l'entrée sans que le feu n°3 la voie jamais. D'où une
  pression acceptée jusqu'à 400 bar (TP4 : 300), comme le contrat ;
  aucune valeur de telemetry.csv ne dépasse 300 bar, le Gold n'en est
  pas modifié.
"""

import pandas as pd
from pydantic import BaseModel, ValidationError, field_validator

from indusense.processing.gold_features import TYPE_COLS

PRESSURE_MAX_BAR = 400  # = borne haute du contrat DAT (feu vert n°3)


class TelemetryRow(BaseModel):
    machine_id: str
    timestamp: str
    temperature_c: float | None = None
    pressure_bar: float | None = None
    voltage_mean_v: float | None = None
    rotation_mean_rpm: float | None = None
    pieces_produced: int | None = None

    model_config = {"str_strip_whitespace": True}

    @field_validator("machine_id", "timestamp")
    @classmethod
    def not_empty(cls, v):
        if not v or not v.strip():
            raise ValueError("champ obligatoire vide")
        return v

    @field_validator("temperature_c")
    @classmethod
    def temp_plausible(cls, v):
        if v is not None and not (-50 <= v <= 500):
            raise ValueError(f"temperature hors plage [-50, 500] : {v}")
        return v

    @field_validator("pressure_bar")
    @classmethod
    def pressure_plausible(cls, v):
        if v is not None and not (0 <= v <= PRESSURE_MAX_BAR):
            raise ValueError(f"pression hors plage [0, {PRESSURE_MAX_BAR}] : {v}")
        return v

    @field_validator("pieces_produced")
    @classmethod
    def pieces_positive(cls, v):
        if v is not None and v < 0:
            raise ValueError(f"pieces_produced negatif : {v}")
        return v


class IncidentRow(BaseModel):
    incident_id: str
    date: str
    time: str
    operator_name: str | None = None
    machine_id: str
    severity: int
    operator_badge: str | None = None
    comment: str | None = None
    shift: str | None = None
    type_surchauffe: int
    type_baisse_pression: int
    type_vibration: int
    type_bruit_mecanique: int
    type_surconsommation: int
    type_blocage_mecanique: int
    type_alarme_capteur: int
    type_arret_urgence: int
    type_defaut_qualite: int

    model_config = {"str_strip_whitespace": True}

    @field_validator("severity")
    @classmethod
    def severity_range(cls, v):
        if not (1 <= v <= 5):
            raise ValueError(f"severity hors [1-5] : {v}")
        return v

    @field_validator(*TYPE_COLS)
    @classmethod
    def binary_flag(cls, v):
        if v not in (0, 1):
            raise ValueError(f"flag binaire attendu (0 ou 1), recu : {v}")
        return v


class MaintenanceRow(BaseModel):
    maintenance_id: int
    machine_id: str
    maintenance_at: str
    maintenance_type: str
    action_type: str
    component: str
    description: str | None = None
    related_incident_id: str | None = None
    duration_hours: float

    model_config = {"str_strip_whitespace": True}

    @field_validator("maintenance_type")
    @classmethod
    def type_valid(cls, v):
        if v not in ("proactive", "reactive"):
            raise ValueError(f"maintenance_type inconnu : {v}")
        return v

    @field_validator("duration_hours")
    @classmethod
    def duration_positive(cls, v):
        if v <= 0:
            raise ValueError(f"duration_hours doit etre positif : {v}")
        return v


ROW_MODELS: dict[str, type[BaseModel]] = {
    "telemetry": TelemetryRow,
    "incidents": IncidentRow,
    "maintenance": MaintenanceRow,
}

# Typage des colonnes numériques avant écriture (TP4 §6) : les lignes
# rejetées gardent leur valeur brute, convertie ici en NaN si illisible.
_FLOAT_COLS = {
    "telemetry": ["temperature_c", "pressure_bar", "voltage_mean_v", "rotation_mean_rpm"],
    "incidents": [],
    "maintenance": ["duration_hours"],
}
_INT_COLS = {
    "telemetry": ["pieces_produced"],
    "incidents": ["severity", *TYPE_COLS],
    "maintenance": ["maintenance_id"],
}

HOURLY_DUPLICATE_RULE = "DOUBLON_HORAIRE"
PYDANTIC_RULE = "PYDANTIC_ERROR"


def _format_errors(e: ValidationError) -> str:
    return " | ".join(
        f"{'.'.join(str(x) for x in err['loc'])} : {err['msg']}" for err in e.errors()
    )


def _clean(record: dict) -> dict:
    """Chaîne vide ou NaN -> None, pour que Pydantic voie un champ absent."""
    return {
        k: (None if v == "" or (isinstance(v, float) and pd.isna(v)) else v)
        for k, v in record.items()
    }


def validate_rows(raw: pd.DataFrame, model: type[BaseModel]) -> pd.DataFrame:
    """Applique le modèle Pydantic ligne par ligne. Ordre de sortie de TP4 :
    lignes valides d'abord, puis lignes rejetées."""
    valid, invalid = [], []
    for record in raw.to_dict("records"):
        cleaned = _clean(record)
        try:
            row = model(**cleaned).model_dump()
            valid.append({**row, "parse_ok": True, "parse_ok_reason": ""})
        except ValidationError as e:
            invalid.append({**cleaned, "parse_ok": False, "parse_ok_reason": _format_errors(e)})
    return pd.DataFrame(valid + invalid, columns=[*raw.columns, "parse_ok", "parse_ok_reason"])


def flag_hourly_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """Rejette toutes les copies d'un relevé (machine_id, timestamp) présent
    plus d'une fois parmi les lignes encore valides."""
    df = df.copy()
    valid = df[df["parse_ok"]]
    counts = valid.groupby(["machine_id", "timestamp"])["machine_id"].transform("count")
    dup_idx = counts.index[counts > 1]
    df.loc[dup_idx, "parse_ok"] = False
    df.loc[dup_idx, "parse_ok_reason"] = (
        f"{HOURLY_DUPLICATE_RULE} : "
        + counts[dup_idx].astype(str)
        + " releves pour "
        + df.loc[dup_idx, "machine_id"]
        + " a "
        + df.loc[dup_idx, "timestamp"]
    )
    return df


def validate_bronze(raw: pd.DataFrame, source_name: str) -> pd.DataFrame:
    """Lot brut (CSV lu en texte) -> lignes Bronze avec `parse_ok` /
    `parse_ok_reason`, colonnes numériques typées."""
    df = validate_rows(raw, ROW_MODELS[source_name])
    if source_name == "telemetry":
        df = flag_hourly_duplicates(df)
    for col in _FLOAT_COLS[source_name]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    for col in _INT_COLS[source_name]:
        df[col] = pd.to_numeric(df[col], errors="coerce").astype("Int64")
    return df


def _entity_key(row: pd.Series, source_name: str) -> str:
    if source_name == "telemetry":
        return f"{row.get('machine_id', '?')}|{row.get('timestamp', '?')}"
    key = "incident_id" if source_name == "incidents" else "maintenance_id"
    return str(row.get(key, "?"))


def data_quality_rows(bronze: pd.DataFrame, source_name: str, batch_id) -> list[dict]:
    """Une ligne `data_quality_issue` par ligne rejetée (TP4 §7 et §10e).
    Un doublon horaire est un WARNING (le relevé existe, on ne sait pas
    lequel garder) ; une erreur Pydantic est une ERROR (la ligne est
    inexploitable)."""
    rows = []
    for _, row in bronze[~bronze["parse_ok"]].iterrows():
        reason = str(row["parse_ok_reason"])
        rule = HOURLY_DUPLICATE_RULE if reason.startswith(HOURLY_DUPLICATE_RULE) else PYDANTIC_RULE
        rows.append(
            {
                "ingestion_batch_id": str(batch_id),
                "dataset_name": f"bronze_{source_name}",
                "rule_code": rule,
                "severity": "WARNING" if rule == HOURLY_DUPLICATE_RULE else "ERROR",
                "entity_key": _entity_key(row, source_name),
                "details": reason,
            }
        )
    return rows
