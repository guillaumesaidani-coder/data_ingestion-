"""Module 36 — interface de revue terrain (InduSense).

Deux flux, adaptés du gabarit `feuille_de_route_mlops_indusense.md` à
nos vraies features (température/pression/vibration = nos colonnes
`temp_mean_1h`/`pressure_mean_1h`/`rotation_mean_1h`, pas les libellés
génériques du gabarit) :

1. Affiner une fausse alerte — la vérité terrain connue (§`scripts/
   simulate_feedback.py`) dit seulement « pas de panne dans les 24h »,
   jamais « capteur sain ». Un technicien tranche capteur défaillant /
   maintenance préventive / vraie fausse alerte.
2. Déclarer un incident non prédit — le retour le plus précieux
   (feuille de route §4) : sans ce bouton, les pannes manquées par le
   modèle ne sont jamais visibles dans les retours (mesuré dans
   hitl_proof.md, module 36).

Usage : uv run --frozen streamlit run scripts/streamlit_review.py
"""

import json
import sys
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from indusense.config import get_predictions_engine
from indusense.predictions_store import review_prediction

st.set_page_config(page_title="InduSense — Revue terrain", layout="wide")
st.title("Validation des alertes machines — InduSense")

engine = get_predictions_engine()

st.header("Affiner une fausse alerte")
st.caption(
    "« Pas de panne dans les 24h » ne veut pas dire « capteur sain ». "
    "Ces fenêtres viennent du split test réel (scripts/backfill_predictions.py) "
    "et ont déjà été classées FAUSSE_ALERTE par le simulateur — à un technicien "
    "de trancher la vraie cause."
)

pending_fa = pd.read_sql(
    "SELECT machine_id, window_start, failure_proba_24h, features_payload "
    "FROM predictions WHERE review_status = 'FAUSSE_ALERTE' "
    "ORDER BY window_start LIMIT 1",
    engine,
)

if pending_fa.empty:
    st.success("Plus aucune fausse alerte en attente d'affinage.")
else:
    row = pending_fa.iloc[0]
    features = json.loads(row["features_payload"])
    st.subheader(f"Machine : {row['machine_id']} — fenêtre {row['window_start']}")
    col1, col2, col3 = st.columns(3)
    col1.metric("Probabilité de panne", f"{row['failure_proba_24h']:.1%}")
    temp = features.get("temp_mean_1h")
    col2.metric("Température (temp_mean_1h)", f"{temp:.1f} °C" if temp is not None else "N/A")
    rotation = features.get("rotation_mean_1h")
    col3.metric("Vibration (rotation_mean_1h)", f"{rotation:.0f} tr/min" if rotation is not None else "N/A")

    choix = st.radio(
        "Constat réel sur le terrain :",
        [
            "Fausse alerte confirmée (machine saine)",
            "Capteur défaillant",
            "Maintenance préventive effectuée",
        ],
        key="verdict_fausse_alerte",
    )
    commentaire = st.text_input("Commentaire ou numéro de bon d'intervention :", key="commentaire_fausse_alerte")

    if st.button("Enregistrer la qualification", type="primary", key="valider_fausse_alerte"):
        statut = {
            "Fausse alerte confirmée (machine saine)": "FAUSSE_ALERTE",
            "Capteur défaillant": "CAPTEUR_DEFAILLANT",
            "Maintenance préventive effectuée": "MAINTENANCE_PREVENTIVE",
        }[choix]
        review_prediction(
            engine,
            row["machine_id"],
            row["window_start"],
            review_status=statut,
            ground_truth=False,
            reviewer_comment=commentaire or None,
            reviewed_at=datetime.now(UTC).isoformat(),
        )
        st.success("Merci ! Votre retour est enregistré.")
        st.rerun()

st.divider()
st.header("Déclarer un incident non prédit")
st.caption(
    "Le modèle n'a rien vu venir, mais la machine s'est arrêtée — "
    "c'est le retour le plus précieux (feuille de route MLOps, §5)."
)

candidates = pd.read_sql(
    "SELECT machine_id, window_start FROM predictions WHERE review_status = 'A_VALIDER' "
    "ORDER BY window_start LIMIT 200",
    engine,
)
if candidates.empty:
    st.info("Aucune fenêtre en attente de revue.")
else:
    options = [f"{r.machine_id} — {r.window_start}" for r in candidates.itertuples()]
    selection = st.selectbox("Machine et fenêtre concernées :", options, key="select_incident")
    commentaire_incident = st.text_input("Détail de l'incident :", key="commentaire_incident")
    if st.button("Déclarer l'incident", key="declarer_incident"):
        machine_id, window_start = selection.split(" — ")
        review_prediction(
            engine,
            machine_id,
            window_start,
            review_status="INCIDENT_NON_PREDIT",
            ground_truth=True,
            reviewer_comment=commentaire_incident or None,
            reviewed_at=datetime.now(UTC).isoformat(),
        )
        st.success("Incident enregistré — il sera pris en compte au prochain arbitrage (module 37).")
        st.rerun()
