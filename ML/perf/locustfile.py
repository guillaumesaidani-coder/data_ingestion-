"""Harnais de charge InduSense (doc `sous-la-jauge`, modules 33-34) —
trafic réel vers `/predict-tabular` pour observer les panneaux Grafana
(débit, latence, taux d'erreur) en conditions réelles, pas seulement en
lisant du code.

`FEATURES` est une ligne réelle du Gold (MACH-01, 2026-04-14, split
test — voir `scripts/perf_sample_row.py` pour la régénérer), gelée ici
pour que chaque requête simulée soit identique à ce que le modèle voit
vraiment en production, pas un dict inventé.

Usage (doc de référence, 3 users, 10 min bornées) :
    uv run --frozen locust -f perf/locustfile.py --host http://127.0.0.1:8010 \
        --headless -u 3 -r 1 -t 10m

Interface web (par défaut sur :8089) :
    uv run --frozen locust -f perf/locustfile.py --host http://127.0.0.1:8010

INDUSENSE_API_KEY doit correspondre à API_KEY côté service (voir
compose.yaml, `dev-local-key` en local).
"""

import os

from locust import HttpUser, between, task

API_KEY = os.getenv("INDUSENSE_API_KEY", "dev-local-key")

# MACH-01, fenêtre 2026-04-14T01:00:00Z, split test (gold_machine_hourly_feature).
FEATURES = {
    "temp_mean_24h": 48.163250000000005,
    "temp_max_24h": 54.412,
    "temp_std_24h": 3.473918114351519,
    "pressure_mean_24h": 199.99224999999998,
    "pressure_std_24h": 1.3773335060747278,
    "incident_count_prev_24h": 0,
    "incident_max_severity_prev_24h": 0,
    "temp_mean_1h": 50.782,
    "temp_max_1h": 50.782,
    "temp_std_1h": None,
    "pressure_mean_1h": 201.725,
    "pressure_max_1h": 201.725,
    "pressure_std_1h": None,
    "voltage_mean_1h": 229.12,
    "rotation_mean_1h": 1604.36,
    "pieces_produced_sum_1h": 31,
    "temp_mean_6h": 46.52866666666666,
    "temp_max_6h": 48.642,
    "temp_std_6h": 1.6882377399728246,
    "pressure_mean_6h": 199.6155,
    "pressure_max_6h": 200.78,
    "pressure_std_6h": 0.6852406146751617,
    "voltage_mean_6h": 226.75833333333333,
    "voltage_std_6h": 0.6819213053338955,
    "rotation_mean_6h": 1593.1933333333334,
    "rotation_std_6h": 24.03394821220683,
    "temp_mean_12h": 45.47616666666667,
    "temp_max_12h": 48.642,
    "temp_std_12h": 1.7076643183743694,
    "pressure_mean_12h": 199.26424999999998,
    "pressure_max_12h": 200.78,
    "pressure_std_12h": 0.7183419197250603,
    "voltage_mean_12h": 226.81083333333333,
    "voltage_std_12h": 0.7305347771378542,
    "rotation_mean_12h": 1593.868333333333,
    "rotation_std_12h": 21.028917679897408,
    "pressure_max_24h": 202.579,
    "voltage_mean_24h": 227.68416666666667,
    "voltage_std_24h": 1.1094727171702086,
    "rotation_mean_24h": 1600.6683333333333,
    "rotation_std_24h": 26.954307821666056,
    "temp_delta_1h": 2.1399999999999935,
    "temp_delta_3h": 5.469999999999999,
    "temp_trend_6h": 5.859999999999999,
    "pressure_delta_1h": 0.9449999999999932,
    "pressure_delta_3h": 2.4429999999999836,
    "pressure_trend_6h": 2.1359999999999957,
    "voltage_delta_1h": 2.180000000000007,
    "voltage_delta_3h": 2.7700000000000102,
    "voltage_trend_6h": 2.4000000000000057,
    "rotation_delta_1h": 14.899999999999864,
    "rotation_delta_3h": 30.199999999999818,
    "rotation_trend_6h": -12.400000000000091,
    "temp_zscore_24h": 0.7538318157763592,
    "pressure_zscore_24h": 1.258046792848441,
    "temp_zscore_machine": 0.13888672844209843,
    "pressure_zscore_machine": 1.041758075880343,
    "pieces_produced_sum_24h": 1010.0,
    "incident_count_prev_7d": 0,
    "hours_since_last_incident": 292.73333333333335,
    "type_surchauffe_count_prev_24h": 0,
    "type_baisse_pression_count_prev_24h": 0,
    "type_vibration_count_prev_24h": 0,
    "type_bruit_mecanique_count_prev_24h": 0,
    "type_surconsommation_count_prev_24h": 0,
    "type_blocage_mecanique_count_prev_24h": 0,
    "type_alarme_capteur_count_prev_24h": 0,
    "type_arret_urgence_count_prev_24h": 0,
    "type_defaut_qualite_count_prev_24h": 0,
    "days_since_last_maintenance": 2.625,
    "maintenance_count_prev_30d": 18,
    "voltage_max_6h": 228.03,
    "rotation_max_6h": 1627.26,
    "voltage_max_12h": 228.12,
    "rotation_max_12h": 1627.26,
    "voltage_max_24h": 229.56,
    "rotation_max_24h": 1664.06,
    "capacity_utilization_pct": 64.58333333333334,
}


class IndusenseApiUser(HttpUser):
    wait_time = between(1, 3)

    @task
    def predict_tabular(self):
        self.client.post(
            "/predict-tabular",
            json={"features": FEATURES},
            headers={"X-API-Key": API_KEY},
            name="/predict-tabular",
        )
