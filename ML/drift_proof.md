# Preuve — dérive, observabilité, runbook (modules 31-34)

Sous-système construit sur le vrai Gold (`gold_machine_hourly_feature`,
15 machines, juin 2025 → juin 2026) et le vrai modèle de production
(`artifacts/models/model.joblib`, b11-gkf — pas de modèle de dérive
séparé, décision prise en amont de ce module pour éviter de dupliquer un
pipeline d'entraînement déjà existant, cf. la leçon du module 30). Les
chiffres du guide formateur (rf.joblib, panne_v1, PSI 6,83 sur un
capteur fictif) sont un exemple pédagogique — tout ce qui suit a été
mesuré sur nos vraies données.

| Contrôle | Statut | Preuve |
|---|---|---|
| PSI + KS | Implémenté | `src/indusense/drift.py`, 7 tests (`tests/test_drift.py`) |
| Références figées | Implémenté | `reports/drift/reference_normale.csv` (2026-03), `reference_haute_charge.csv` (2025-09) |
| 4 scénarios rejoués | Implémenté | `reports/drift/drift_spec.md` §7 — F1 réel, F2/F3 synthétiques documentés, F4 réel |
| Spec figée | Implémenté | `reports/drift/drift_spec.md` |
| Runbook | Implémenté | `docs/runbook.md`, alerte `IndusenseDriftPSIEleve` réellement déclenchée et observée |
| Export Prometheus | Implémenté | `scripts/export_drift_metrics.py` — port 9110/metrics, 2/2 cibles Prometheus UP |
| Dashboard Grafana | Implémenté | `grafana/dashboards/indusense_drift.json` (6 panneaux) + `indusense_slo.json` (4 panneaux), provisionnés automatiquement |
| SLO | Implémenté | Basé sur les métriques réellement exposées (`indusense_http_requests_total`, `indusense_http_request_duration_seconds_bucket`, module 28) |
| Non-régression | Implémenté | `uv run pytest -q` → 94 passed, 1 xfailed ; `ruff check src tests` + `black --check src tests` → OK |

## Piège rencontré : port 9109 déjà pris, en permanence

`netstat` a révélé un process Python écoutant sur `0.0.0.0:9109` depuis
le 2026-09-03 (PID persistant) — l'exporteur du formateur lui-même,
laissé en fonctionnement sur cette machine partagée pour produire les
captures de `m33-m34-sous-la-jauge-recap.html` (daté 2026-09-03).
Curl sur 9109 renvoyait ses métriques à lui (`feature="temperature"`,
`fenetre="janvier"`, PSI 6,20...), pas les nôtres — piège découvert en
testant, pas supposé. Notre exporteur tourne sur **9110** à la place
(`scripts/export_drift_metrics.py`, `prometheus.yml`).

## Reçu vs construit

Contrairement au jalon du guide (drift.py + modèle + exporteur +
dashboard "reçus", seule l'observabilité restant à faire), rien
n'existait dans ce dépôt avant ce module : tout — PSI/KS, références,
4 scénarios, spec, runbook, exporteur, dashboards — a été construit et
prouvé dans cette passe.

## TP1 — Références et fenêtres, sur le vrai Gold

```
uv run --frozen python scripts/drift_windows.py
reference_normale       (2026-03)      : 11040 lignes
reference_haute_charge  (2025-09)      : 10701 lignes
F1 témoin               (2026-04)      : 10680 lignes
F2 capteur menteur      (F1 + 8°C synth.) : 10680 lignes
F3 concept drift        (2026-05, labels mélangés synth.) : 11045 lignes
F4 campagne planifiée   (2025-10)      : 11047 lignes
```

## TP2 — 4 scénarios rejoués contre le vrai modèle

```
uv run --frozen python scripts/evaluate_drift.py --fenetre 1
PSI max : temp_mean_24h = 0.0406 (seuil 0.25) — Alerte : non
Rappel modèle : 0.9480 | Précision : 0.9379 | ROC-AUC : 0.9987

uv run --frozen python scripts/evaluate_drift.py --fenetre 2
PSI max : temp_mean_24h = 7.3845 (seuil 0.25) — Alerte : OUI
Rappel modèle : 0.8871 | Précision : 0.8746 | ROC-AUC : 0.9974
# rejoué une 2e fois : "persistante (2 fenêtres)"

uv run --frozen python scripts/evaluate_drift.py --fenetre 3
PSI max : temp_mean_24h = 0.1653 (seuil 0.25) — Alerte : non
Rappel modèle : 0.0433 | Précision : 0.0259 | ROC-AUC : 0.4977
KPI second rideau : rappel sous le plancher -> réentraînement à envisager

uv run --frozen python scripts/evaluate_drift.py --fenetre 4 --reference normale
PSI max : temp_mean_24h = 2.6149 — Alerte : OUI
uv run --frozen python scripts/evaluate_drift.py --fenetre 4 --reference haute_charge
PSI max : pressure_std_24h = 0.0259 — Alerte : non  # contre-épreuve
```

F3 illustre l'angle mort du PSI sans avoir eu besoin de forcer les
chiffres : PSI sous le seuil (0,1653), rappel effondré (0,0433) — la
relation X→y a changé (labels mélangés), invisible à une mesure qui ne
regarde que la distribution des features. Détail complet et
interprétation : `reports/drift/drift_spec.md`.

## TP3 — Pile en conditions réelles

```
docker compose config -q
docker compose up -d --build --wait
 Container ml-db-1         Healthy
 Container ml-api-1        Healthy
 Container ml-prometheus-1 Healthy
 Container ml-grafana-1    Healthy

uv run --frozen python scripts/export_drift_metrics.py &   # hôte, port 9110

curl -s http://127.0.0.1:9091/api/v1/targets
indusense-api   up
indusense-drift up

curl -s http://127.0.0.1:9091/api/v1/rules
IndusenseDriftPSIEleve  firing  ok
```

L'alerte est passée à `firing` dès le premier scrape, sans action
supplémentaire — les CSV de F2 (PSI 7,38 > 0,25) étaient déjà sur
disque au démarrage de l'exporteur.

Panneau lu en conditions réelles (requête exacte du panneau 1 du
dashboard dérive) :

```
curl -sG http://127.0.0.1:9091/api/v1/query --data-urlencode 'query=indusense_drift_psi{fenetre="2"}'
-> 8 séries, temp_mean_24h = 7.3845...
```

Panneau SLO lu en conditions réelles (trafic `/health` réel généré) :

```
curl -sG http://127.0.0.1:9091/api/v1/query --data-urlencode 'query=sum by (path) (rate(indusense_http_requests_total[2m]))'
-> {path="/health"} 0.0528, {path="/metrics"} 0.0234
```

Dashboards provisionnés automatiquement, vérifié via l'API Grafana
(`GET /api/search?query=InduSense`) : `InduSense — dérive & métriques`
et `InduSense — SLO API`, dossier `InduSense`, datasource `Prometheus`
provisionnée (`GET /api/datasources`).

## Runbook joué pour de vrai

`docs/runbook.md` a été suivi pas à pas sur l'alerte F2 réellement
déclenchée ci-dessus : isoler la feature (seule `temp_mean_24h` dérive,
les 7 autres restent sous 0,03), vérifier le rappel (0,89, dans la
norme), exclure un changement de régime
(`evaluate_drift.py --fenetre 2 --reference haute_charge` — non exécuté
pour F2 : la contre-épreuve n'a de sens que pour F4, un vrai changement
de mois ; documenté dans le runbook comme étape générique). Verdict :
dérive capteur, étalonnage physique, pas de réentraînement.
