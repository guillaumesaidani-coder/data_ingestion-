# Analyse du pipeline Bronze → Silver → Gold — Artifacts InduSense

Synthèse du contenu de `ML/artifacts/` et du code source qui le produit. Notebooks concernés copiés dans ce répertoire : [TP1.ipynb](TP1.ipynb), [TP2.ipynb](TP2.ipynb), [TP6.ipynb](TP6.ipynb).

## Contenu de ML/artifacts/

- **`model_card.md`** — model card pour `indusense-xgb-maintenance-b11-gkf` (XGBoost, prédiction panne à 24h). 112 996 obs. horaires, 15 machines, 69 features. PR-AUC test = 0.8799, ROC-AUC = 0.9949, F1 = 0.76, rappel 87.8 %. Limites documentées : forte hétérogénéité par machine (PR-AUC 0.39 à 1.00), sur-ajustement structurel (train 1.00 vs CV ~0.78), historique de fuite de données corrigé (`feature_row_id` retiré depuis TP8b), pas de calibration des probabilités. Artefact MLflow : `runs:/fa336bc0880b48bc849a4a6b1e6f412d/xgboost_b11_gkf`.
- **`ingestions/incidents/run_log.md`** — journal de 6 runs (16/06/2026) sur `releves_incidents.csv`. **Non maintenu** : le tableau s'arrête au run `202606161651` alors que le dossier contient ~25 runs jusqu'au 22/06/2026.
- **`ingestions/incidents/`** — ~25 sous-dossiers horodatés (`YYYYMMDDHHmm`), CSV bronze/silver anonymisés + PNG (distributions, boxplots, corrélations, ACF/PACF, tendances mensuelles, taux d'arrêts). Nombreux runs quasi-identiques (pas de nettoyage).
- **`ingestions/telemetry/202606181708/`** — un seul run (boxplots/distribution bronze + `telemetry_silver.csv`).
- **`emissions.csv`** — suivi CodeCarbon des entraînements.

## Code générateur

### TP1.ipynb — Bronze incidents + premiers artefacts

Fonction pivot (réutilisée à l'identique dans TP2, copiée-collée, pas de module partagé) :

```python
from datetime import datetime
from pathlib import Path

def create_ingestion_dir(base: str = "artifacts/ingestions", topic: str = "incidents") -> Path:
    timestamp = datetime.now().strftime("%Y%m%d%H%M")
    run_dir   = Path(base) / topic / timestamp
    run_dir.mkdir(parents=True, exist_ok=True)
    print(f"Répertoire créé : {run_dir}")
    return run_dir
```

Chaque cellule productrice d'artefact rappelle `create_ingestion_dir()` puis `plt.savefig(run_dir / "xxx.png")` ou `to_csv(run_dir / "xxx.csv")` → un nouveau sous-dossier horodaté à chaque exécution, ce qui explique la prolifération de runs quasi-identiques.

Pipeline :
1. `pd.read_csv("releves_incidents.csv")`
2. Anonymisation SHA-256 salée (`anon()`) → colonne `operator_anon`, `operator_name` supprimée
3. Export `releves_incidents_anonymised.csv`
4. Distribution (jour/semaine/shift) → `distribution_incidents.png`
5. Histogrammes signal/machine → `histogramme_par_signal.png`, `histogramme_par_machine.png`
6. Corrélations Pearson + co-occurrence (seaborn heatmap) → `correlation_pearson.png`, `correlation_cooccurrence.png`

### TP2.ipynb — Bronze → Silver télémétrie

Pipeline "Bronze to Silver" sur `telemetry.csv` (135 626 lignes, 15 machines) :

1. Chargement bronze, analyse NaN (MAR, gaps courts ≤ 15h) et outliers (IQR facteur 3) → `temperature_mach01.png`, `Distribution des variables — Bronze.png`, `Boxplots par machine — Bronze.png`
2. Croisement outliers × incidents (±24h) → conclusion : outliers conservés en feature (signal prédictif), pas supprimés
3. **Production du silver** : outliers → NaN → interpolation linéaire par machine (`groupby("machine_id").transform(interpolate)`), renommage colonnes, export `telemetry_silver.csv` (1201 → 11 outliers résiduels)
4. Analyse séries temporelles sur le silver :
   - `tendance_mensuelle_temp.png` (moyenne mensuelle par machine)
   - `taux_arrets_mensuel.png` (% heures à l'arrêt/mois)
   - `rolling_7j_temperature.png` (moyenne/écart-type glissants 168h, bande ±2σ)
   - `acf_pacf_temp_mach01.png` (`statsmodels.plot_acf/plot_pacf`, lags=48 → saisonnalité journalière, pic lag=24)

### TP6.ipynb — Silver → Gold (`gold_machine_hourly_feature`)

Pipeline réel de construction du Gold Dataset (132 940 lignes × 91 colonnes), lu/écrit directement en PostgreSQL — **pas** via le module `src/indusense/processing/ingestion.py` référencé par `ML/gold_roadmap.md` (ce module n'existe pas dans le repo ; la roadmap est une documentation aspirationnelle/générée, pas le code exécuté).

1. Lecture Silver (`silver_sensor_reading` filtré non-manquant/non-dupliqué, `silver_incident` joint à `bronze_incidents`, `silver_maintenance`, référentiel `machine`)
2. Pivot long → large (1 ligne / machine / heure)
3. Rolling par machine (`closed='left'`, anti-leakage) : agrégats 1h bruts, 6h/12h/24h (mean/max/std), deltas 1h/3h, trend 6h
4. `capacity_utilization_pct` (jointure statique avec capacité machine)
5. Z-scores : glissant 24h + z-score "machine" avec baseline **train-only** (split chronologique 70/15/15)
6. Lookback incidents (comptage 24h/7j, sévérité max, `hours_since_last_incident`, par type) — boucle Python non-vectorisée
7. Lookback maintenance (`days_since_last_maintenance`, `maintenance_count_prev_30d`)
8. Labels multi-horizons par lookahead inversé : `label_failure_next_6h/12h/24h/48h` + `future_incident_count_*` (exclu des features, fuite potentielle)
9. Écriture : `TRUNCATE` puis `to_sql(method='multi', chunksize=2000)`, tracé via `ingestion_batch`

Schéma de table cible : `GoldMachineHourlyFeature` dans `ML/models.py:160`.

## Points notables

- **`run_log.md` sans code générateur** : aucune trace dans le repo (notebooks + `.py`) d'un script qui écrit/met à jour ce fichier — il est rédigé manuellement et n'a pas suivi les runs réels.
- **`create_ingestion_dir()` dupliquée** entre TP1 et TP2 (copier-coller), pas de module utilitaire partagé.
- **`gold_roadmap.md`** documente une architecture cible (`src/indusense/processing/ingestion.py`, `build_gold_from_telemetry()`, `indusense.db.gold_loader`) qui n'existe pas dans le code réel — le Gold est en fait produit par le notebook `TP6.ipynb` en SQL/pandas direct.
