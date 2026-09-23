# Spec de dérive InduSense — modules 31-34

Document du data scientist, figé **avant** l'incident, à froid — à ne pas
confondre avec `docs/runbook.md` (l'opérateur, pendant l'incident, sous
pression). Chaque case de ce document a été mesurée en exécutant les
scripts ci-dessous contre le vrai Gold (`gold_machine_hourly_feature`,
15 machines, juin 2025 → juin 2026) et le vrai modèle de production
(`artifacts/models/model.joblib`, b11-gkf — pas un modèle de dérive
séparé : voir la décision prise en amont de ce module).

## 1. Références figées

| Référence | Période | Justification |
|---|---|---|
| `normale` | 2026-03 | Température moyenne la plus basse observée sur 13 mois (46,10 °C) — profil calme. |
| `haute_charge` | 2025-09 | Température moyenne la plus haute observée (50,12 °C, à égalité avec octobre). |

Notre jeu de données synthétique n'a pas de vrai changement de régime
saisonnier marqué comme celui de l'exemple du guide (production quasi
plate, 1125-1196 pièces/24h toute l'année) — ce choix est le plus
honnête disponible dans nos vraies données, pas une reconstitution du
scénario du guide.

## 2. Bins du PSI

10 bins, quantiles calculés sur la référence **seule** (`indusense.drift.reference_bin_edges`)
— jamais recalculés sur la fenêtre courante, sinon deux fenêtres évaluées
contre la même référence ne seraient plus comparables entre elles.

## 3. Features surveillées

`temp_mean_24h`, `temp_std_24h`, `pressure_mean_24h`, `pressure_std_24h`,
`voltage_mean_24h`, `rotation_mean_24h`, `pieces_produced_sum_24h`,
`incident_count_prev_24h` — sous-ensemble représentatif de chaque
capteur, pas les ~70 colonnes du Gold.

## 4. KS : calculé partout, décisionnel nulle part

`ks_pvalue()` tourne pour chaque feature à chaque fenêtre, même cadence
que le PSI, écrit dans les mêmes CSV. Aucune règle d'alerte ne le lit :
il reste disponible pour une lecture humaine de confirmation. Comme le
PSI, il est structurellement aveugle au concept drift de F3 (features
inchangées ⇒ KS n'y voit rien non plus, cf. tableau ci-dessous).

## 5. Règle d'alerte

PSI > 0,25 sur au moins une feature. **Persistant** si la fenêtre
précédente évaluée contre la même référence l'était déjà
(`scripts/evaluate_drift.py` lit `reports/drift/suivi_fenetres.csv`).
Une alerte isolée, non persistante, est surveillée — pas escaladée.

## 6. KPI de second rideau — rappel du modèle

Rappel de référence du modèle en production (holdout test,
`artifacts/models/metrics.json`) : **0,9106**. Plancher d'alerte :
**rappel < 0,70** (≈23 % sous la référence) — c'est ce chiffre, jamais
le PSI seul, qui autorise un réentraînement (protocole m21).

## 7. Les 4 scénarios rejoués — preuve

| Fenêtre | Référence | PSI max (feature) | KS p-value min | Alerte PSI | Rappel | Précision | ROC-AUC | Verdict |
|---|---|---|---|---|---|---|---|---|
| F1 témoin (2026-04, réel) | normale | 0,0406 (temp_mean_24h) | 3,9e-04 | non | 0,9480 | 0,9379 | 0,9987 | RAS |
| F2 capteur menteur (2026-04 + 8 °C, synthétique) | normale | 7,3845 (temp_mean_24h) | 0,0 | **OUI** (persistant sur 2 exécutions) | 0,8871 | 0,8746 | 0,9974 | Dérive capteur — rappel dans la norme |
| F3 concept drift (2026-05, labels mélangés, synthétique) | normale | 0,1653 (temp_mean_24h) | 4,9e-158 | non | **0,0433** | 0,0259 | 0,4977 | PSI muet — rappel effondré |
| F4 campagne (2025-10, réel) | normale | 2,6149 (temp_mean_24h) | 0,0 | **OUI** | 1,0000 | 0,8961 | 1,0000 | Fausse alerte de régime |
| F4 campagne (2025-10, réel) | haute_charge | 0,0259 (pressure_std_24h) | 1,2e-21 | non | 1,0000 | 0,8961 | 1,0000 | Contre-épreuve OK |

Reproductible : `uv run --frozen python scripts/drift_windows.py` puis
`uv run --frozen python scripts/evaluate_drift.py --fenetre N [--reference haute_charge]`.

## 8. Ce que ça démontre

- **F1** : pas de fausse alerte au repos sur un mois réel différent de la référence.
- **F2** : le PSI isole le capteur (température seule dérive, 7,38 —
  les 7 autres features restent sous 0,03) ; le rappel reste dans la
  norme (0,89 vs 0,91 attendu) — **pas un effondrement du modèle**.
- **F3** : le PSI seul ne suffit pas — 0,165, sous le seuil d'alerte —
  alors que le rappel s'effondre à 0,043. La relation X→y a changé
  (labels mélangés), invisible à une mesure qui ne regarde que X.
  D'où le KPI de second rideau (§6).
- **F4** : une seule référence produirait une fausse alerte (PSI 2,61
  vs `normale`) ; la référence alternative (`haute_charge`) l'absorbe
  (PSI 0,03) — changement de régime saisonnier, pas une panne.

## 9. Les 3 réactions possibles à une alerte

| Signal observé | Diagnostic | Action | Modèle modifié ? |
|---|---|---|---|
| PSI fort et persistant sur une seule feature, rappel stable | Dérive physique du capteur | Étalonnage physique **avant** toute action sur le modèle | non |
| PSI fort vs `normale`, RAS vs `haute_charge` | Changement de régime | Basculer sur la référence adaptée, ne pas alerter | non |
| PSI quasi nul, rappel et ROC effondrés | Concept drift — X→y a changé | Réentraînement complet (protocole m21) + nouvelle référence figée + cette spec republiée | **oui** |

Ce qui est explicitement interdit : un simple ajustement de seuil pour
faire taire une alerte gênante. Le seuil (§6) ne bouge qu'après un
réentraînement complet et une nouvelle validation — jamais isolément.
