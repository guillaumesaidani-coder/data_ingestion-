# Runbook — alerte de dérive InduSense

Document de l'**opérateur**, joué **pendant** l'incident, sous pression —
à ne pas confondre avec `reports/drift/drift_spec.md` (le data scientist,
écrit à froid, avant l'incident). L'opérateur suit la règle ici écrite ;
il ne la réinvente jamais et ne touche jamais au modèle ni au seuil gelé.

## Alerte : `IndusenseDriftPSIEleve`

```
indusense_drift_psi{feature="temperature", reference="normale"} > 0.25
```

## Diagnostic (dans l'ordre, jamais sauté)

1. **Isoler la feature.** Quelles autres features du même relevé
   dérivent aussi ? Si `temp_mean_24h` seule dépasse 0,25 et que les 7
   autres restent sous 0,03 (cas F2 réel, `drift_spec.md` §7), c'est un
   signal localisé sur un capteur — pas un effondrement généralisé.
2. **Vérifier le rappel du modèle** (panneau Grafana "Rappel modèle",
   ou `uv run python scripts/evaluate_drift.py --fenetre N`). S'il reste
   proche de la référence (0,91, `artifacts/models/metrics.json`), le
   modèle n'est pas en cause — ce n'est **jamais** une raison de
   réentraîner à ce stade.
3. **Exclure un changement de régime.** Rejouer la même fenêtre contre
   la référence alternative :
   ```
   uv run --frozen python scripts/evaluate_drift.py --fenetre N --reference haute_charge
   ```
   Si le PSI retombe sous 0,25, c'est une campagne planifiée ou une
   saison différente — basculer la référence utilisée en production,
   ne pas alerter davantage.
4. **Confirmer la cause physique.** Si le PSI reste fort quelle que
   soit la référence testée, et que la persistance est confirmée sur 2
   fenêtres consécutives (`suivi_fenetres.csv`, colonne `persistant`) :
   capteur hors service ou mal étalonné.

## Action

Étalonnage physique du capteur concerné **avant** toute action sur le
modèle. Ne pas réentraîner. Ne pas toucher au seuil gelé
(`artifacts/models/metrics.json`, `threshold: 0.5`).

## Retour arrière

Si l'étalonnage échoue ou n'est pas immédiatement possible : isoler la
machine concernée (retirer ses prédictions du tableau de bord opérateur
le temps de l'intervention), sans impacter les autres machines du parc.

## Ce que l'opérateur ne fait jamais

- Décider seul d'un réentraînement — même si le PSI est très élevé.
  C'est le rappel du modèle (§6 de `drift_spec.md`, plancher 0,70) qui
  autorise cette décision, jamais le PSI seul, et seul le data
  scientist l'exécute (protocole m21).
- Ajuster le seuil gelé pour faire taire une alerte gênante.
- Recalculer les bins du PSI sur la fenêtre courante — ils sont figés
  sur la référence, sinon deux fenêtres ne sont plus comparables.

## Preuve rejouable

```
uv run --frozen python scripts/drift_windows.py
uv run --frozen python scripts/evaluate_drift.py --fenetre 2
# -> PSI max temp_mean_24h = 7.38 > 0.25 : alerte
uv run --frozen python scripts/evaluate_drift.py --fenetre 2 --reference haute_charge
# -> vérifie si la référence alternative absorbe l'alerte (pas le cas pour F2 : capteur, pas un régime)
```

Résultats mesurés et verdicts complets : `reports/drift/drift_spec.md`,
§7.

## Qui décide

| | Opérateur | Data scientist |
|---|---|---|
| Quand | Pendant l'incident | Après confirmation (~15 j, KPI de second rideau) |
| Constate | Rappel sous plancher (Grafana) | — |
| Décide un réentraînement | **Non, jamais** | Oui — seul habilité |
| Touche au modèle/seuil | **Non, jamais** | Oui — protocole m21 |
