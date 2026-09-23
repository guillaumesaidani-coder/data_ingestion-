# Progression pédagogique (esprit DL/ML) → notebook certifiant final

> Cas d'usage 01 — Résiliation client SaaS (churn). Le règlement de la certification
> impose **un seul notebook final** en 16 sections (cf. `Enonce cas usage_churn_saas.pdf`
> §3.1, compétences C1→C9). Cette progression ne remplace pas ce plan imposé : elle
> organise le travail *en amont*, TP par TP, pour que le notebook final n'ait plus qu'à
> **assembler** une matière déjà validée — pas à explorer à chaud sous 3 semaines.

## Principe

Comme pour `DL/` et `ML/`, chaque TP est un notebook d'expérimentation autonome,
numéroté, qui valide une étape avant de passer à la suivante. Le notebook certifiant
final n'est écrit qu'une fois TP1→TP10 stabilisés — il reprend leurs résultats dans
l'ordre imposé par le règlement, pas dans l'ordre de découverte.

**Leçon DL/TP10 appliquée d'emblée** : si un artefact de *documentation* doit être
produit à part (ex. model card du modèle churn), ce sera un script `.py → .md`, jamais
un notebook qui prétend être le document lui-même.

## Table de progression

| TP | Contenu | Compétence(s) | Équivalent DL/ML |
|---|---|---|---|
| **TP1** | Cadrage métier + gouvernance des données — relire le besoin (churn/CLV/rétention), inventorier les 3 fichiers (`churn_saas_complet.csv`, `churn_saas_echantillon.csv`, `catalogue_plans.csv`), dictionnaire de données, questions RGPD (données B2B, pas de données personnelles sensibles a priori — à vérifier) | C1, C2 (amorce) | TP1 DL (cadrage MVTec) |
| **TP2** | Nettoyage : dédoublonnage, parsing dates multi-formats, conversion texte→nombre (`%`, `€`, virgules décimales), normalisation casse/espaces des catégorielles, jointure `catalogue_plans.csv`, taux de NaN documenté + stratégie d'imputation. **Chargement du résultat nettoyé dans une vraie table PostgreSQL** (comme `ML/` : requêtes SQL plutôt que CSV en mémoire pour tous les TP suivants) | C3 | TP1-2 DL (normalisation), TP1-2 ML + schéma DB `ML/` |
| **TP3** | Détection de la fuite et des leurres — corrélation `sante_compte_fin_periode` ↔ `churn` démontrée empiriquement (pas juste affirmée), test des variables suspectes (`couleur_theme_interface`, `code_datacenter`, `groupe_experimentation`, `jour_souscription`) | C3 | ML TP7→TP8b (fuite `feature_row_id` trouvée puis corrigée) |
| **TP4** | EDA — distribution du churn (déséquilibre de classe), corrélations, analyse par secteur/taille/plan, ancienneté, lien usage↔churn | C3 | TP3 DL (EDA visuelle) |
| **TP5** | Baseline + comparaison ≥ 2 familles de modèles (LogReg + RandomForest ou XGBoost), sans tuning — démarche scientifique avant performance | C4 | TP3 ML (baseline comparatif) |
| **TP6** | Entraînement/validation croisée, courbe ROC + AUC, matrice de confusion, courbe précision-rappel (PR-AUC), seuil justifié par le coût métier d'un faux négatif | C5 | TP4 DL (arbitrage rappel/précision), TP9-11 ML (CV, Optuna optionnel) |
| **TP7** | Explicabilité — importance des variables / permutation, confirmation empirique des leurres, lecture métier des facteurs dominants | C4 (lecture explicable), C2 (biais) | TP5 DL (SHAP) |
| **TP8** | Régression CLV — modèle séparé, RMSE/MAE/R², vérification stricte qu'aucune fuite (ni `churn` ni `sante_compte_fin_periode` en feature du modèle churn, ni CLV en feature du modèle churn) | C3 (cible secondaire) | nouveau — pas d'équivalent DL/ML direct |
| **TP9** | Implémentation — sérialisation du modèle (joblib/MLflow), esquisse d'API de scoring, architecture cible (batch mensuel + intégration CRM) | C6, C7 | DL/TP7 (`config.py` + `scripts/`), ML `generate_model_card.py` |
| **TP10** | Impact métier + amélioration continue — traduire AUC/PR en « comptes sauvés » (rappel × CLV moyenne des vrais positifs), plan de monitoring (dérive d'usage, ré-entraînement, versioning) | C8, C9 | TP6 DL (mesure d'impact), ML MLOps |

## Notebook final certifiant

Reprend les 16 sections imposées (§3.1 de l'énoncé) en s'appuyant sur ce qui a été
validé TP par TP — pas de nouvelle exploration à ce stade, seulement synthèse, mise en
forme et journal de bord à chaque grande étape.

## Point technique à trancher avant TP5

`xgboost` n'est pas installé dans le venv racine (`pandas`/`scikit-learn` oui) — à
ajouter si XGBoost est retenu comme 2e famille de modèles, ou rester sur
LogReg + RandomForest (suffisant pour l'exigence « au moins deux familles »).

## Décision — PostgreSQL à partir de TP2

Contrairement à l'énoncé de certification (qui ne mentionne aucune base — données
livrées en CSV), ce projet introduit **volontairement** PostgreSQL dès TP2, pour
retrouver l'esprit `ML/` (requêtes SQL, table versionnée) plutôt que des CSV en mémoire.

- `psycopg2` (2.9.12) et `sqlalchemy` (2.0.51) sont **déjà présents** dans le venv
  racine (`py-init/.venv`) — rien à installer côté Python.
- **Postgres n'est pas joignable au moment de la rédaction de ce plan** (port 5432
  fermé, ni Docker Desktop ni service Windows Postgres détectés) — il l'était pendant
  les TP du projet `ML/`, donc probablement un conteneur/service à redémarrer avant
  d'implémenter TP2.
- Reste à trancher au moment de TP2 : nouvelle base dédiée (ex. `churn_saas_db`) ou
  nouvelle table dans la base `indusense_db` déjà utilisée par `ML/` (ex. table
  `churn_saas_clients`) — les deux projets restent indépendants, donc une base séparée
  est probablement plus propre.
