# InduSense 4.0 : de la donnée brute à l'intelligence

- Conception d'une IA prédictive / hybride sur 21 jours.
- Ratio théorie / pratique : 30% / 70% - **Learning by Doing**
- Stack technique :
  - Python 3.13.x, SQLAlchemy, Scikit-Learn, PyTorch, MLFlow, Optuna, Prefect, FastAPI, Prometheus, Grafana, Evidently AI, CodeCarbon, ...

**Enjeu**
Concevoir une solution d'IA industrielle capable de 
prédire les pannes machines via des données de capteur (Machine Learning) 
et de détecter les défauts de production via des images (Deep Learning). 
L'ensemble en production avec une démarche CI/CD, éco-responsable, éthique.

**En amont**

- Mise à disposition d'un dépôt GIT avec les principales librairies,
- Projet basé sur pyproject.toml (pas requirements... trop vieux),
- Librairies avec les bonnes versions pour Python 3.13.x, (uv de préférence),
- Jeux de données qui seront disponibles (y compris images)

## Architecture globale

```mermaid
gantt
    title InduSense 4.0 : 21 Jours de la Data à la Production
    dateFormat YYYY-MM-DD
    axisFormat %d/%m
    
    section Acte 1 : Fondation
    Enquête Data & Anonymisation (C1/C2)       :a1, 2024-01-01, 1d
    Amalgamation & BDD (SQLAlchemy) (C1)       :a2, after a1, 1d
    Nettoyage & Outliers (C3)                   :a3, after a2, 1d
    Feature Eng. & Data Leakage (C3/C4)         :a4, after a3, 1d
    Réduction Dimensionnelle (C3)                :a5, after a4, 1d
    
    section Acte 2 : Modélisation
    POC ML & Métriques (Accuracy vs Recall)     :b1, after a5, 1d
    Préparation Images (C1/C3)                   :b2, after b1, 1d
    Auto-encodeur CNN & Loss Curves (C4/C5)      :b3, after b2, 1d
    Optuna & Eco-conception (CodeCarbon)         :b4, after b3, 1d
    Validation & ROC/AUC (C5)                     :b5, after b4, 1d
    Explicabilité (SHAP) (C4)                     :b6, after b5, 1d
    
    section Acte 3 : Industrialisation
    Packaging & Versioning (Docker/Git)          :c1, after b6, 1d
    Architecture API (FastAPI) (C7)               :c2, after c1, 1d
    Conteneurisation (C6)                          :c3, after c2, 1d
    Orchestration (Prefect) (C7)                   :c4, after c3, 1d
    Data Drift (Evidently AI) (C8)                 :c5, after c4, 1d
    Monitoring (Prometheus/Grafana) (C8)           :c6, after c5, 1d
    
    section Acte 4 : Production
    Tests de Charge & Impact (C8)                  :d1, after c6, 1d
    Cycle de vie & Retraining (C9)                 :d2, after d1, 1d
    Rétrospective & Documentation (C9)             :d3, after d2, 1d
    Soutenance Finale (Transverse)                  :d4, after d3, 1d
```

## Story telling

Le projet est architecturé autour de 4 *sprints* favorisant la montée en compétence et permettant l'ancrage des acquis par alternance d'approches théoriques et pratiques.

### Sprint 1

**Objectif du sprint** : Enquête et nettoyage de données.

A partir de données brutes, "sales", hétérogènes, l'objectif est de transformer le chaos en base de connaissance fiable, solide, évolutive.

**User Story # 1.1 (compétences C1, C2)**
En tant qu'architecte IA, je dois étudier les données fournies, enquêter sur leur utilité, leur importance métier, m'assurer de la conformité réglementaire des données et définir une ou plusieurs stratégies pour anonymiser, minimiser, retirer ou au contraire sur-alimenter les données pour :

- Garantir l'anonymat des opérateurs, éviter l'identification par combinaisons de features,
- Eviter le biais de données : machines sur-représentées, pannes minimisée, ...
**Techniquement** :
- Explorer des données CSV brutes,
- Extraire les principaux métriques (lignes, colonnes, doublons, features, ...),
- Produire des scripts/classes métiers pour anonymiser, minimiser, imputer

**User Story # 1.2 (compétences C1, C3)**
En tant qu'architecte IA, je dois uniformiser la donnée hétérogène, organiser relationnellement une base de données pour répondre aux exigences, prévoir un système de migration fiable pour anticiper le futur.

**Techniquement** :

- Automatiser la fusion et l'intégration de données dans une base de données relationnelle,
- Documenter le processus d'intégration,
- Créer les scripts de migration et de rollback à l'aide d'un ORM (SQLAlchemy)

**User Story # 1.3 (compétences C3, C4)**
En tant qu'architecte IA, je dois m'assurer avec l'aide des parties prenantes métier, de la détection des valeurs "hors normes", distinguer le bruit vs anomalie.

**Techniquement** :

- Déterminer / utiliser un algorithme de détection et de traitement des "outliers" (Z-Score, IQR, ...) et documenter / argumenter le choix,
- Déterminer / utiliser un algorithme d'imputation de valeurs manquantes (Mean, Median, KNN) et documenter /argumenter le choix,
- Créer les viewers (Boxplots) pour la détection et le traitement des outliers.

**User Story # 1.4 (compétences C3, C4)**
En tant qu'architecte IA, je dois anticiper le Data Leakage, le traitement de séries temporelles afin d'éviter que le futur influence le passé.

**Techniquement** :

- Comprendre pourquoi "train test split" est incorrect pour des séries temporelles,
- Comprendre le "feature engeeniring temporel" (Lags, Rolling Windows),
- Implémenter Time Series Split, créer le jeu de validation "futur".

**User Story # 1.5 (compétences C3)**
En tant qu'architecte IA, je dois simplifier pour réussir le jeu de données final, préparer le Gold Dataset prêt pour l'entraînement.

**Techniquement** :

- Utiliser PCA , Feature Selection pour réduire les dimensions et simplifier,
- Nettoyer le DataSet,
- Créer le Gold Dataset.

### Sprint 2

**Objectif du sprint** : Entraînement

Les données étant saines, propres, éthiquement acceptables, il faut prouver par l'expérimentation, la valeur apportée et réaliser une analyse critique des métriques.

**User Story # 2.1 (compétences C4, C5)**
Le Proof of Concept (POC) Machine learning et le piège de l'Accuracy

**Apports théoriques** :

- Comprendre les métriques : Accuracy vs Recall/F1-score, comprendre pourquoi un Accuracy élevé ne sert à rien avec un Recall faible,
- Générer la matrice de confusion (Faux Positif / Faux Négatifs),
- Démontrer l'impact "business" du Recall (coût d'une panne vs arrêt de production)

**Techniquement**

- Entraînement selon deux ou trois algorithmes et logging dans MLFlow (Random Forest - XGBoost - Linear Regression),
- Logger métriques / artefacts / modèles avec MLFlow

**User Story # 2.2 (compétences C1, C4)**
En tant qu'expert IA, j'ajoute une dimension "perceptive" en traitant des images dans ma solution. En tant qu'opérateur, je dois détecter au plus vite si ma machine produit des pièces défectueuses, possiblement annonciateur de panne.

Focus IAAct sur l'utilisation de l'image dans une solution d'IA.
Identifier les risques de dérives YOLO

**Techniquement**

- Préparation d'un dataset d'images (MVTec AD),
- Augmentation,
- Redimensionnement
- Normalisation

**User Story # 2.3 (compétences C4, C5)**
Laisser le système apprendre seul en apprentissage "non supervisé" : apprendre le "normal" pour prédire l'anomalie.

Focus sur l'architecture CNN auto-encoder, théorie de la reconstruction.
Focus sur les courbes de Loss (Overfitting detection)

**Techniquement**

- Préparation d'un dataset d'images (MVTec AD),
- Augmentation,
- Redimensionnement
- Normalisation

**User Story # 2.4 (compétences C2, C5)**
En tant que responsable IA, je dois m'assurer de l'impact de ma solution au regard des contraintes d'éco-conception et donc, d'optimiser le ratio Performance / Empreinte carbone

Focus sur l'éco conception / green IT
Focus sur les hyper-paramètres d'entraînement

**Techniquement**

- Optimisation des paramètres avec Optuna,
- Calcul automatique de la consommation avec CodeCarbon,
- Logging des métriques
- Choix de l'équilibre

**User Story # 2.5 (compétences C5)**
En tant que responsable IA, je dois analyser les métriques et les courbes afin de déterminer la validation finale du modèle.

Focus courbe ROC, AUC et seuils

**Techniquement**

- Analyser les artefacts MLFlow (ROC, AUC, ...) et comparer les modèles "sains" vs modèles "malades"

**User Story # 2.6 (compétences C4, C5)**
En tant qu'expert en solution IA, je dois expliquer les prédictions "tabulaires", comprendre et faire comprendre les prédictions.

Focus eXplainable AI

**Techniquement**

- Identifier les "features" qui influencent les détections,
- Identifier les zones d'anomalies sur une "heatmap",
- Fournir les métriques d'influence des décisions avec SHAP

### Sprint 3

**Objectif du sprint** : Industrialiser

Le POC est validé, il faut industrialiser la solution, mettre en place un processus de surveillance et d'automatisation.

**User Story # 3.1 (compétences C6)**
En tant qu'architecte IA, je dois mettre en place une solution d'industrialisation pour passer du POC à une solution robuste.

Focus DVC vs Git
Focus Github Actions - Gitlab CI/CD
Focus sur les hooks et secrets Github / Gitlab

**Techniquement**

- Refactorage Notebook -> Classes Python (modules),
- Mise en place d'une chaîne CI (Intégration Continue avec un flow GIT),
- Utiliser DVC (Data Version Control) pour gérer le versionning de modèles
- Intégrer les tests unitaire *pytest*

**User Story # 3.2 (compétences C7)**
En tant qu'architecte IA, je dois fournir une interface standard pour permettre à mes utilisateurs de réaliser leurs prédictions.

Focus FastAPI / Architecture REST / Conception Micro-Services
Focus Github Actions - Gitlab CI/CD

**Techniquement**

- Mettre en place un routage FastAPI,
- Produire la documentation Swagger,
- S'assurer de la bonne santé de l'API

**User Story # 3.3 (compétences C6)**
En tant qu'architecte IA, je dois fournir une solution déployable, isolée afin de faciliter le déploiement en production.

Focus Docker / Kubernetes / Docker Swarm

**Techniquement**

- Ecrire un Dockerfile pour l'API,
- Composer un docker-compose pour faciliter la mise en oeuvre,
- Créer un Makefile pour optimiser les commandes

**User Story # 3.4 (compétences C6, C7)**
En tant que spécialiste des solutions IA, je dois fournir une méthode d'orchestration : Ingest -> Predict -> Store

Focus Prefect

**Techniquement**

- Mettre en place un pipeline Prefect simple

**User Story # 3.5 (compétences C3, C8)**
En tant que spécialiste des solutions IA, je dois m'assurer d'être averti quand le modèle dérive.

Focus sur le Datadrift (Covariate Drift, Concept Drift)

**Techniquement**

- Mettre en place un rapport de Drift avec Evidently AI,
- Générer des alertes automatiques si changement de distribution dans les données d'entrées

**User Story # 3.6 (compétences C6, C8)**
En tant que fournisseur de solution d'IA je dois m'assurer de la continuité de service de l'API fournie et de la conformité de mon modèle.

Focus sur Prometheus / Grafana

**Techniquement**

- Instrumenter l'API avec Prometheus,
- Création d'un tableau de bord Grafana (temps réel, taux d'erreurs, Consommation mémoire, Drift alerts, ...)

### Sprint 4

**Objectif du sprint** : gérer le cycle de vie de la solution

La solution est déployable, déployée, robuste, supervisée. Il est nécessaire de mesurer son impact et assurer sa durée de vie.

**User Story # 4.1 (compétences C8)**
En tant qu'expert professionnel, je dois m'assurer de la robustesse de ma solution face à une montée en charge.

**Techniquement**

- Réaliser un test de simulations de requêtes massives,
- Calculer un ROI sur la base d'une simulation (pannes réelle / panne détectée / coût de la solution directs - indirects - de fonctionnement)
- Générer le rapport Carbon final

**User Story # 4.2 (compétences C9)**
En tant qu'expert professionnel, je dois anticiper le réentraînement, l'injection de nouvelles données, automatiser le processus, et valider le flux

**Techniquement**

- Préparer un scénario de "mise à jour" avec injection de nouvelles données,
- Déclencher un ré-entraînement automatique par l'intermédiaire de Prefect
- Valider le nouveau modèle

**User Story # 4.3 (compétences C9)**
En tant qu'expert professionnel, je dois fournir une documentation exhaustive, propre, lisible interprétable et pratique.

**Techniquement**

- Automatiser la production technique à l'aide d'AST, au format Markdown
- Gérer les diagrammes via "mermaid"
- Créer un script de réalisation automatisé de RELEASE_NOTES.md

## Dernier jour

Entraînement à la soutenance :

- Présenter l'application opérationnelle, conteneurisée,
- Démonstration du flow, production des artefacts, génération d'un réentraînement,
- Argumenter les choix techniques, les choix éthiques
- Série de questions / réponses

# Artefacts

- Capteurs de température : relevés sur les 6 derniers mois,
- Capteurs de pression hydraulique : relevés sur les 6 derniers mois,
- Relevés manuels d'incidents (fichiers Excel) : relevés sur la dernière année,
- Relevés manuels de production (fichiers Excel) : relevés sur la dernière année,
- Images de pièces produites sans défaut, doit permettre la détection des défauts# Compétences

# Compétences pour le titre : Concevoir et implémenter une solution IA pour l'entreprise

- C1. Identifier un jeu de données pour répondre aux besoins métiers et aux cas d’usage, en tenant compte des enjeux de pertinence et de cohérence
- C2. Identifier les risques éthiques et sociétaux à prendre en compte dans le cadre de l’exploitation de la solution d’IA pour prévenir les dérives éventuelles, en tenant compte du cadre règlementaire
- C3. Préparer les données pour renforcer leur intégrité et leur pertinence en vue du développement de la solution IA, en mobilisant les techniques de traitement adaptées et en tenant compte des attendus (besoins métiers, cas d’usage etc.) identifiés en phase de cadrage du projet
- C4. Choisir un modèle IA pour disposer d’une solution adaptée et performante par rapport aux cas d‘usage, en mesurant sa pertinence et en mobilisant une démarche scientifique
- C5. Entraîner le modèle d’IA de façon automatique et supervisée pour valider la pertinence des solutions envisagées, au regard des cas d’usage énoncés par le métier
- C6. Implémenter le modèle d’IA en intégrant les briques technologiques (moteurs, reporting, suivi des prévisions etc.) au sein de l’environnement technique choisi pour exploiter la solution
- C7. Contribuer à la conception et à l’évaluation de la proposition d’architecture cible, en identifiant les contraintes avec l’appui des acteurs pertinents, pour garantir les performances attendues
- C8. Mesurer la performance et les impacts de la solution d’IA pour maintenir son application fonctionnelle, conformément aux cas d’usage et aux enjeux identifiés
- C9. Adopter une démarche d’amélioration continue de la solution IA, pour garantir son évolution au fil du temps, dans le respect des exigences de la commande initiale et en tenant compte des évolutions des besoins utilisateurs et des données mobilisables
