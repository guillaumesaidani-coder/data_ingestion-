# Plan d'action — boucle HITL et arbitrage champion/challenger

Adapte `feuille_de_route_mlops_indusense.md` (générique, vocabulaire
« vibrations », modèle non précisé) à notre système réel (b11-gkf,
`gold_machine_hourly_feature`, `predictions`, `predict_flow.py`,
Prometheus/Grafana) et le rend cohérent avec `gap_analysis_dat_v1.md`.
Objectif demandé : simuler des retours terrain après un premier
déploiement et améliorer le modèle en fonction de ces retours.

## Pourquoi ce plan referme aussi des écarts du gap analysis

La feuille de route et le DAT parlent de la même chose avec un
vocabulaire différent — les construire ensemble évite de dupliquer le
travail :

| Besoin HITL (feuille de route) | Écart DAT correspondant (gap analysis) |
|---|---|
| `model_version` sur chaque prédiction | §3/§8 — aucun `release_id`/version tracée nulle part |
| Table de review + arbitrage champion/challenger | §3 — pas de `model_promotions` (CANDIDATE→APPROVED) |
| Mode fantôme avant bascule | §3/§4 — pas de bascule contrôlée ni de rollback |
| Quotas Prefect avant réentraînement | §4 — `ensure_model(retrain=False)` existe déjà (M29/M30), juste jamais piloté par un vrai critère |

Construire la boucle HITL, c'est donc construire une version *plus
simple mais réelle* de la gouvernance de release qui manquait — pas un
chantier à côté.

## Différence de contexte à assumer (pas à cacher)

La feuille de route suppose que la vérité terrain vient **uniquement**
d'un humain qui valide après coup. Chez nous, `label_failure_next_24h`
est déjà un label réel connu (incidents réels, horizon 24h) — on n'a
pas besoin d'attendre un technicien pour savoir si une fenêtre passée
était une vraie panne. Ça ne rend pas l'exercice inutile : l'objectif
devient de **construire le mécanisme** (journal, UI, arbitrage, shadow
mode) et de le **prouver** en rejouant notre historique réel comme si
chaque ligne venait d'être validée par un technicien — même méthode que
les scénarios de dérive (modules 31-34) : données réelles, mécanisme
réel, juste le calendrier qui est simulé plutôt qu'attendu.

Deux sources de vérité terrain à construire, pas une seule :
1. **Simulateur** — rejoue l'historique réel et déclare la vérité
   connue (`label_failure_next_24h`) comme si un technicien venait de
   la confirmer. Fait tourner la boucle immédiatement, sans attendre.
2. **Streamlit réel** — pour les cas qu'un label automatique ne peut
   pas trancher (« capteur défaillant », « maintenance préventive » =
   vérité ambiguë même avec le label). C'est elle qui a une vraie valeur
   pédagogique et pratique, le simulateur n'est qu'un accélérateur de
   preuve.

## Phasage proposé

### M35 — Journal de prédiction versionné
**Construit :** étendre `predictions` (pas une table parallèle — une
seule source de vérité, leçon du module 30) avec `model_version`
(hash du bundle `model.joblib`, calculé une fois au chargement),
`features_payload` (JSONB, snapshot des ~70 features au moment du
scoring — aujourd'hui non stocké, seul `failure_proba_24h` l'est),
`review_status`, `ground_truth`, `reviewer_comment`, `reviewed_at`.
Migration Alembic (le repo en a déjà 8, même convention).
**Preuve :** `predict_flow.py` rejoué → nouvelles colonnes peuplées,
`model_version` stable entre deux runs sans réentraînement, migration
appliquée deux fois sans erreur (comme la matrice de recette du DAT le
demande pour les autres migrations).
**Ferme :** le point P0 du gap analysis (aucune version de modèle
tracée nulle part).

### M36 — Simulateur de retour terrain + Streamlit réel
**Construit :**
- `scripts/simulate_feedback.py` : rejoue une plage de fenêtres déjà
  scorées, déclare `ground_truth` = label réel connu, `review_status`
  cohérent (panne confirmée / fausse alerte), avec le biais de
  sélection **explicitement introduit puis mesuré** (on ne « valide »
  que les alertes, comme un vrai technicien le ferait) — puis on montre
  ce que ça rate sans le bouton « incident non prédit ».
- Streamlit minimal (une page) sur nos vraies features (température,
  pression, vibration = `rotation_mean_1h`, pas les libellés génériques
  du template) pour les cas ambigus, avec le bouton « déclarer un
  incident non prédit ».
**Preuve :** N fenêtres réelles rejouées, distribution des
`review_status` mesurée, biais de sélection démontré par comparaison
avec/sans le bouton incident-non-prédit (pas juste affirmé).

### M37 — Arbitrage champion (b11 actuel) vs challenger
**Construit :** `scripts/arbitrate_challenger.py` — reprend la matrice
gain/stabilité/régression/angle mort de la feuille de route, sur des
données réelles validées (issues de M36). Le challenger n'est pas un
réentraînement à l'identique (qui ne prouverait rien de neuf sur notre
modèle déjà fort, rappel 0,91) : il est entraîné sur le Gold **enrichi
des corrections du retour terrain** (labels corrigés sur les
« capteur défaillant », incidents non prédits ajoutés) — c'est
exactement le mécanisme qui doit démontrer une amélioration réelle,
pas fabriquée.
**Preuve :** décision `ACCEPTATION_DIRECTE` / `SOUS_DEROGATION` /
`REJET` mesurée sur un vrai jeu de retours simulés, jamais sur un
pourcentage moyen seul — reproduit la règle d'or de la feuille de route.
**Ferme :** l'équivalent simplifié de `model_promotions`
(CANDIDATE→APPROVED) du DAT, absent aujourd'hui (gap analysis §3).

### M38 — Prefect : quotas avant réentraînement
**Construit :** nouvelle task `verifier_conditions_reentrainement`
dans `predict_flow.py` (ou un flow dédié `retrain_flow.py`, à trancher
selon si on veut garder un flow unique ou séparer — voir gap analysis
§4, qui recommande déjà deux deployments distincts) : quota de
validations, minimum de pannes confirmées, contrôle Pandera sur les
plages physiques des capteurs, appel à `evaluate_drift` existant comme
4e feu vert. Ce flow pilote `ensure_model(..., retrain=True)`, qui
existe déjà mais n'est aujourd'hui jamais déclenché par un vrai critère.
**Preuve :** cycle suspendu si quotas non atteints (log explicite),
cycle déclenché pour de vrai une fois les seuils franchis dans les
données simulées de M36.

### M39 — Mode fantôme avant bascule
**Construit :** le challenger accepté (M37) score en silence à côté du
champion pendant que le champion continue de servir — deux colonnes
`model_version` dans `predictions` (déjà porté par M35) suffisent à
distinguer les deux séries sans nouvelle table. Bascule = changer le
pointeur `model.joblib` (aujourd'hui : un seul fichier) une fois la
fenêtre d'observation confirmée.
**Preuve :** N jours de scoring silencieux réels (accéléré : rejoué sur
historique), comparaison champion/challenger sur la même période,
bascule effective si confirmée.
**Ferme :** une version légère de la bascule contrôlée `release_activations`
du DAT (gap analysis §3/§4), sans le manifeste signé complet — assumé
et documenté comme tel, pas caché.

## Ce qui reste hors périmètre (assumé, pas oublié)

- Signature/manifeste de release, SBOM, `images.lock.yml` — gap
  analysis §3/§6, hors sujet de cette boucle HITL, à traiter séparément
  si la conformité DAT complète devient un objectif.
- Deux deployments Prefect séparés train/score avec work pools dédiés
  — la boucle HITL peut se construire sur le flow actuel ; séparer les
  deployments reste une décision indépendante (gap analysis §4).
- Alertmanager — une alerte de dérive qui déclenche un cycle de
  réentraînement (M38) n'a pas besoin d'un routage humain pour cette
  démonstration.

## Prochaine étape

M35 est le prérequis de tout le reste (sans `model_version` ni
`features_payload`, rien n'est comparable ni rejouable). Je peux
commencer par là dès votre feu vert — migration Alembic + modification
de `predict_flow.py`/`predictions_store.py`, prouvée par une exécution
réelle du flow.
