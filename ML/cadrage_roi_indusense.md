# Cadrage du retour sur investissement — InduSense 4.0

> Complète [architecture_deploiement_indusense.md](architecture_deploiement_indusense.md) (coûts de déploiement par tranche, coût de fonctionnement d'un arrêt machine). Ici : comment on structure, formalise et négocie le ROI avec le client.

## 1. Objectif du cadrage

Sans ROI démontré, pas de projet : le client n'engage pas sans preuve que le gain dépasse le coût. Le cadrage ROI doit donc être posé **dès le cadrage projet**, pas après le déploiement, et formalisé par écrit (hypothèses, échéance, méthode de vérification) — c'est la pièce qui sécurise la négociation contractuelle autant que la vente.

**Règle d'or** : à l'instant *t* où l'on travaille sur le projet, la solution mise en place doit permettre un ROI — sinon il faut revoir le scope, pas forcer le calcul.

## 2. Formule et logique du calcul

```
ROI cumulé (année N) =
      Σ (pannes évitées estimées × coût unitaire d'un arrêt évité)
    − (coûts d'investissement + coûts de fonctionnement cumulés jusqu'à N)
```

- **Coût unitaire d'un arrêt évité** : cf. décomposition du coût de fonctionnement (immobilisation brute + réparation + redémarrage + opérateurs non productifs). Référence de départ : **9 k€ pour un arrêt de 6h** — chiffre à faire valider par le Métier, jamais estimé par l'équipe IA.
- **Pannes évitées estimées** : hypothèse statistique (taux de détection × taux de pannes historique), pas un fait acquis — à documenter comme hypothèse, avec sa marge d'incertitude.
- Toujours présenter le résultat en **fourchette basse / haute**, jamais un chiffre unique.

## 3. Coûts d'investissement (ce qui entre dans le calcul)

| Poste | Nature | Référence |
|---|---|---|
| Développement (déjà engagé) | Coût sunk — modèle, pipeline, drift monitoring | Sprint 1-3 InduSense |
| Déploiement par tranche | T1 pilote (10-15 k€), T2 généralisation (+3-6 k€), T3 multi-site (variable) | cf. architecture_deploiement_indusense.md §3 |
| Fonctionnement (run) | Infra edge, supervision centrale, astreinte, ré-entraînement | cf. architecture_deploiement_indusense.md §4 |
| Ressource de suivi de déploiement | Côté éditeur (0,5 ETP indicatif) **et** côté client (intégration, coordination interne) | Souvent oublié côté client — à faire chiffrer par lui |

## 4. Coûts évités / gains — collecte de l'information Métier

Le problème principal : **l'information est disséminée chez le client, aucune personne ne détient tout**. C'est au porteur de projet de l'agréger.

| Type de coût à chiffrer | Détenteur côté client |
|---|---|
| Coût horaire d'immobilisation machine | Production / Exploitation |
| Coût de réparation par type de panne (abaques) | Maintenance |
| Coût des opérateurs non productifs | RH / Production |
| Pénalités de retard imposées par le client final | Commercial / Contrats |
| Capital confiance client (chiffrable si pannes récurrentes) | Direction commerciale |
| Impact assurantiel | Finance / Assurance |

→ Prévoir des entretiens dédiés avec chaque interlocuteur Métier (Métier pur, Finance, Assurance) ; ne pas se limiter au sponsor projet.

## 5. Ce qui rend le calcul fragile — à documenter comme hypothèse

- Le prédictif est **statistique par nature** : un événement exceptionnel qui pollue la base historique (nouvelle machine, changement de process, panne rare non représentée) invalide la projection — sans que la solution soit en cause.
- Le nombre de pannes évitées est une **estimation**, pas un engagement de résultat — à formuler comme hypothèse contractuelle explicite, avec la période d'observation retenue pour la valider (ex. : 12 mois glissants).
- Écart estimation/réel systématique sur les coûts de déploiement (cf. architecture) → impacte directement le dénominateur du ROI.

## 6. Volet contractuel

- Le ROI promis **engage juridiquement** — cadrer les hypothèses de calcul (méthode, période de mesure, périmètre machines) avant signature, avec l'appui des juristes.
- Prévoir dans le contrat une clause de **requalification** : si les objectifs ne sont pas atteints en fin de période probatoire (T1), il existe des leviers d'ajustement (scope, seuils, modèle) — mais l'absence de ROI reste rédhibitoire pour la poursuite.
- Séparer clairement dans le contrat : engagement de moyens (déploiement, supervision, ré-entraînement) vs. hypothèse de gain (jamais un engagement de résultat ferme sur le prédictif).

## 7. Projection pluriannuelle — l'argument de vente

Le ROI ne se lit pas en instantané mais en trajectoire, à partir des observations du pilote (T1) :

| Horizon | Ce qu'on présente au client |
|---|---|
| Année 1 | Coût réel engagé (dev + déploiement + run) vs. pertes évitées observées sur le périmètre pilote/généralisé |
| Année 2 | Effet de la montée en charge (mutualisation edge, coûts marginaux plus faibles par machine/site) |
| Année 3 | Effet cumulé + anticipation (obsolescence, tendance des pannes) à partir de l'historique constitué |

Message client : *« Voilà ce que ça vous a coûté, voilà ce que ça vous a évité de perdre en année 1, 2, 3 »* — jamais une promesse abstraite, toujours adossée aux observations factuelles du déploiement progressif.

## 8. Livrable attendu du cadrage ROI

- [ ] Formule de calcul figée et validée avec le client (méthode, période de mesure, périmètre)
- [ ] Coûts d'investissement chiffrés par tranche (cf. §3)
- [ ] Coûts évités chiffrés et validés par le Métier, pas par l'équipe IA (cf. §4)
- [ ] Hypothèses et incertitudes explicitées (cf. §5)
- [ ] Fourchette basse/haute du ROI, pas un chiffre unique
- [ ] Clauses contractuelles associées validées avec les juristes (cf. §6)
- [ ] Trajectoire pluriannuelle de présentation (cf. §7)
