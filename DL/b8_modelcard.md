# B8 — Model card : documenter un modèle

> Couvre le module Documentation / packaging du parcours.
> On suit le **template officiel Hugging Face** ([modelcard_template.md](https://github.com/huggingface/huggingface_hub/blob/main/src/huggingface_hub/templates/modelcard_template.md)).

## Scénario

**Product Owner** : Le modèle est validé, mais personne d'autre que toi ne sait *à quoi il sert, ce qu'il vaut, ni ce qu'il ne faut pas lui faire faire*. Il nous faut une fiche de référence, lisible par le client comme par un auditeur.

**Developer** : C'est le rôle de la **model card**. Plutôt que d'inventer un format, on remplit le **template standard de Hugging Face** : c'est celui que tout le monde reconnaît, et il couvre déjà usage, données, métriques, limites et impact.

## Objectifs pédagogiques

À l'issue du TP, vous saurez :

1. Remplir une model card au **format standard Hugging Face**.
2. Distinguer **usage direct**, **usage en aval** et **usage hors-périmètre**.
3. Reporter les **métriques** d'évaluation et documenter **biais, risques et limites**.
4. Renseigner l'**impact environnemental** (réutiliser la mesure CodeCarbon).

## Le principe

Une model card répond, sur une page, à : *que fait ce modèle, sur quelles données, avec quelle performance, dans quelles limites, et comment l'utiliser ?* Le template HF est en **Markdown** : on remplace les champs entre `{{ }}` ; les sections marquées **[optionnel]** peuvent être laissées vides pour une v1.

## Étapes du TP

### Étape 1 — Rassembler les éléments
- Repartir du **modèle validé** : objectif, données d'entraînement, métriques de test (et seuil), procédure d'entraînement, mesure CodeCarbon.

### Étape 2 — Remplir le template Hugging Face
Compléter la structure ci-dessous (sections obligatoires en priorité) :


> Astuce : `huggingface_hub` génère ce squelette automatiquement —
> `from huggingface_hub import ModelCard; ModelCard.from_template(card_data=...)`.

### Étape 3 — Usage, biais, risques, limites
- Renseigner **Direct Use** et surtout **Out-of-Scope Use** (ex. aucune décision automatique).
- Décrire **Bias, Risks, and Limitations** honnêtement (ex. signal faible, faux négatifs) et les **Recommendations**.
- 🔎 **Point de réflexion** : un lecteur non technique comprend-il *quand faire confiance au modèle et quand se méfier* ?

### Étape 4 — Évaluation, impact, contact
- Remplir **Evaluation** (métriques de test + seuil retenu) et **Environmental Impact** à partir de votre mesure CodeCarbon.
- Renseigner la **version**, les **auteurs** et le **contact**.

## Livrables attendus

1. Une **model card** au format Hugging Face, sections obligatoires renseignées.
2. Les trois usages explicites : **Direct / Downstream / Out-of-Scope**.
3. Les sections **Bias, Risks, and Limitations** et **Evaluation** complétées.

## Ressources

- [Template officiel Hugging Face](https://github.com/huggingface/huggingface_hub/blob/main/src/huggingface_hub/templates/modelcard_template.md)
- [Hugging Face — Model Cards (guide)](https://huggingface.co/docs/hub/model-cards)
- [Model Cards for Model Reporting (Mitchell et al.)](https://arxiv.org/abs/1810.03993)
- [ML CO2 Impact calculator (Lacoste et al., 2019)](https://mlco2.github.io/impact)
