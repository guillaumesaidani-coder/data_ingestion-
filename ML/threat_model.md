# Modèle de menaces — `/predict-tabular` (STRIDE)

Pipeline réel, dans l'ordre exécuté (`src/indusense/api/main.py`) :

```
requête → limit_body_size (413/400) → add_request_id → require_api_key (401)
        → rate_limit_dependency (429) → validation Pydantic (422) → modèle chargé ? (503)
        → prédiction
```

Pour chaque famille STRIDE : un exemple concret sur `/predict-tabular`, et le
contrôle qui y répond aujourd'hui — ou l'absence assumée.

## Spoofing (usurpation)
**Exemple** : un client sans autorisation se fait passer pour un consommateur légitime de l'API.
**Contrôle** : clé statique `X-API-Key` (`require_api_key`) — `401` si absente ou invalide.
**Limite assumée** : une seule clé pour tous les clients, pas d'identité par appelant (cf. `security_controls.md`).

## Tampering (altération)
**Exemple** : un payload malformé, un champ du mauvais type, un dict de features vide envoyé pour forcer un comportement inattendu du modèle.
**Contrôle** : validation stricte du schéma par Pydantic (`PredictRequest`) — `422` si le payload ne correspond pas, ou si `features` est vide.

## Repudiation (déni d'action)
**Exemple** : un incident survient (prédiction erronée exploitée en aval) et personne ne peut prouver qui a appelé l'API, quand, avec quelles données.
**Contrôle** : **aucun**, assumé. `X-Request-ID` permet de corréler des lignes de log entre elles si elles existent, mais ne constitue pas un événement d'audit structuré et durable (horodatage, identité, route, code, en un seul endroit fiable). Statut : **Planifié v0**.

## Information disclosure (fuite)
**Exemple** : la clé API ou le contenu du payload (potentiellement sensible : identifiants machine, mesures) se retrouve en clair dans les logs applicatifs.
**Contrôle** : le seul logging actuel (method/path/status/request_id, dans `add_request_id`) ne loggue jamais les en-têtes ni le corps de la requête — vérifié par `test_no_secret_or_payload_leak_in_logs` avec une clé et un marqueur de payload volontairement identifiables.

## Denial of service
**Exemple** : un client (volontaire ou bogué) sature l'API d'appels, ou envoie un payload volumineux pour consommer de la mémoire/CPU.
**Contrôles** :
- Rate limit : 60 requêtes/minute/IP, `429` au-delà (`rate_limit_dependency`).
- Taille de payload : `413` au-delà de 64 Ko (`limit_body_size`), `400` si l'en-tête `Content-Length` est illisible.

## Elevation of privilege
**Exemple** : accéder à `/predict-tabular` sans posséder de clé API valide, ou contourner l'authentification via une route non protégée.
**Contrôle** : `require_api_key` est une dépendance explicite sur `/predict-tabular`. `/health` reste volontairement ouvert (`/ready` aussi) : ce sont des sondes de liveness/readiness pour un orchestrateur, jamais des routes métier — les protéger casserait le monitoring sans réduire le risque réel.

## Bilan
4 familles sur 6 ont une preuve automatisée (test + code HTTP observé) :
Spoofing, Tampering, Information disclosure, Denial of service (+ Elevation of
privilege via la même dépendance d'auth). Repudiation reste ouverte,
explicitement, en `Planifié v0` — voir `security_controls.md`.
