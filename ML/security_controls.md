# Registre des contrôles de sécurité — `/predict-tabular`

Règle du module : un contrôle « documenté » ou « prévu » n'est pas un contrôle
implémenté. Seul un code HTTP observé par un test compte comme preuve.
Exactement 4 lignes *Implémenté*, 1 ligne *Planifié v0* — jamais l'inverse.

| Contrôle | Statut | Preuve | Risque résiduel | Action suivante |
|---|---|---|---|---|
| Auth | Implémenté | `401` sans clé / clé invalide (`tests/test_api.py::test_predict_tabular_requires_api_key`, `test_predict_tabular_rejects_wrong_api_key`) | Clé statique unique pour tout le monde, pas de rotation ni de rôles | Clés par client + rotation |
| Validation | Implémenté | `422` payload vide/malformé (`test_predict_tabular_rejects_empty_features`, `test_predict_tabular_rejects_malformed_payload`) | Valide la forme (dict de floats), pas les plages physiques (ex. température aberrante) | Bornes métier par feature |
| Rate limit | Implémenté | `429` au 61ᵉ appel / 60 req/min/IP — preuve directe (`test_rate_limit_function_blocks_after_limit`) **et** rafale HTTP réelle de 70 requêtes (`test_rate_limit_burst_returns_429_over_http`) | Basé sur l'IP (contournable via IP rotative) ; état en mémoire, perdu au redémarrage, non partagé entre instances | Externaliser l'état (ex. Redis) si déploiement multi-instance |
| Taille payload | Implémenté | `413` si Content-Length > 64 Ko (`test_predict_tabular_rejects_oversized_payload`), `400` si Content-Length illisible (`test_predict_tabular_rejects_unreadable_content_length`) | Ne protège pas un corps envoyé sans Content-Length (chunked) | Exiger Content-Length ou limiter la lecture en flux |
| Audit logging | **Planifié v0** | Aucun événement structuré — `X-Request-ID` corrèle des lignes de log entre elles, il ne constitue ni un événement d'audit ni sa preuve. Ce qui est déjà logué (method/path/status/request_id) ne fuite pas de secret (`test_no_secret_or_payload_leak_in_logs`), mais ça ne prouve pas qu'un audit existe | Impossible de reconstituer qui a appelé quoi/quand en cas d'incident | Événement structuré et durable : horodatage, request_id, route, code retour, identité appelante |

## Ce que « Planifié v0 » veut dire ici

Écrire *Audit logging : Implémenté* parce que `X-Request-ID` existe serait faux.
Le middleware sert à retrouver, dans des logs déjà existants, les lignes qui
correspondent à une même requête — il ne crée aucun événement d'audit en soi.
Le module 26 s'arrête volontairement avant cette dernière étape.
