# Preuve de l'image Docker — `indusense-api:m27`

Règle du module : une propriété n'est prouvée que par une commande exécutée
qui l'observe — jamais déduite du Dockerfile ou supposée parce que le build
se termine sans erreur.

| Propriété | Statut | Preuve |
|---|---|---|
| Build multi-stage | Implémenté | `docker build` réussit, étapes `build`/`runtime` séparées (voir `Dockerfile`) |
| Utilisateur non-root | Implémenté | `docker inspect --format '{{.Config.User}}'` → `appuser` ; `docker exec ... whoami` / `id` → `appuser`, `uid=10001(appuser) gid=999(appgroup)` |
| Santé | Implémenté | `GET /health` → `{"status":"ok"}` ; `GET /ready` → `{"status":"ready"}` sur le conteneur réellement démarré |
| Déterminisme du contenu | Implémenté | Deux `docker build` indépendants (`m27-repro-a`, `m27-repro-b`) → `docker inspect --format '{{json .RootFS.Layers}}'` identiques, `diff` vide |
| Taille de l'image | Implémenté | `docker inspect --format '{{.Size}}'` = 724 487 215 octets (690,9 Mo), confirmé par `docker save` (724 506 624 octets) ; seuil `scripts/check_image.py` : `MAX_MB = 750` |
| Garde-fous sécurité (module 26) | Implémenté | 401 / 422 / 413 / 429 rejoués en HTTP réel sur le conteneur (détail ci-dessous) |

## Build multi-stage

```
docker build -t indusense-api:m27 .
```
Réussit. Étape `build` (`python:3.13-slim`, installe `uv`, `uv sync --frozen --no-editable`
depuis `pyproject.toml`/`uv.lock` figés) puis étape `runtime` (repart d'une image
`python:3.13-slim` propre, ne récupère que `/app/.venv` — pas le code source,
`--no-editable` l'a rendu inutile à l'exécution — et `model.joblib`).

## Utilisateur non-root

```
docker inspect indusense-api:m27 --format '{{.Config.User}}'
→ appuser

docker exec indusense-m27 whoami
→ appuser
docker exec indusense-m27 id
→ uid=10001(appuser) gid=999(appgroup) groups=999(appgroup)
```
Le `USER appuser` déclaré dans le Dockerfile est bien celui qui tourne
réellement dans le conteneur — pas seulement celui écrit dans le fichier.

## Santé

Conteneur démarré (`docker run --env-file .env.example -p 8001:8000 ...`) :
```
GET /health  → {"status":"ok"}
GET /ready   → {"status":"ready"}
```

## Déterminisme du contenu

```
docker build -t indusense-api:m27-repro-a .
docker build -t indusense-api:m27-repro-b .
diff <(docker inspect --format '{{json .RootFS.Layers}}' indusense-api:m27-repro-a) \
     <(docker inspect --format '{{json .RootFS.Layers}}' indusense-api:m27-repro-b)
→ (rien — layers identiques)
```
Les 8 hashes de layers sont identiques entre les deux builds — le lockfile
figé (`uv sync --frozen`) produit un contenu reproductible, pas seulement des
versions de paquets identiques.

## Taille de l'image

```
docker inspect indusense-api:m27 --format '{{.Size}}'  → 724487215  (690,9 Mo)
docker save indusense-api:m27 -o image.tar ; taille du .tar → 724506624 (690,9 Mo, confirme .Size)
```

**Piège Docker Desktop rencontré** : `docker images` affiche `DISK USAGE 2.47GB`
à côté de `CONTENT SIZE 724MB` pour la même image — l'écart vient du backend
containerd qui compte, dans "Disk Usage", des layers intermédiaires de l'étape
`build` (jetables, jamais livrés) comme s'ils faisaient partie de l'image
finale. Seul `docker inspect`/`docker save` — l'export réel — donne la taille
qui compte.

**Seuil** : `scripts/check_image.py` fixe `MAX_MB = 750`, une marge réaliste
au-dessus des 690,9 Mo mesurés. Ce projet garde `jupyter`, `matplotlib`,
`mlflow`, `optuna`, `huggingface-hub`, `codecarbon`, `dvc`... dans les
dépendances de base de `pyproject.toml` (pas dans un groupe séparé de
l'API) : ils servent aux notebooks et à l'entraînement, présents dans
`uv.lock` et donc dans l'image même si `indusense.api.main` ne les importe
jamais lui-même. Les isoler dans un groupe "api" à part réduirait
sensiblement la taille de l'image, mais casserait l'hypothèse que ces outils
sont disponibles par défaut pour qui clone le dépôt — hors scope de ce
correctif.

## Garde-fous sécurité (module 26) rejoués en HTTP réel sur le conteneur

| Cas | Code observé |
|---|---|
| Sans `X-API-Key` | `401` |
| `features` vide | `422` |
| Corps de 84 014 octets (> 64 Ko) | `413` |
| Rafale de requêtes authentifiées | `429` (apparu à la 58ᵉ requête de la rafale de cette séance, pas à la 61ᵉ) |

**Sur le seuil du rate limit** : la fenêtre glissante (60 s, en mémoire, par
IP) ne repart pas de zéro entre deux séries de tests rapprochées dans la
même séance — les appels de vérification précédents (401, 200 de contrôle)
sur ce même conteneur comptaient déjà dans la fenêtre au moment de lancer la
rafale. Ce n'est pas une anomalie : c'est la définition même d'un compteur
en fenêtre glissante, pas remis à zéro par test. Le comportement observé en
isolation (`tests/test_security.py::test_rate_limit_burst_returns_429_over_http`,
conteneur/process frais) reste : les 60 premiers appels passent, le 61ᵉ échoue.
