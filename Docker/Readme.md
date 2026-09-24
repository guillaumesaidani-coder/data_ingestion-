> **Remplacé par la stack unique `ML/compose.yaml`** (base Postgres initialisée
> automatiquement + pgAdmin sur :8082). Ne pas lancer les deux en même temps :
> elles publient toutes deux les ports 5432 et 8082. Les données de cette
> ancienne stack restent dans le volume `pg-db-data`.

# Docker database with GUI

## Customize Composition
Copy .env.sample to .env and edit it at your convenience

## Run Container Database
### 1 - First time
docker-compose up -d

docker-compose -p projectname up -d

docker-compose --env-file .env.test up -d

docker-compose -p projectname --env-file .env.test up -d
### 2 - Start/Stop
docker-compose start

docker-compose start db

docker-compose start gui

