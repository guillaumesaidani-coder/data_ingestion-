IMAGE ?= indusense-api:m27
CONTAINER ?= indusense-m27
PORT ?= 8000

.PHONY: build run stop logs check-user check-health check-size check-determinism proof

build:
	docker build -t $(IMAGE) .

run:
	docker run --rm -d --name $(CONTAINER) -p $(PORT):8000 --env-file .env $(IMAGE)

stop:
	docker rm -f $(CONTAINER)

logs:
	docker logs $(CONTAINER)

check-user:
	docker inspect $(IMAGE) --format '{{.Config.User}}'
	docker exec $(CONTAINER) whoami

check-health:
	curl -s http://localhost:$(PORT)/health
	curl -s http://localhost:$(PORT)/ready

check-size:
	uv run python scripts/check_image.py $(IMAGE)

check-determinism:
	docker build -t $(IMAGE)-repro-a .
	docker build -t $(IMAGE)-repro-b .
	diff \
		<(docker inspect --format '{{json .RootFS.Layers}}' $(IMAGE)-repro-a) \
		<(docker inspect --format '{{json .RootFS.Layers}}' $(IMAGE)-repro-b) \
		&& echo "RootFS.Layers identiques"

proof: build check-user check-size
	@echo "Voir docs/07_preuves/image_proof.md pour le registre complet des preuves."

# ---- Stack unique (R1) : compose.yaml ------------------------------------

.PHONY: stack-up stack-down stack-reset stack-etl stack-predict stack-migrate stack-revision seed-dump seed-check

# Commit du code embarqué dans l'image (build arg lu par compose.yaml) ;
# « -dirty » si l'arbre de travail a des modifications non commitées.
export GIT_SHA := $(shell git describe --always --dirty 2>/dev/null)

# Sous Windows, Python écrit dans la console en cp1252 par défaut : les
# accents des scripts s'affichaient mal (« Seed � jour »).
export PYTHONUTF8 := 1

stack-up:        ## Démarre la stack (base initialisée au 1er démarrage)
	docker compose up -d --build

stack-down:      ## Arrête la stack, en gardant les données
	docker compose down

# Pas de --build ici : compose.yaml (pull_policy: build) reconstruit
# l'image à chaque lancement, quelques secondes via le cache si rien n'a
# changé. Un job ne tourne jamais sur l'ancien code, avec ou sans make.
stack-etl:       ## Reconstruit Silver + Gold dans la base de la stack
	docker compose --profile jobs run --rm etl

stack-predict:   ## Score les dernières fenêtres et les enregistre en base
	docker compose --profile jobs run --rm predict

stack-migrate:   ## Applique les migrations alembic
	docker compose --profile jobs run --rm migrate

stack-revision:  ## Compare le commit de l'image indusense-api:local au dépôt
	@echo "depot : $(GIT_SHA)"
	@echo "image : $$(docker inspect --format '{{index .Config.Labels "org.opencontainers.image.revision"}}' indusense-api:local)"

stack-reset: seed-check  ## Repart du seed (down -v), refusé si des verdicts n'y sont pas
	docker compose down -v

seed-dump:       ## Fige ml-db-1 dans docker/initdb (puis dvc add)
	uv run --frozen python scripts/build_seed_dump.py --container ml-db-1

seed-check:      ## Échoue si ml-db-1 contient des verdicts absents du seed
	uv run --frozen python scripts/build_seed_dump.py --container ml-db-1 --check

# ---- Modèle servi : certification puis model card -------------------------

.PHONY: model-card

model-card:      ## Après tout changement de model.joblib : certifie, puis régénère la card
	uv run --frozen python scripts/certify_model.py
	uv run --frozen python generate_model_card.py
