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
	@echo "Voir image_proof.md pour le registre complet des preuves."

# ---- Stack unique (R1) : compose.yaml ------------------------------------

.PHONY: stack-up stack-down stack-etl stack-migrate seed-dump

stack-up:        ## Démarre la stack (base initialisée au 1er démarrage)
	docker compose up -d --build

stack-down:      ## Arrête la stack, en gardant les données
	docker compose down

stack-etl:       ## Reconstruit Silver + Gold dans la base de la stack
	docker compose --profile jobs run --rm etl

stack-migrate:   ## Applique les migrations alembic
	docker compose --profile jobs run --rm migrate

seed-dump:       ## Régénère docker/initdb depuis une base peuplée (puis dvc add)
	uv run --frozen python scripts/build_seed_dump.py --container ml-db-1
