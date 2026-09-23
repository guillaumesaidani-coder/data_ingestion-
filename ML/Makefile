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
