.PHONY: install test qdrant-up qdrant-down

install:
	python -m pip install -r requirements.txt

test:
	python -m pytest -q

qdrant-up:
	docker compose up -d qdrant

qdrant-down:
	docker compose down
