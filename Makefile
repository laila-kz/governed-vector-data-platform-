.PHONY: install test coverage qdrant-up qdrant-down

install:
	python -m pip install -r requirements.txt

test:
	python -m pytest -q

coverage:
	python -m pytest -q --cov=contracts --cov=ingestion --cov=chunking --cov=embedding --cov-report=term-missing --cov-fail-under=90

qdrant-up:
	docker compose up -d qdrant

qdrant-down:
	docker compose down
