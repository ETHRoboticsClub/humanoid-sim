set shell := ["bash", "-uc"]

test:
    uv run --extra dev pytest

lint:
    uv run --extra dev ruff check src tests scripts/install-uv.py

build:
    docker compose build sim

smoke:
    docker compose run --rm sim

viewer:
    docker compose --profile viewer run --rm --service-ports viewer
