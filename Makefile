SHELL := /bin/bash
VENV := .venv
PY := $(VENV)/bin/python
UV_CACHE := .uv-cache

.PHONY: help install lock run ingest test lint eval build-naive build up down logs clean

help:
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  %-14s %s\n", $$1, $$2}'

install: ## Create the venv and install dev dependencies
	UV_CACHE_DIR=$(UV_CACHE) uv venv --python 3.13 $(VENV)
	UV_CACHE_DIR=$(UV_CACHE) uv pip install --python $(PY) -r requirements-dev.txt

lock: ## Re-lock requirements.txt from requirements*.in
	UV_CACHE_DIR=$(UV_CACHE) uv pip compile requirements.in --python 3.13 -o requirements.txt
	UV_CACHE_DIR=$(UV_CACHE) uv pip compile requirements-dev.in --python 3.13 -o requirements-dev.txt

run: ## Run the API locally with autoreload
	$(PY) -m uvicorn app.main:app --reload --port 8000

ingest: ## Ingest data/policies into the local index
	$(PY) -m scripts.ingest

test: ## Run the test suite
	$(PY) -m pytest

lint: ## Lint the codebase
	$(PY) -m ruff check app tests scripts

eval: ## Run the 10-question evaluation against localhost:8000
	$(PY) -m scripts.run_eval

build-naive: ## Build the deliberately naive image (size comparison)
	docker build -f docker/Dockerfile.naive -t policy-rag:naive .

build: ## Build the production image
	docker build -t policy-rag:latest .

up: ## Start app + Qdrant with docker compose
	docker compose up --build -d

down: ## Stop compose services (keeps named volumes)
	docker compose down

logs: ## Tail compose logs
	docker compose logs -f

clean: ## Remove local virtualenv, index and caches
	rm -rf $(VENV) $(UV_CACHE) data/index data/uploads .pytest_cache .ruff_cache
