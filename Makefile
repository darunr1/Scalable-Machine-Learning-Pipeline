# ============================================================================
# Scalable ML Pipeline — Makefile
# ============================================================================
# Run `make help` to see all available commands.

.PHONY: help install ingest train api dashboard test lint clean all

PYTHON ?= python
PIP ?= pip
PYTEST ?= $(PYTHON) -m pytest

# Default dates for ingestion
START_DATE ?= $(shell $(PYTHON) -c "from datetime import datetime, timedelta; print((datetime.now() - timedelta(days=30)).strftime('%Y-%m-%d'))")
END_DATE ?= $(shell $(PYTHON) -c "from datetime import datetime; print(datetime.now().strftime('%Y-%m-%d'))")

help: ## Show this help message
	@echo ""
	@echo "  Scalable ML Pipeline — Available Commands"
	@echo "  =========================================="
	@echo ""
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-15s\033[0m %s\n", $$1, $$2}'
	@echo ""

install: ## Install all dependencies
	$(PIP) install -r requirements.txt

install-dev: ## Install with dev dependencies (lint, format)
	$(PIP) install -r requirements.txt
	$(PIP) install ruff black isort

ingest: ## Ingest 30 days of weather data from Open-Meteo
	$(PYTHON) -m ingestion.ingest --source weather_api --start-date $(START_DATE) --end-date $(END_DATE)

train: ## Train a model with hyperparameter tuning
	$(PYTHON) -m training.train

retrain: ## Force a retraining cycle
	$(PYTHON) -m training.retrain --force

api: ## Start the FastAPI inference server
	uvicorn api.app:app --reload --host 0.0.0.0 --port 8000

dashboard: ## Launch the Streamlit monitoring dashboard
	streamlit run dashboard/app.py

scheduler: ## Start the automation scheduler
	$(PYTHON) -m scheduler.runner

test: ## Run all tests with verbose output
	$(PYTEST) tests/ -v

test-fast: ## Run tests excluding integration tests
	$(PYTEST) tests/ -v -k "not integration"

lint: ## Run code linting with ruff
	$(PYTHON) -m ruff check .

format: ## Format code with black and isort
	$(PYTHON) -m black .
	$(PYTHON) -m isort .

clean: ## Remove generated artifacts (data, models, logs, etc.)
	rm -rf data/ models/*.pkl models/metadata.json
	rm -rf features/fitted_pipeline.pkl
	rm -rf monitoring/prediction_log.jsonl monitoring/reports/
	rm -rf drift/baseline_stats.json drift/reports/
	rm -rf experiments/
	rm -rf logs/
	rm -rf __pycache__ */__pycache__ */*/__pycache__
	@echo "Cleaned all generated artifacts ✓"

all: ingest train test ## Run full pipeline: ingest → train → test
	@echo ""
	@echo "  Full pipeline complete ✓"
	@echo ""
