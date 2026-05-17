.PHONY: help install lint format lint-check test run-eda run-baseline run-experiments docker-build docker-up docker-down

help:
	@echo "Доступные команды:"
	@echo "  make install        — Установить зависимости"
	@echo "  make lint           — Запустить линтер (ruff) и проверить стиль"
	@echo "  make format         — Автоматически отформатировать код (ruff format)"
	@echo "  make lint-check     — Только проверка без исправлений (для CI)"
	@echo "  make test           — Запустить тесты (pytest)"
	@echo "  make run-eda        — Выполнить EDA ноутбук через papermill"
	@echo "  make run-baseline   — Выполнить Baseline ноутбук"
	@echo "  make run-experiments— Выполнить ноутбук с экспериментами"
	@echo "  make docker-build   — Собрать Docker-образ"
	@echo "  make docker-up      — Запустить Jupyter через docker-compose"
	@echo "  make docker-down    — Остановить контейнеры"

# ── Dependencies ──────────────────────────────────────────────────────────────
install:
	pip install -r requirements.txt

# ── Linting & Formatting ──────────────────────────────────────────────────────
lint:
	ruff check src/ tests/ --fix
	ruff format src/ tests/
	flake8 src/ tests/ --max-line-length=120 --ignore=E203,W503

format:
	ruff format src/ tests/ notebooks/

lint-check:
	ruff check src/ tests/
	flake8 src/ tests/ --max-line-length=120 --ignore=E203,W503

# ── Tests ─────────────────────────────────────────────────────────────────────
test:
	pytest tests/ -v --tb=short

# ── Notebooks (via papermill) ─────────────────────────────────────────────────
run-data-loading:
	papermill notebooks/00_data_loading.ipynb notebooks/00_data_loading_output.ipynb

run-eda:
	papermill notebooks/01_eda.ipynb notebooks/01_eda_output.ipynb

run-baseline:
	papermill notebooks/02_baseline.ipynb notebooks/02_baseline_output.ipynb

run-experiments:
	papermill notebooks/03_experiments.ipynb notebooks/03_experiments_output.ipynb

run-all: run-data-loading run-eda run-baseline run-experiments

# ── Docker ────────────────────────────────────────────────────────────────────
docker-build:
	docker-compose build

docker-up:
	docker-compose up -d jupyter
	@echo "Jupyter доступен по адресу: http://localhost:8888"

docker-down:
	docker-compose down

docker-test:
	docker-compose --profile testing run --rm tests
