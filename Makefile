.PHONY: install install-dev run train test lint format requirements clean

ENTRY := cancer_detection_main.py

install:
	uv sync

install-dev:
	uv sync --extra dev --extra training

run:
	uv run streamlit run $(ENTRY)

train:
	uv run python -m training.train

tune:
	uv run python -m training.tune

search:
	uv run python -m training.search

search-quick:
	uv run python -m training.search --quick

test:
	uv run pytest tests/ -v

lint:
	uv run ruff check app/ training/ tests/

format:
	uv run ruff format app/ training/ tests/

requirements:
	uv export --no-dev --format requirements-txt -o requirements.txt

clean:
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -name "*.pyc" -delete
	rm -rf .ruff_cache .pytest_cache
