.PHONY: install dev lint type-check test test-all clean smoke

install:
	pip install -e .

dev:
	pip install -e ".[dev,full]"

lint:
	ruff check src/ tests/
	ruff format --check src/ tests/

format:
	ruff format src/ tests/
	ruff check --fix src/ tests/

type-check:
	mypy src/

test:
	pytest tests/ -x -q

test-all:
	pytest tests/ -v --tb=short

smoke:
	pytest tests/ -m "not slow and not gpu and not aws" -x -q

clean:
	rm -rf build/ dist/ *.egg-info .pytest_cache .mypy_cache .ruff_cache
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
