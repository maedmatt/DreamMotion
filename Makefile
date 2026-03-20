.DEFAULT_GOAL := default

.PHONY: default install lint clean

default: install lint

install:
	uv sync --all-extras

lint:
	uv run python devtools/lint.py

clean:
	-rm -rf .ruff_cache/ .venv/ __pycache__/
