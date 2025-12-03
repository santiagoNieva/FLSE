.PHONY: help install activate test lint format typecheck all

help:
	@echo "Targets:"
	@echo "  install    - Instala dependencias con Poetry"
	@echo "  activate   - Activa el entorno virtual de Poetry"
	@echo "  test       - Corre los tests con pytest"
	@echo "  lint       - Corre black e isort"
	@echo "  format     - Formatea el código con black e isort"
	@echo "  typecheck  - Corre mypy sobre el paquete flse"
	@echo "  all        - format + test + typecheck"

install:
	poetry install

activate:
	@echo "source $$(poetry env info --path)/bin/activate"

test:
	poetry run pytest -vv --disable-warnings -o log_cli=true -o log_cli_level=INFO

lint:
	poetry run black flse tests
	poetry run isort flse tests

format: lint

typecheck:
	poetry run mypy flse

all: format test typecheck
