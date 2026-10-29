PY ?= .venv/bin/python
EXPERIMENTS := $(sort $(wildcard experiments/e*/run.py))

.PHONY: help install lint typecheck test test-slow check figures report clean

help:
	@echo "install    - venv z zależnościami deweloperskimi"
	@echo "lint       - ruff (lint + format --check)"
	@echo "typecheck  - mypy --strict"
	@echo "test       - szybkie testy (bez @slow)"
	@echo "test-slow  - testy statystyczne i krzyżowe (@slow)"
	@echo "check      - lint + typecheck + test"
	@echo "figures    - uruchamia experiments/e*/run.py, wykresy trafiają do figures/"
	@echo "report     - kompiluje report/main.tex"

install:
	python3 -m venv .venv
	$(PY) -m pip install -e '.[dev,crosscheck]'

lint:
	$(PY) -m ruff check .
	$(PY) -m ruff format --check .

typecheck:
	$(PY) -m mypy

test:
	$(PY) -m pytest -m "not slow"

test-slow:
	HYPOTHESIS_PROFILE=ci $(PY) -m pytest -m slow

check: lint typecheck test

figures:
	@if [ -z "$(EXPERIMENTS)" ]; then echo "Brak eksperymentów w experiments/e*/run.py"; fi
	@for f in $(EXPERIMENTS); do echo "== $$f"; $(PY) $$f || exit 1; done

report:
	cd report && latexmk -pdf -interaction=nonstopmode main.tex

clean:
	rm -rf .pytest_cache .mypy_cache .ruff_cache .hypothesis build dist
	cd report && latexmk -C main.tex
