# Common tasks. Run `make help` for the list.
PY ?= .venv/bin/python
SPLITS := official holdout
FEATURES := 8 41

.PHONY: help setup data test lint format results notebooks experiments all clean

help:  ## Show this help
	@grep -E '^[a-z]+:.*## ' $(MAKEFILE_LIST) | awk -F':.*## ' '{printf "  %-12s %s\n", $$1, $$2}'

setup:  ## Create .venv and install the package with dev + notebook extras
	python3 -m venv .venv
	$(PY) -m pip install -r requirements.txt -e ".[dev,notebooks]"

data:  ## Download NSL-KDD into data/
	$(PY) -m dosdetect.download

test:  ## Run unit tests (no dataset needed)
	$(PY) -m pytest

lint:  ## Lint and check formatting
	$(PY) -m ruff check .
	$(PY) -m ruff format --check .

format:  ## Auto-fix lint issues and format code
	$(PY) -m ruff check --fix .
	$(PY) -m ruff format .

results:  ## Regenerate results/metrics/*.csv for every split and feature count
	@for s in $(SPLITS); do for n in $(FEATURES); do \
		$(PY) -m dosdetect.run --split $$s --n-features $$n > /dev/null && echo "  $$s, $$n features"; \
	done; done

notebooks:  ## Re-execute all notebooks in place (~1-2 min)
	$(PY) -m jupyter nbconvert --to notebook --execute --inplace notebooks/*.ipynb

experiments:  ## Re-run the Random Forest analysis scripts (~10 min)
	$(PY) experiments/random_forest/logic_test.py
	$(PY) experiments/random_forest/feature_importance.py
	$(PY) experiments/random_forest/importance_checks.py
	$(PY) experiments/random_forest/tuning.py

all: lint test results notebooks  ## Lint, test, and regenerate every result

clean:  ## Remove caches
	find . -path ./.venv -prune -o \( -name __pycache__ -o -name .pytest_cache -o -name .ruff_cache -o -name .ipynb_checkpoints \) -type d -print | xargs rm -rf
