.PHONY: help setup data features train eval test lint clean

help:
	@echo "Targets:"
	@echo "  setup     Install the environment from uv.lock"
	@echo "  data      Download the Olist dataset into data/raw and verify it"
	@echo "  features  Build the feature matrices for t0 and t1"
	@echo "  train     Train the models"
	@echo "  eval      Evaluate: PR-AUC, recall@k, expected-cost threshold"
	@echo "  test      Run the test suite"
	@echo "  lint      Check formatting and lint rules"
	@echo "  clean     Remove derived data, models and caches (keeps data/raw)"

setup:
	uv sync

data:
	uv run python -m src.data

features:
	uv run python -m src.features

train:
	uv run python -m src.train

eval:
	uv run python -m src.evaluate

test:
	uv run pytest

lint:
	uv run ruff check .
	uv run ruff format --check .

clean:
	find data/interim data/processed models -type f ! -name '.gitkeep' -delete
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
