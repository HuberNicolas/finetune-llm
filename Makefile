# Makefile for the legionary Scrum Master project
# Usage: make <target>

.PHONY: help setup install dataset train inference lint upload all clean

help:
	@echo "🦅 Legionary Scrum Master - make targets"
	@echo "========================================"
	@echo ""
	@echo "  make setup        - create folders and .env"
	@echo "  make install      - install the dependencies with Pixi"
	@echo "  make dataset      - write the training data"
	@echo "  make train        - train the LoRA adapters"
	@echo "  make inference    - chat with or test the trained model"
	@echo "  make lint         - check and format the code with Ruff"
	@echo "  make upload       - upload the adapters to the Hugging Face Hub"
	@echo "  make all          - setup, dataset, train and inference in a row"
	@echo "  make clean        - delete caches"
	@echo ""

setup:
	python scripts/setup_project.py

install:
	pixi install

dataset:
	python src/01_dataset.py

train:
	python src/02_train.py

inference:
	python src/03_inference.py

lint:
	uvx ruff check .
	uvx ruff format --check .

upload:
	python hf_login.py upload

all: setup dataset train inference

clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete
	rm -rf .ipynb_checkpoints .ruff_cache
