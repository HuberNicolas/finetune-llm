# Makefile für Legionär Scrum Master Projekt
# Nutze: make <target>

.PHONY: help setup install dataset train inference test upload clean

help:
	@echo "🦅 Legionär Scrum Master - Make Targets"
	@echo "======================================="
	@echo ""
	@echo "Verfügbare Befehle:"
	@echo "  make setup        - Setup: Verzeichnisse + .env"
	@echo "  make install      - Installiere Abhängigkeiten mit Pixi"
	@echo "  make dataset      - Generiere Trainingsdaten"
	@echo "  make train        - Starte Training"
	@echo "  make inference    - Starte Inference/Testing"
	@echo "  make test         - Führe Tests durch"
	@echo "  make upload       - Upload zu HuggingFace Hub"
	@echo "  make all          - Alles der Reihe nach (dataset → train → inference)"
	@echo "  make clean        - Lösche generierte Dateien"
	@echo ""

setup:
	@echo "🔧 Starte Setup..."
	python setup.py

install:
	@echo "📦 Installiere mit Pixi..."
	pixi install

dataset:
	@echo "📚 Generiere Trainingsdaten..."
	python src/01_dataset.py

train:
	@echo "🏋️  Starte Training..."
	python src/02_train.py

inference:
	@echo "🦅 Starte Inferenz..."
	python src/03_inference.py

test:
	@echo "✅ Führe Tests durch..."
	@echo "   (Noch keine Tests implementiert)"

upload:
	@echo "📤 Upload zu HuggingFace..."
	python hf_login.py upload

all: setup dataset train inference
	@echo "✅ Alle Schritte abgeschlossen!"

clean:
	@echo "🧹 Cleanup..."
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete
	rm -rf .ipynb_checkpoints
	@echo "✅ Cleanup abgeschlossen!"
