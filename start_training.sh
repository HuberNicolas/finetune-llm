#!/bin/bash
# start_training.sh - Einfaches Script um alles nacheinander zu starten

set -e  # Exit bei Fehler

echo "========================================"
echo "🦅 LEGIONÄR SCRUM MASTER - START SCRIPT"
echo "========================================"
echo ""

# Check OS
if [[ "$OSTYPE" == "linux-gnu"* ]]; then
    PYTHON="python3"
else
    PYTHON="python"
fi

echo "1️⃣ Generiere Trainingsdaten..."
$PYTHON src/01_dataset.py
echo ""
echo "2️⃣ Starte Training..."
echo "   (Das dauert ~20 Minuten auf T4 GPU)"
echo ""
$PYTHON src/02_train.py
echo ""
echo "3️⃣ Starte Inference..."
$PYTHON src/03_inference.py
echo ""
echo "✅ Fertig! 🎉"
