#!/bin/bash
# start_training.sh - Run all steps in a row: dataset, training, inference

set -e  # stop on the first error

echo "========================================"
echo "🦅 LEGIONARY SCRUM MASTER - START SCRIPT"
echo "========================================"
echo ""

if [[ "$OSTYPE" == "linux-gnu"* ]]; then
    PYTHON="python3"
else
    PYTHON="python"
fi

echo "1️⃣ Writing the training data..."
$PYTHON src/01_dataset.py
echo ""
echo "2️⃣ Training..."
echo "   (about 20 minutes on a T4 GPU)"
echo ""
$PYTHON src/02_train.py
echo ""
echo "3️⃣ Inference..."
$PYTHON src/03_inference.py
echo ""
echo "✅ Done! 🎉"
