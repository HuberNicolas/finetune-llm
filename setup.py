"""
setup.py - Projekt-Setup & Konfiguration

Nutze dieses Script um das Projekt richtig zu starten.
"""

import os
import sys
from pathlib import Path

def erstelle_verzeichnisse():
    """Erstelle notwendige Verzeichnisse"""
    dirs = [
        "data",
        "models",
        "models/legionaer-sft",
        "models/legionaer-final",
        "notebooks",
        "logs",
    ]

    for dir_name in dirs:
        Path(dir_name).mkdir(parents=True, exist_ok=True)
        print(f"✓ {dir_name}")

    print("\n✅ Verzeichnisse erstellt!")


def erstelle_env_template():
    """Erstelle .env.example Datei"""
    env_content = """# HuggingFace
HF_TOKEN=your_token_here

# Kaggle (optional)
KAGGLE_USERNAME=
KAGGLE_KEY=

# Trainingskonfiguration
CUDA_VISIBLE_DEVICES=0
WANDB_DISABLED=true
"""

    env_file = Path(".env.example")
    if not env_file.exists():
        with open(env_file, "w") as f:
            f.write(env_content)
        print("✓ .env.example erstellt")
    else:
        print("✓ .env.example existiert bereits")

    print("\n💡 Kopiere .env.example zu .env und fülle die Werte aus!")


def main():
    print("=" * 70)
    print("🦅 LEGIONÄR SCRUM MASTER - SETUP")
    print("=" * 70)

    print("\n📁 Erstelle Verzeichnisstruktur...")
    erstelle_verzeichnisse()

    print("\n🔐 Erstelle Environment-Template...")
    erstelle_env_template()

    print("\n" + "=" * 70)
    print("✅ Setup abgeschlossen!")
    print("=" * 70)
    print("\n📋 Nächste Schritte:")
    print("   1. Kopiere .env.example zu .env")
    print("   2. Fülle deine HuggingFace Token ein (optional)")
    print("   3. Starte: python src/01_dataset.py")
    print("   4. Dann: python src/02_train.py")
    print("   5. Schließlich: python src/03_inference.py")
    print("\n💻 Oder nutze Pixi:")
    print("   pixi install")
    print("   pixi run python src/01_dataset.py")
    print("\n🚀 Viel Erfolg!")


if __name__ == "__main__":
    main()
