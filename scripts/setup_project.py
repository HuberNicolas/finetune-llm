"""
setup_project.py - Prepare the project folders and the .env file

Usage:
    python scripts/setup_project.py
"""

import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

FOLDERS = [
    "data",
    "models",
    "models/legionaer-sft",
    "models/legionaer-final",
]


def create_folders():
    """Create the folders for data and models."""
    for name in FOLDERS:
        (ROOT / name).mkdir(parents=True, exist_ok=True)
        print(f"✓ {name}/")


def create_env_file():
    """Copy .env.example to .env if there is no .env yet."""
    env_file = ROOT / ".env"
    if env_file.exists():
        print("✓ .env exists already")
        return

    shutil.copy(ROOT / ".env.example", env_file)
    print("✓ .env created from .env.example; add your HF_TOKEN there to upload models")


def main():
    print("=" * 70)
    print("🦅 LEGIONARY SCRUM MASTER - SETUP")
    print("=" * 70)

    print("\n📁 Folders")
    create_folders()

    print("\n🔐 Environment")
    create_env_file()

    print("\n📋 Next steps:")
    print("   python src/01_dataset.py")
    print("   python src/02_train.py")
    print("   python src/03_inference.py")


if __name__ == "__main__":
    main()
