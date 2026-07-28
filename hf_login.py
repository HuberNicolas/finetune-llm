"""
hf_login.py - HuggingFace Login & Model Upload

Mit diesem Script kannst du dich bei HuggingFace anmelden
und dein trainiertes Modell uploaden.
"""

import os
from pathlib import Path
from huggingface_hub import login, HfApi, create_repo

def login_hf():
    """Login zu HuggingFace"""
    print("=" * 70)
    print("🤗 HuggingFace Login")
    print("=" * 70)
    print("\n1. Gehe zu https://huggingface.co/settings/tokens")
    print("2. Klicke 'New token'")
    print("3. Gib einen Namen ein (z.B. 'legionaer-training')")
    print("4. Wähle 'Write' Rechte")
    print("5. Kopiere den Token\n")

    try:
        login()  # Interaktive Login
        print("\n✅ Erfolgreich angemeldet!")
    except Exception as e:
        print(f"\n❌ Login fehlgeschlagen: {e}")


def upload_model_zu_hf(
    modell_pfad: str = "./models/legionaer-final",
    repo_name: str = "legionaer-scrum-master",
    repo_type: str = "model"
):
    """
    Upload dein trainiertes Modell zu HuggingFace Hub.

    Args:
        modell_pfad: Lokaler Pfad zum Modell
        repo_name: Name des Repos auf HF (z.B. "legionaer-scrum-master")
        repo_type: "model", "dataset", oder "space"
    """
    print("=" * 70)
    print(f"📦 Upload Modell zu HuggingFace: {repo_name}")
    print("=" * 70)

    # Prüfe ob Modell existiert
    modell_path = Path(modell_pfad)
    if not modell_path.exists():
        print(f"❌ Modell-Ordner nicht gefunden: {modell_pfad}")
        return False

    try:
        api = HfApi()

        # Token von credentials auslesen
        from huggingface_hub import get_token
        token = get_token()

        if not token:
            print("❌ HuggingFace Token nicht gefunden. Bitte erst login() ausführen.")
            return False

        # Repo-ID zusammensetzen (username/repo-name)
        repo_id = f"{api.whoami()['name']}/{repo_name}"

        print(f"\n🚀 Uploade zu: {repo_id}")
        print(f"   Modell-Pfad: {modell_pfad}")

        # Upload
        api.upload_folder(
            folder_path=modell_pfad,
            repo_id=repo_id,
            repo_type=repo_type,
            commit_message="Upload trained LoRA adapter",
            token=token,
        )

        print(f"\n✅ Upload erfolgreich!")
        print(f"   Model URL: https://huggingface.co/{repo_id}")
        print(f"\n💻 So nutzt jemand das Modell:")
        print(f"   from transformers import AutoTokenizer, AutoModelForCausalLM")
        print(f"   from peft import PeftModel")
        print(f"   base_model_id = 'mistralai/Mistral-7B-Instruct-v0.3'")
        print(f"   lora_model_id = '{repo_id}'")
        print(f"   model = AutoModelForCausalLM.from_pretrained(base_model_id)")
        print(f"   model = PeftModel.from_pretrained(model, lora_model_id)")

        return True

    except Exception as e:
        print(f"\n❌ Upload fehlgeschlagen: {e}")
        return False


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "upload":
        # Nutze: python hf_login.py upload
        login_hf()
        upload_model_zu_hf()
    else:
        # Standard: nur login
        login_hf()
