"""
hf_login.py - Hugging Face login and model upload

Log in to Hugging Face and upload the trained LoRA adapters.

Usage:
    python hf_login.py          # log in only
    python hf_login.py upload   # log in, then upload

If HF_TOKEN is set in the environment or in a .env file, it is used instead of
the interactive login.
"""

import sys
from pathlib import Path

from dotenv import load_dotenv
from huggingface_hub import HfApi, get_token, login


def login_hf():
    """Log in to Hugging Face interactively."""
    print("=" * 70)
    print("🤗 Hugging Face login")
    print("=" * 70)
    print("\n1. Go to https://huggingface.co/settings/tokens")
    print("2. Click 'Create new token'")
    print("3. Enter a name (e.g. 'legionaer-training')")
    print("4. Choose 'Write' access")
    print("5. Copy the token\n")

    try:
        login()
        print("\n✅ Logged in")
    except Exception as e:
        print(f"\n❌ Login failed: {e}")


def upload_model_to_hf(
    model_path: str = "./models/legionaer-final",
    repo_name: str = "legionaer-scrum-master",
    repo_type: str = "model",
):
    """
    Upload the trained adapters to the Hugging Face Hub.

    Args:
        model_path: Local folder with the adapters
        repo_name: Name of the repository on the Hub
        repo_type: "model", "dataset" or "space"
    """
    print("=" * 70)
    print(f"📦 Uploading model to Hugging Face: {repo_name}")
    print("=" * 70)

    if not Path(model_path).exists():
        print(f"❌ Model folder not found: {model_path}")
        return False

    try:
        api = HfApi()

        token = get_token()
        if not token:
            print("❌ No Hugging Face token found. Log in first.")
            return False

        repo_id = f"{api.whoami()['name']}/{repo_name}"

        print(f"\n🚀 Uploading to: {repo_id}")
        print(f"   Model path: {model_path}")

        api.create_repo(repo_id=repo_id, repo_type=repo_type, exist_ok=True)
        api.upload_folder(
            folder_path=model_path,
            repo_id=repo_id,
            repo_type=repo_type,
            commit_message="Upload trained LoRA adapter",
            token=token,
        )

        print("\n✅ Upload done")
        print(f"   Model URL: https://huggingface.co/{repo_id}")
        print("\n💻 How to use it:")
        print("   from transformers import AutoModelForCausalLM")
        print("   from peft import PeftModel")
        print("   base_model_id = 'mistralai/Mistral-7B-Instruct-v0.3'")
        print(f"   lora_model_id = '{repo_id}'")
        print("   model = AutoModelForCausalLM.from_pretrained(base_model_id)")
        print("   model = PeftModel.from_pretrained(model, lora_model_id)")

        return True

    except Exception as e:
        print(f"\n❌ Upload failed: {e}")
        return False


if __name__ == "__main__":
    load_dotenv()
    if get_token():
        print("✅ Using the Hugging Face token from HF_TOKEN or an earlier login")
    else:
        login_hf()
    if len(sys.argv) > 1 and sys.argv[1] == "upload":
        upload_model_to_hf()
