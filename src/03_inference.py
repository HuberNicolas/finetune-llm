"""
03_inference.py - Inferenz & Testing

Jetzt testen wir unser trainiertes Modell!

Mit nur den trainierten LoRA-Adaptern können wir:
1. Das Basis-Mistral-Modell laden
2. Die LoRA-Adapter laden
3. Fragen stellen und römisch-korrekte Antworten bekommen!

Der römische Legionär Scrum Master ist geboren! 🦅
"""

import torch
from pathlib import Path
from typing import Optional
import logging

# HuggingFace Imports
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    pipeline,
)
from peft import PeftModel

# ============================================================================
# LOGGING
# ============================================================================
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ============================================================================
# KONFIGURATION
# ============================================================================

MODELL_ID = "mistralai/Mistral-7B-Instruct-v0.3"
LORA_MODELL_ID = "./models/legionaer-final"

# ============================================================================
# MODELL LADEN - MIT LORA ADAPTERN
# ============================================================================

def lade_modell_mit_lora(
    basis_modell_id: str = MODELL_ID,
    lora_modell_id: str = LORA_MODELL_ID,
    use_4bit: bool = True,
) -> tuple:
    """
    Lade Basis-Modell + LoRA-Adapter zusammen.

    Das ist das Schöne an LoRA:
    - Die LoRA-Adapter sind winzig (~1% der Modell-Größe)
    - Wir speichern nur die Adapter
    - Beim Laden kombinieren wir Basis-Modell + Adapter
    - Resultat: Der "römische Legionär" ist bereit! 🦅

    Args:
        basis_modell_id: HuggingFace ID des Basis-Modells
        lora_modell_id: Pfad zu den trainierten LoRA-Adaptern
        use_4bit: Nutze 4-bit Quantisierung?

    Returns:
        (model, tokenizer)
    """
    logger.info(f"🚀 Lade Basis-Modell: {basis_modell_id}")

    # ========== TOKENIZER ==========
    tokenizer = AutoTokenizer.from_pretrained(basis_modell_id)
    tokenizer.pad_token = tokenizer.eos_token

    # ========== BASIS-MODELL MIT 4-BIT QUANTISIERUNG ==========
    if use_4bit:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
        )
        model = AutoModelForCausalLM.from_pretrained(
            basis_modell_id,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True,
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            basis_modell_id,
            device_map="auto",
            torch_dtype=torch.float16,
        )

    logger.info("✅ Basis-Modell geladen")

    # ========== LADEN DER LORA-ADAPTER ==========
    logger.info(f"🎯 Lade LoRA-Adapter: {lora_modell_id}")

    # Verprüfe ob LoRA-Modell existiert
    if not Path(lora_modell_id).exists():
        logger.error(f"❌ LoRA-Adapter nicht gefunden: {lora_modell_id}")
        logger.info("   Bitte erst 02_train.py ausführen!")
        return None, None

    # Merge die LoRA-Adapter mit dem Basis-Modell
    model = PeftModel.from_pretrained(model, lora_modell_id)

    logger.info("✅ LoRA-Adapter geladen und gemerged!")

    return model, tokenizer


# ============================================================================
# INFERENCE - GENERIERE ANTWORTEN
# ============================================================================

def generiere_antwort(
    model,
    tokenizer,
    frage: str,
    max_length: int = 256,
    temperature: float = 0.7,
    top_p: float = 0.9,
) -> str:
    """
    Generiere eine Antwort vom römischen Legionär!

    Args:
        model: Das geladene Modell mit LoRA
        tokenizer: Der Tokenizer
        frage: Die Frage an den Legionär
        max_length: Max Ausgabe-Länge
        temperature: Kreativität (0=deterministisch, 1=kreativ)
        top_p: Nucleus Sampling (höher = vielfältiger)

    Returns:
        Die römische Antwort!
    """
    # Formatiere die Frage im Mistral-Format
    prompt = f"""<s>[INST] {frage} [/INST]"""

    # Tokenisiere
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    # Generiere
    with torch.no_grad():
        outputs = model.generate(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            max_new_tokens=max_length,
            temperature=temperature,
            top_p=top_p,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
        )

    # Dekodiere die Ausgabe
    antwort = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # Entferne den Prompt aus der Antwort
    antwort = antwort.replace(prompt, "").strip()

    return antwort


# ============================================================================
# INTERAKTIVE TESTING SESSION
# ============================================================================

def starte_chat_session(model, tokenizer):
    """
    Starte eine interaktive Chat-Sitzung mit dem Legionär!

    Beispiel-Fragen:
    - "Was ist ein Sprint?"
    - "Wie handle ich mit Konflikten im Team?"
    - "Erkläre mir Agile!"
    """
    print("\n" + "=" * 70)
    print("🦅 LEGIONÄR SCRUM MASTER - CHAT SESSION")
    print("=" * 70)
    print("\nDu sprichst jetzt mit Marcus Productivus Maximus!")
    print("Der römische Legionär Scrum Master antwortet auf deine Fragen.")
    print("\nSchreibe 'exit' um zu beenden.\n")

    while True:
        # Benutzer-Input
        frage = input("Du: ").strip()

        if frage.lower() in ["exit", "quit", "q"]:
            print("\n🦅 Marcus: Vale! (Auf Wiedersehen!) Für Rom!\n")
            break

        if not frage:
            continue

        # Generiere Antwort
        print(f"\n🦅 Marcus denkt...\n")
        antwort = generiere_antwort(model, tokenizer, frage)
        print(f"🦅 Marcus: {antwort}\n")


# ============================================================================
# BATCH TESTING - VORDEFINIERTE FRAGEN
# ============================================================================

def teste_vordefinierte_fragen(model, tokenizer):
    """
    Teste das Modell mit vorfdefinierten Fragen.

    Nützlich um die Qualität zu evaluieren.
    """
    test_fragen = [
        "Was ist ein Sprint?",
        "Wie führe ich ein effektives Daily Standup?",
        "Was ist eine gute User Story?",
        "Wie gehe ich mit Konflikten im Team um?",
        "Erkläre mir Agile in deinen eigenen Worten.",
        "Was macht ein guter Scrum Master aus?",
        "Wie plane ich einen erfolgreichen Sprint?",
    ]

    print("\n" + "=" * 70)
    print("🦅 LEGIONÄR SCRUM MASTER - TEST SUITE")
    print("=" * 70)

    for i, frage in enumerate(test_fragen, 1):
        print(f"\n{'='*70}")
        print(f"Test {i}/{len(test_fragen)}")
        print(f"❓ Frage: {frage}")
        print(f"{'='*70}")

        antwort = generiere_antwort(model, tokenizer, frage)
        print(f"🦅 Antwort:\n{antwort}\n")


# ============================================================================
# EXPORTIERE MODELL
# ============================================================================

def exportiere_modell_fuer_deployment(
    model,
    tokenizer,
    output_dir: str = "./models/legionaer-deployed"
):
    """
    Exportiere das Modell für Deployment.

    Du kannst das Modell dann:
    1. Zu HuggingFace Hub uploaden
    2. Mit Ollama lokal laufen
    3. In einer FastAPI deployen
    """
    logger.info(f"📦 Exportiere Modell für Deployment: {output_dir}")

    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # Speichere das gemerged-Modell
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

    logger.info(f"✅ Modell exportiert! Du kannst es jetzt uploaden zu HuggingFace.")
    logger.info(f"   Befehle:")
    logger.info(f"   huggingface-cli login")
    logger.info(f"   huggingface-cli upload {output_dir} <repo-name>")


# ============================================================================
# MAIN
# ============================================================================

def main():
    """Haupt-Funktion"""

    # Lade Modell
    model, tokenizer = lade_modell_mit_lora()

    if model is None:
        logger.error("❌ Fehler beim Laden des Modells!")
        return

    # Menü
    print("\n" + "=" * 70)
    print("🦅 LEGIONÄR SCRUM MASTER - INFERENZ")
    print("=" * 70)
    print("\nWas möchtest du tun?")
    print("1. Chat-Session (interaktiv)")
    print("2. Test-Fragen durchlaufen")
    print("3. Modell exportieren für Deployment")
    print("4. Alle durchlaufen (1 + 2)")

    wahl = input("\nWähle (1-4): ").strip()

    if wahl == "1":
        starte_chat_session(model, tokenizer)
    elif wahl == "2":
        teste_vordefinierte_fragen(model, tokenizer)
    elif wahl == "3":
        exportiere_modell_fuer_deployment(model, tokenizer)
    elif wahl == "4":
        teste_vordefinierte_fragen(model, tokenizer)
        starte_chat_session(model, tokenizer)
    else:
        print("❌ Ungültige Wahl!")


if __name__ == "__main__":
    main()
