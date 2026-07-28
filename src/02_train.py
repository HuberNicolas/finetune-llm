"""
02_train.py - Fine-Tuning mit LoRA

Das ist das Herzstück! Hier trainieren wir das Modell mit:
- torch: Die Berechnungs-Engine
- transformers: Das HuggingFace Mistral-Modell
- bitsandbytes: 4-bit Quantisierung (spart RAM!)
- peft: LoRA-Adapter (trainiert nur ~1% der Parameter)
- trl: SFTTrainer (der Trainer-Loop)

Ziel: Das Modell lernt den römischen Legionär-Stil in ~30 Minuten auf einer T4 GPU!
"""

import os
import json
import torch
from pathlib import Path
from typing import Optional
import logging

# HuggingFace Imports
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
)
from peft import LoraConfig, get_peft_model
from trl import SFTTrainer, SFTConfig
from datasets import Dataset

# ============================================================================
# LOGGING SETUP
# ============================================================================
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ============================================================================
# KONFIGURATION
# ============================================================================

# Basis-Modell von HuggingFace
MODELL_ID = "mistralai/Mistral-7B-Instruct-v0.3"

# LoRA Hyperparameter - "wie fein-granular lernt der Adapter?"
LORA_CONFIG = {
    "r": 16,                      # Rank - breite der Adapter-Matrix
    "lora_alpha": 32,             # Scaling-Faktor (meist 2x r)
    "target_modules": [           # Welche Layer bekommen LoRA?
        "q_proj",                 # Query Projektion (Attention)
        "v_proj",                 # Value Projektion (Attention)
    ],
    "lora_dropout": 0.05,         # Dropout gegen Overfitting
    "bias": "none",               # Keine Bias-Anpassung
    "task_type": "CAUSAL_LM",     # Causal Language Modeling = nächstes Wort vorhersagen
}

# Training Hyperparameter - "wie intensiv trainieren?"
TRAINING_CONFIG = {
    "output_dir": "./models/legionaer-sft",           # Wo speichern?
    "num_train_epochs": 3,                            # 3x durch alle Daten
    "per_device_train_batch_size": 4,                 # 4 Beispiele auf einmal (GPU)
    "gradient_accumulation_steps": 2,                 # Effektive batch_size = 4*2 = 8
    "learning_rate": 2e-4,                            # Wie große Lernschritte?
    "warmup_ratio": 0.1,                              # Erst langsam "aufwärmen"
    "weight_decay": 0.01,                             # L2 Regularisierung
    "save_steps": 100,                                # Checkpoint alle 100 Schritte
    "logging_steps": 10,                              # Logs alle 10 Schritte
    "save_total_limit": 3,                            # Max 3 Checkpoints speichern
}

# ============================================================================
# SETUP - GPU / CUDA Erkennung
# ============================================================================

def pruefe_gpu():
    """Prüfe ob GPU verfügbar ist"""
    if torch.cuda.is_available():
        logger.info(f"✅ GPU gefunden: {torch.cuda.get_device_name(0)}")
        logger.info(f"   VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        return True
    else:
        logger.warning("⚠️  Keine GPU gefunden! Training wird sehr langsam sein.")
        logger.warning("   Empfohlene GPUs:")
        logger.warning("   - Kaggle: NVIDIA T4 (16 GB) oder P100 (40 GB)")
        logger.warning("   - Colab: NVIDIA T4 (16 GB)")
        logger.warning("   - HuggingFace Spaces: A10 (24 GB)")
        return False

# ============================================================================
# HAUPTFUNKTION: MODELL LADEN & VORBEREITEN
# ============================================================================

def lade_modell_und_tokenizer(
    modell_id: str = MODELL_ID,
    use_4bit: bool = True,
) -> tuple:
    """
    Laden des Mistral-Modells mit optionaler 4-bit Quantisierung.

    Die 4-bit Quantisierung (via bitsandbytes) reduziert VRAM von 28GB → 4GB!
    Das ist der "Gepäckträger-Trick" aus der Vorlesung.

    Args:
        modell_id: HuggingFace Modell-ID
        use_4bit: Sollen wir 4-bit quantisieren?

    Returns:
        (model, tokenizer)
    """
    logger.info(f"🚀 Laden Modell: {modell_id}")

    # ========== TOKENIZER laden ==========
    # Der Tokenizer wandelt Text → Token-IDs und zurück
    # z.B. "Für Rom!" → [123, 456, 789]
    tokenizer = AutoTokenizer.from_pretrained(modell_id)
    tokenizer.pad_token = tokenizer.eos_token  # Wichtig für Batching!
    logger.info(f"✅ Tokenizer geladen. Vocab-size: {tokenizer.vocab_size}")

    # ========== 4-BIT QUANTISIERUNG CONFIG ==========
    if use_4bit:
        logger.info("📦 Verwende 4-bit Quantisierung (bitsandbytes)...")
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,                    # Lade das Modell in 4-bit
            bnb_4bit_quant_type="nf4",           # NormalFloat4 = beste Qualität
            bnb_4bit_compute_dtype=torch.float16, # Rechne in float16
            bnb_4bit_use_double_quant=True,      # Double Quantization = noch mehr RAM-Sparen
        )
        model = AutoModelForCausalLM.from_pretrained(
            modell_id,
            quantization_config=bnb_config,
            device_map="auto",                    # Automatisch auf GPU verteilen
            trust_remote_code=True,
        )
    else:
        logger.info("⚠️  Keine Quantisierung! Braucht viel RAM...")
        model = AutoModelForCausalLM.from_pretrained(
            modell_id,
            device_map="auto",
            torch_dtype=torch.float16,
        )

    logger.info(f"✅ Modell geladen.")

    return model, tokenizer


# ============================================================================
# LORA ADAPTER ERSTELLEN
# ============================================================================

def erstelle_lora_adapter(model, lora_config: dict):
    """
    Wrappt das Modell mit LoRA-Adaptern.

    LoRA (Low-Rank Adaptation) ist der Trick um große Modelle effizient zu trainieren:

    Statt alle 7 Milliarden Parameter zu trainieren:
        Modell: [4096 × 4096] = ~16M Parameter pro Layer ← EINGEFROREN
        LoRA:   [4096 × 16] + [16 × 4096] = ~131K Parameter ← NUR DAS trainieren!

    Das spart ~99% der Rechenzeit und Speicher!

    Args:
        model: Das geladene Modell
        lora_config: LoRA-Konfiguration

    Returns:
        Das Modell mit LoRA-Adaptern
    """
    logger.info("🎯 Erstelle LoRA-Adapter...")

    # Konfiguriere LoRA
    peft_config = LoraConfig(
        r=lora_config["r"],
        lora_alpha=lora_config["lora_alpha"],
        lora_dropout=lora_config["lora_dropout"],
        bias=lora_config["bias"],
        target_modules=lora_config["target_modules"],
        task_type=lora_config["task_type"],
    )

    # Wrappt das Modell mit LoRA
    model = get_peft_model(model, peft_config)

    # Zeige Parameter-Statistik
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())

    logger.info(f"✅ LoRA Adapter erstellt!")
    logger.info(f"   Trainierbare Parameter: {trainable_params:,}")
    logger.info(f"   Gesamt Parameter: {total_params:,}")
    logger.info(f"   Anteil trainiert: {100 * trainable_params / total_params:.2f}%")

    return model


# ============================================================================
# DATENSATZ VORBEREITEN
# ============================================================================

def lade_datensatz(datensatz_pfad: str = "data/legionaer_training_data.json") -> Dataset:
    """
    Laden des Trainings-Datensatzes.

    Das Datensatz-Format ist JSON mit "instruction" und "output" Keys.
    Wir konvertieren das in ein HuggingFace Dataset.

    Args:
        datensatz_pfad: Pfad zur JSON-Datei

    Returns:
        HuggingFace Dataset
    """
    logger.info(f"📂 Laden Datensatz: {datensatz_pfad}")

    # JSON laden
    with open(datensatz_pfad, "r", encoding="utf-8") as f:
        data = json.load(f)

    logger.info(f"✅ {len(data)} Trainingsbeispiele geladen")

    # Formatiere als "text" für SFTTrainer
    # SFTTrainer erwartet eine "text" Spalte
    formatted_data = []
    for item in data:
        # Kombination aus Instruction + Output
        # Das Modell lernt: Instruction → Output
        text = f"""<s>[INST] {item['instruction']} [/INST] {item['output']} </s>"""
        formatted_data.append({"text": text})

    # Konvertiere zu HuggingFace Dataset
    dataset = Dataset.from_dict({"text": [d["text"] for d in formatted_data]})

    return dataset


# ============================================================================
# TRAINING STARTEN
# ============================================================================

def starte_training(
    model,
    tokenizer,
    datensatz: Dataset,
    training_config: dict,
    output_dir: str = "models/legionaer-sft",
):
    """
    Trainiere das Modell mit SFTTrainer.

    SFTTrainer (Supervised Fine-Tuning Trainer) ist der "Drill-Sergeant":
    - Lädt die Daten
    - Berechnet Loss (wie weit war die Vorhersage weg?)
    - Macht Backpropagation (berechnet Gradienten)
    - Aktualisiert die LoRA-Adapter
    - Speichert Checkpoints

    Args:
        model: Das Modell mit LoRA-Adaptern
        tokenizer: Der Tokenizer
        datensatz: Das Training Dataset
        training_config: Hyperparameter
        output_dir: Wo speichern wir die trainierten Adapter?
    """
    logger.info("🏋️ Starte Training...")

    # Output-Verzeichnis erstellen
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # Training-Argumente
    training_args = SFTConfig(
        output_dir=output_dir,
        num_train_epochs=training_config["num_train_epochs"],
        per_device_train_batch_size=training_config["per_device_train_batch_size"],
        gradient_accumulation_steps=training_config["gradient_accumulation_steps"],
        learning_rate=training_config["learning_rate"],
        warmup_ratio=training_config["warmup_ratio"],
        weight_decay=training_config["weight_decay"],
        save_strategy="steps",
        save_steps=training_config["save_steps"],
        logging_steps=training_config["logging_steps"],
        save_total_limit=training_config["save_total_limit"],
        report_to=[],  # Keine W&B oder mlflow Reporting (optional)
        max_seq_length=512,  # Max Sequenz-Länge
        optim="paged_adamw_32bit",  # Memory-effizient
    )

    # Erstelle Trainer
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=datensatz,
        args=training_args,
        packing=True,  # Packe kurze Beispiele zusammen = schneller
    )

    # Los geht's!
    logger.info("⚔️ TRAINING STARTET! Für Rom!")
    trainer.train()

    logger.info(f"✅ Training abgeschlossen! Modell gespeichert in: {output_dir}")

    return trainer


# ============================================================================
# MAIN
# ============================================================================

def main():
    """Führe den gesamten Training-Pipeline aus"""

    print("\n" + "=" * 70)
    print("🦅 LEGIONÄR SCRUM MASTER - FINE-TUNING")
    print("=" * 70)

    # 1. GPU Check
    pruefe_gpu()

    # 2. Stelle sicher dass Datensatz existiert
    datensatz_pfad = "data/legionaer_training_data.json"
    if not Path(datensatz_pfad).exists():
        logger.error(f"❌ Datensatz nicht gefunden: {datensatz_pfad}")
        logger.info("   Bitte erst 01_dataset.py ausführen!")
        return

    # 3. Lade Modell
    model, tokenizer = lade_modell_und_tokenizer(use_4bit=True)

    # 4. Erstelle LoRA Adapter
    model = erstelle_lora_adapter(model, LORA_CONFIG)

    # 5. Lade Datensatz
    datensatz = lade_datensatz(datensatz_pfad)
    logger.info(f"📊 Datensatz-Größe: {len(datensatz)} Beispiele")

    # 6. Starte Training
    trainer = starte_training(
        model,
        tokenizer,
        datensatz,
        TRAINING_CONFIG,
    )

    # 7. Speichere finale LoRA-Adapter
    final_model_dir = "./models/legionaer-final"
    Path(final_model_dir).mkdir(parents=True, exist_ok=True)
    model.save_pretrained(final_model_dir)
    logger.info(f"✅ Finale LoRA-Adapter gespeichert: {final_model_dir}")

    print("\n" + "=" * 70)
    print("🎉 TRAINING ERFOLGREICH!")
    print("=" * 70)
    print(f"Modell gespeichert in: ./models/legionaer-final")
    print(f"Training-Logs: ./models/legionaer-sft/")
    print("\nNächster Schritt: Starten Sie 03_inference.py um das Modell zu testen!")


if __name__ == "__main__":
    main()
