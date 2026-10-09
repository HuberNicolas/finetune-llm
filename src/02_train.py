"""
02_train.py - Fine-tuning with LoRA

The core of the project. It trains the model with:
- torch: the compute engine
- transformers: the Hugging Face Mistral model
- bitsandbytes: 4-bit quantisation (saves memory)
- peft: LoRA adapters (trains well under 1% of the parameters)
- trl: SFTTrainer (the training loop)

Goal: the model picks up the Roman legionary's style in about 30 minutes on a T4 GPU.
"""

import json
import logging
import os
import sys
import traceback
from pathlib import Path

import psutil
import torch
from datasets import Dataset
from peft import LoraConfig, get_peft_model
from tqdm import tqdm
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)
from trl import SFTConfig, SFTTrainer

# ============================================================================
# LOGGING
# ============================================================================
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# ============================================================================
# SYSTEM MONITORING
# ============================================================================


def show_system_info():
    """Print GPU, CPU and RAM information."""
    print("\n" + "=" * 70)
    print("🖥️  SYSTEM INFORMATION")
    print("=" * 70)

    if torch.cuda.is_available():
        print(f"✅ GPU: {torch.cuda.get_device_name(0)}")
        print(f"   CUDA version: {torch.version.cuda}")
        gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"   VRAM: {gpu_mem:.1f} GB")
    else:
        print("⚠️  GPU: NOT AVAILABLE (training would be very slow)")

    cpu_count = psutil.cpu_count()
    print(f"\n💻 CPU: {cpu_count} cores")

    ram_info = psutil.virtual_memory()
    print(f"   RAM: {ram_info.total / 1e9:.1f} GB (available: {ram_info.available / 1e9:.1f} GB)")

    print("=" * 70 + "\n")


# ============================================================================
# CONFIGURATION
# ============================================================================

# Base model on Hugging Face
MODEL_ID = "mistralai/Mistral-7B-Instruct-v0.3"

# LoRA hyperparameters - how much capacity the adapters get
LORA_CONFIG = {
    "r": 16,  # rank: width of the adapter matrices
    "lora_alpha": 32,  # scaling factor (usually 2x r)
    "target_modules": [  # layers that get adapters
        "q_proj",  # query projection (attention)
        "v_proj",  # value projection (attention)
    ],
    "lora_dropout": 0.05,  # dropout against overfitting
    "bias": "none",  # do not train biases
    "task_type": "CAUSAL_LM",  # causal language modelling = predict the next token
}

# Training hyperparameters
TRAINING_CONFIG = {
    "output_dir": "./models/legionaer-sft",  # checkpoints
    "num_train_epochs": 3,  # passes over the data
    "per_device_train_batch_size": 4,  # examples per step on the GPU
    "gradient_accumulation_steps": 2,  # effective batch size = 4 * 2 = 8
    "learning_rate": 2e-4,
    "warmup_ratio": 0.1,  # warm up during the first 10% of the steps
    "weight_decay": 0.01,  # L2 regularisation
    "save_steps": 100,  # checkpoint every 100 steps
    "logging_steps": 10,  # log every 10 steps
    "save_total_limit": 3,  # keep at most 3 checkpoints
}

DATASET_PATH = "data/legionaer_training_data.json"
FINAL_MODEL_DIR = "./models/legionaer-final"

# ============================================================================
# GPU CHECK
# ============================================================================


def check_gpu():
    """Return True if a CUDA GPU is available."""
    print("\n" + "=" * 70)
    print("🔍 GPU DETECTION")
    print("=" * 70)

    if torch.cuda.is_available():
        print(f"✅ GPU found: {torch.cuda.get_device_name(0)}")
        print(f"   VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        torch.cuda.empty_cache()
        print("✅ GPU cache cleared")
        return True

    logger.error("❌ NO GPU AVAILABLE!")
    logger.error("   Training without a GPU is not supported.")
    logger.error("   Enable a GPU in:")
    logger.error("   - Colab: Runtime → Change runtime type → GPU")
    logger.error("   - Kaggle: Notebook settings → GPU")
    return False


# ============================================================================
# LOAD MODEL AND TOKENIZER
# ============================================================================


def load_model_and_tokenizer(
    model_id: str = MODEL_ID,
    use_4bit: bool = True,
) -> tuple:
    """Load the Mistral model, optionally quantised to 4 bit."""
    print("\n" + "=" * 70)
    print("🚀 LOADING MODEL")
    print("=" * 70)

    try:
        logger.info(f"1️⃣  Loading tokenizer: {model_id}")
        with tqdm(total=1, desc="Tokenizer", unit="step") as pbar:
            tokenizer = AutoTokenizer.from_pretrained(model_id)
            tokenizer.pad_token = tokenizer.eos_token
            pbar.update(1)
        logger.info(f"✅ Tokenizer loaded. Vocabulary size: {tokenizer.vocab_size}")

        if use_4bit:
            logger.info("2️⃣  Configuring 4-bit quantisation...")
            with tqdm(total=1, desc="Quantisation", unit="step") as pbar:
                bnb_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_compute_dtype=torch.float16,
                    bnb_4bit_use_double_quant=True,
                )
                pbar.update(1)
            logger.info("✅ 4-bit config created (~14 GB → ~4 GB)")

            logger.info("3️⃣  Loading model (large download, ~15 GB)...")
            model = AutoModelForCausalLM.from_pretrained(
                model_id,
                quantization_config=bnb_config,
                device_map="auto",
                trust_remote_code=True,
            )
        else:
            logger.warning("⚠️  No quantisation (needs ~15 GB of VRAM)")
            model = AutoModelForCausalLM.from_pretrained(
                model_id,
                device_map="auto",
                torch_dtype=torch.float16,
            )

        logger.info("✅ Model loaded")
        print("=" * 70)

        return model, tokenizer

    except Exception as e:
        logger.error("❌ Failed to load the model!")
        logger.error(f"   {str(e)}")
        logger.error("   Hints:")
        logger.error("   - Is a GPU available? (torch.cuda.is_available())")
        logger.error("   - Enough VRAM? (a T4 has 16 GB)")
        logger.error("   - Internet connection OK? (large download)")
        raise


# ============================================================================
# LORA ADAPTERS
# ============================================================================


def add_lora_adapters(model, lora_config: dict):
    """
    Wrap the model with LoRA adapters.

    LoRA (Low-Rank Adaptation) trains large models efficiently. Instead of
    updating a full weight matrix, it trains two small matrices next to it:

        Weight: [4096 × 4096] = ~16.8M parameters per layer ← frozen
        LoRA:   [4096 × 16] + [16 × 4096] = ~131K parameters ← trained

    Args:
        model: The loaded model
        lora_config: LoRA settings

    Returns:
        The model with LoRA adapters
    """
    logger.info("🎯 Adding LoRA adapters...")

    peft_config = LoraConfig(
        r=lora_config["r"],
        lora_alpha=lora_config["lora_alpha"],
        lora_dropout=lora_config["lora_dropout"],
        bias=lora_config["bias"],
        target_modules=lora_config["target_modules"],
        task_type=lora_config["task_type"],
    )

    model = get_peft_model(model, peft_config)

    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())

    logger.info("✅ LoRA adapters added")
    logger.info(f"   Trainable parameters: {trainable_params:,}")
    logger.info(f"   Total parameters: {total_params:,}")
    logger.info(f"   Share trained: {100 * trainable_params / total_params:.2f}%")

    return model


# ============================================================================
# DATASET
# ============================================================================


def load_dataset(dataset_path: str = DATASET_PATH) -> Dataset:
    """Load the training examples and format them in the Mistral prompt format."""
    print("\n" + "=" * 70)
    print("📂 LOADING DATASET")
    print("=" * 70)

    if not os.path.exists(dataset_path):
        logger.error(f"❌ Dataset not found: {dataset_path}")
        raise FileNotFoundError(f"Dataset not found: {dataset_path}")

    with open(dataset_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    logger.info(f"✅ {len(data)} training examples loaded")

    # The tokenizer adds the BOS token <s> itself
    texts = [
        f"""[INST] {item["instruction"]} [/INST] {item["output"]} </s>"""
        for item in tqdm(data, desc="Formatting dataset", unit="item")
    ]

    dataset = Dataset.from_dict({"text": texts})
    logger.info("✅ Dataset formatted")
    print("=" * 70)

    return dataset


# ============================================================================
# TRAINING
# ============================================================================


def train(
    model,
    tokenizer,
    dataset: Dataset,
    training_config: dict,
):
    """
    Train the model with TRL's SFTTrainer.

    SFTTrainer (supervised fine-tuning) runs the loop:
    - batches the data
    - computes the loss (how far off was the prediction?)
    - backpropagates (computes the gradients)
    - updates the LoRA adapters
    - saves checkpoints

    Args:
        model: The model with LoRA adapters
        tokenizer: The tokenizer
        dataset: The training dataset
        training_config: Hyperparameters, including output_dir
    """
    logger.info("🏋️ Starting training...")

    output_dir = training_config["output_dir"]
    Path(output_dir).mkdir(parents=True, exist_ok=True)

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
        report_to=[],  # no W&B or MLflow reporting
        max_length=512,  # maximum sequence length
        packing=True,  # pack short examples together = faster
        optim="paged_adamw_32bit",  # memory-efficient optimiser
    )

    trainer = SFTTrainer(
        model=model,
        processing_class=tokenizer,
        train_dataset=dataset,
        args=training_args,
    )

    print("\n" + "=" * 70)
    print("⚔️  TRAINING STARTS! For Rome!")
    print("=" * 70)

    try:
        trainer.train()
        print("\n" + "=" * 70)
        print("✅ TRAINING COMPLETED")
        print("=" * 70)
        logger.info(f"📁 Checkpoints saved in: {output_dir}")
    except Exception as e:
        print("\n" + "=" * 70)
        print("❌ TRAINING FAILED!")
        print("=" * 70)
        logger.error(f"Error: {str(e)}")
        logger.error("Hints:")
        logger.error("   - Out of memory (OOM)? Reduce the batch size")
        logger.error("   - Network error? Try again")
        logger.error("   - GPU error? Restart the Colab runtime")
        raise

    return trainer


# ============================================================================
# MAIN
# ============================================================================


def main() -> int:
    """Run the whole training pipeline."""

    print("\n" + "=" * 70)
    print("🦅 LEGIONARY SCRUM MASTER - FINE-TUNING")
    print("=" * 70)

    try:
        # 1. System info and GPU check
        show_system_info()
        if not check_gpu():
            logger.error("Aborting: no GPU available")
            return 1

        # 2. Make sure the dataset exists
        if not Path(DATASET_PATH).exists():
            logger.error(f"❌ Dataset not found: {DATASET_PATH}")
            logger.info("   Run src/01_dataset.py first")
            return 1

        # 3. Load the model
        model, tokenizer = load_model_and_tokenizer(use_4bit=True)

        # 4. Add LoRA adapters
        model = add_lora_adapters(model, LORA_CONFIG)

        # 5. Load the dataset
        dataset = load_dataset(DATASET_PATH)

        # 6. Train
        train(model, tokenizer, dataset, TRAINING_CONFIG)

        # 7. Save the final LoRA adapters
        print("\n" + "=" * 70)
        print("💾 SAVING FINAL ADAPTERS")
        print("=" * 70)
        Path(FINAL_MODEL_DIR).mkdir(parents=True, exist_ok=True)

        with tqdm(total=1, desc="Saving model", unit="step") as pbar:
            model.save_pretrained(FINAL_MODEL_DIR)
            pbar.update(1)

        print("\n" + "=" * 70)
        print("🎉 DONE!")
        print("=" * 70)
        print(f"✅ Trained adapters: {FINAL_MODEL_DIR}/")
        print("✅ Next step: python src/03_inference.py")
        print("=" * 70 + "\n")

        return 0

    except KeyboardInterrupt:
        print("\n" + "=" * 70)
        print("⚠️  TRAINING CANCELLED (Ctrl+C)")
        print("=" * 70)
        return 1

    except Exception as e:
        print("\n" + "=" * 70)
        print("❌ ERROR!")
        print("=" * 70)
        logger.error(f"Error: {type(e).__name__}: {str(e)}")
        traceback.print_exc()
        print("=" * 70)
        return 1


if __name__ == "__main__":
    sys.exit(main())
