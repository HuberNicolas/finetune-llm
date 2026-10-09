"""
03_inference.py - Inference and testing

Tests the trained model. With the trained LoRA adapters we:
1. load the base Mistral model,
2. put the LoRA adapters on top,
3. ask questions and get answers in the legionary's style.

The test questions are in German, like the training data.
"""

import logging
from pathlib import Path

import torch
import yaml
from peft import PeftModel
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)

# ============================================================================
# LOGGING
# ============================================================================
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ============================================================================
# CONFIGURATION (config.yaml in the repository root)
# ============================================================================

CONFIG_PATH = Path(__file__).resolve().parent.parent / "config.yaml"


def load_config(path: Path = CONFIG_PATH) -> dict:
    """Read the settings from config.yaml."""
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


CONFIG = load_config()

# ============================================================================
# LOAD MODEL WITH LORA ADAPTERS
# ============================================================================


def load_model_with_lora(
    base_model_id: str = CONFIG["model"]["base_model_id"],
    lora_model_path: str = CONFIG["paths"]["lora_adapter"],
    use_4bit: bool = CONFIG["quantization"]["use_4bit"],
) -> tuple:
    """
    Load the base model and put the LoRA adapters on top.

    The adapters are tiny compared to the model, so only they are saved after
    training. At inference time they are combined with the base model again.

    Args:
        base_model_id: Hugging Face ID of the base model
        lora_model_path: Path to the trained LoRA adapters
        use_4bit: Load the base model quantised to 4 bit

    Returns:
        (model, tokenizer), or (None, None) if the adapters are missing
    """
    if not Path(lora_model_path).exists():
        logger.error(f"❌ LoRA adapters not found: {lora_model_path}")
        logger.info("   Run src/02_train.py first")
        return None, None

    logger.info(f"🚀 Loading base model: {base_model_id}")

    tokenizer = AutoTokenizer.from_pretrained(base_model_id)
    tokenizer.pad_token = tokenizer.eos_token

    if use_4bit:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type=CONFIG["quantization"]["quant_type"],
            bnb_4bit_compute_dtype=torch.float16,
        )
        model = AutoModelForCausalLM.from_pretrained(
            base_model_id,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True,
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            base_model_id,
            device_map="auto",
            torch_dtype=torch.float16,
        )

    logger.info("✅ Base model loaded")

    logger.info(f"🎯 Loading LoRA adapters: {lora_model_path}")
    model = PeftModel.from_pretrained(model, lora_model_path)
    logger.info("✅ LoRA adapters loaded")

    return model, tokenizer


# ============================================================================
# GENERATE ANSWERS
# ============================================================================


def generate_answer(
    model,
    tokenizer,
    question: str,
    max_new_tokens: int = CONFIG["inference"]["max_new_tokens"],
    temperature: float = CONFIG["inference"]["temperature"],
    top_p: float = CONFIG["inference"]["top_p"],
) -> str:
    """
    Generate the legionary's answer to a question.

    Args:
        model: The model with LoRA adapters
        tokenizer: The tokenizer
        question: The question for the legionary
        max_new_tokens: Maximum length of the answer
        temperature: Creativity (0 = deterministic, 1 = creative)
        top_p: Nucleus sampling (higher = more varied)

    Returns:
        The answer
    """
    # Mistral prompt format; the tokenizer adds the BOS token <s> itself
    prompt = f"""[INST] {question} [/INST]"""

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
        )

    # Decode only the generated tokens, without the prompt
    new_tokens = outputs[0][inputs["input_ids"].shape[1] :]
    return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()


# ============================================================================
# INTERACTIVE CHAT
# ============================================================================


def start_chat_session(model, tokenizer):
    """
    Chat with the legionary in the terminal.

    Example questions:
    - "Was ist ein Sprint?"
    - "Wie gehe ich mit Konflikten im Team um?"
    - "Erkläre mir Agile!"
    """
    print("\n" + "=" * 70)
    print("🦅 LEGIONARY SCRUM MASTER - CHAT SESSION")
    print("=" * 70)
    print("\nYou are talking to Marcus Productivus Maximus.")
    print("Ask questions about Scrum (in German works best).")
    print("\nType 'exit' to quit.\n")

    while True:
        question = input("You: ").strip()

        if question.lower() in ["exit", "quit", "q"]:
            print("\n🦅 Marcus: Vale! For Rome!\n")
            break

        if not question:
            continue

        print("\n🦅 Marcus is thinking...\n")
        answer = generate_answer(model, tokenizer, question)
        print(f"🦅 Marcus: {answer}\n")


# ============================================================================
# TEST QUESTIONS
# ============================================================================


def run_test_questions(model, tokenizer):
    """Answer a fixed set of questions, to judge the result by eye."""
    test_questions = [
        "Was ist ein Sprint?",
        "Wie führe ich ein effektives Daily Standup?",
        "Was ist eine gute User Story?",
        "Wie gehe ich mit Konflikten im Team um?",
        "Erkläre mir Agile in deinen eigenen Worten.",
        "Was macht ein guter Scrum Master aus?",
        "Wie plane ich einen erfolgreichen Sprint?",
    ]

    print("\n" + "=" * 70)
    print("🦅 LEGIONARY SCRUM MASTER - TEST QUESTIONS")
    print("=" * 70)

    for i, question in enumerate(test_questions, 1):
        print(f"\n{'=' * 70}")
        print(f"Test {i}/{len(test_questions)}")
        print(f"❓ Question: {question}")
        print(f"{'=' * 70}")

        answer = generate_answer(model, tokenizer, question)
        print(f"🦅 Answer:\n{answer}\n")


# ============================================================================
# EXPORT
# ============================================================================


def export_model(model, tokenizer, output_dir: str = CONFIG["paths"]["export"]):
    """
    Save the LoRA adapters together with the tokenizer, ready to upload to the
    Hugging Face Hub. The base model is not included; users load it from the
    Hub and put the adapters on top.
    """
    logger.info(f"📦 Exporting adapters and tokenizer: {output_dir}")

    Path(output_dir).mkdir(parents=True, exist_ok=True)

    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

    logger.info("✅ Export done. Upload it to the Hugging Face Hub with:")
    logger.info("   hf auth login")
    logger.info(f"   hf upload <username>/<repo-name> {output_dir}")


# ============================================================================
# MAIN
# ============================================================================


def main():
    model, tokenizer = load_model_with_lora()

    if model is None:
        return

    print("\n" + "=" * 70)
    print("🦅 LEGIONARY SCRUM MASTER - INFERENCE")
    print("=" * 70)
    print("\nWhat do you want to do?")
    print("1. Chat (interactive)")
    print("2. Run the test questions")
    print("3. Export adapters for upload")
    print("4. Test questions, then chat")

    choice = input("\nChoose (1-4): ").strip()

    if choice == "1":
        start_chat_session(model, tokenizer)
    elif choice == "2":
        run_test_questions(model, tokenizer)
    elif choice == "3":
        export_model(model, tokenizer)
    elif choice == "4":
        run_test_questions(model, tokenizer)
        start_chat_session(model, tokenizer)
    else:
        print("❌ Invalid choice")


if __name__ == "__main__":
    main()
