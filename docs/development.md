# Development

## Environment

| Use case | Setup |
|---|---|
| Local, Linux or Windows with an NVIDIA GPU | [Pixi](https://pixi.sh/): `pixi install`, then `pixi shell`. Versions are locked in [`pixi.lock`](../pixi.lock). |
| Colab or Kaggle | `pip install -r requirements-kaggle.txt`; the ML libraries are pinned to the versions in `pixi.lock` |
| macOS | No CUDA, so no training. The Pixi environment is only defined for `linux-64` and `win-64`. Linting works everywhere. |

The code uses the APIs of trl 1.x and transformers 5.x (e.g. `processing_class=` and `SFTConfig(max_length=…)`).
Older versions of these libraries do not work.

## Commands

| Task | Command |
|---|---|
| Create `models/` folders and `.env` | `make setup` |
| Write the training data | `make dataset` |
| Train | `make train` |
| Chat, test questions, export | `make inference` |
| Upload to the Hugging Face Hub | `make upload` |
| Lint and check formatting | `make lint` |
| Fix lint findings and format | `uvx ruff check --fix .` and `uvx ruff format .` |

[Ruff](https://docs.astral.sh/ruff/) is configured in [`ruff.toml`](../ruff.toml) (line length 120, errors, Pyflakes
and import sorting). The notebook is excluded.

## CI

[`.github/workflows/lint.yml`](../.github/workflows/lint.yml) runs on every push to `main` and on pull requests:
Ruff, a compile check of all scripts, a parse check of `config.yaml`, and a check that
`data/legionaer_training_data.json` matches `src/01_dataset.py`. Training itself is not run in CI (no GPU).

## Where to change what

| Change | File |
|---|---|
| Training examples | `TRAINING_DATA` in [`src/01_dataset.py`](../src/01_dataset.py), then run it |
| Hyperparameters, base model, paths | [`config.yaml`](../config.yaml) |
| Prompt format | `load_dataset()` in [`src/02_train.py`](../src/02_train.py) and `generate_answer()` in [`src/03_inference.py`](../src/03_inference.py) |
| Test questions | `run_test_questions()` in `src/03_inference.py` |
| Hub repository name | `repo_name` in [`hf_login.py`](../hf_login.py) and in the notebook's step 8 |

Code, comments and output are in English; the training data and test questions are in German, because the persona
answers in German.

## Smoke test without a GPU

The training and inference code paths can be checked on a CPU with a tiny random model instead of Mistral-7B. Load
`src/02_train.py` and `src/03_inference.py` with `importlib`, set `base_model_id` to
`hf-internal-testing/tiny-random-MistralForCausalLM`, `use_4bit` to `false` and `optimizer` to `adamw_torch`
(the paged optimiser needs bitsandbytes with CUDA), then call `load_model_and_tokenizer()`, `add_lora_adapters()`,
`load_dataset()`, `train()`, `load_model_with_lora()` and `generate_answer()`. The answers are random tokens; the test
only shows that the APIs fit together.
