# Hugging Face Hub

The training saves only the LoRA adapters (a few MB). To share the model, upload the adapters; users load the base
model from the Hub and put the adapters on top.

## Token

Create a token with **Write** access at <https://huggingface.co/settings/tokens>. Either log in once:

```bash
hf auth login
```

or put it into `.env` (created by `make setup` from [`.env.example`](../.env.example)):

```bash
HF_TOKEN=hf_...
```

## Upload

```bash
python hf_login.py upload
```

[`hf_login.py`](../hf_login.py) uses the token from `HF_TOKEN`/`.env` or an earlier login (otherwise it asks for one),
creates the repository `<your-username>/legionaer-scrum-master` if it does not exist yet (public by default) and uploads
`models/legionaer-final/`. Change `repo_name` in `upload_model_to_hf()` for a different name. The notebook's step 8
does the same, but creates a **private** repository.

Alternatively, export the adapters together with the tokenizer (option 3 in `src/03_inference.py`) and upload the
export with the `hf` CLI:

```bash
hf upload <your-username>/legionaer-scrum-master models/legionaer-deployed
```

## Use the uploaded adapters

```python
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

base_model_id = "mistralai/Mistral-7B-Instruct-v0.3"
tokenizer = AutoTokenizer.from_pretrained(base_model_id)
model = AutoModelForCausalLM.from_pretrained(base_model_id, device_map="auto")
model = PeftModel.from_pretrained(model, "<your-username>/legionaer-scrum-master")
```

Then generate with the prompt format `[INST] {question} [/INST]`, as in `generate_answer()` in
[`src/03_inference.py`](../src/03_inference.py).

The base model is published by Mistral AI under the Apache 2.0 license and can be downloaded without a token.
