# Troubleshooting

| Problem | Cause and fix |
|---|---|
| `NO GPU AVAILABLE` | Training needs a CUDA GPU. On Colab: **Runtime → Change runtime type → T4 GPU**. On Kaggle: **Settings → Accelerator → GPU**. |
| `CUDA out of memory` during training | Lower `training.batch_size` in `config.yaml` (and raise `gradient_accumulation_steps` to keep the effective batch size), or lower `max_seq_length`. |
| `CUDA out of memory` when testing in the notebook | The training process has ended, but another cell may still hold a model. Restart the runtime, run steps 2–3, then step 7. |
| `TypeError: … unexpected keyword argument 'max_seq_length'` or `'tokenizer'` | Libraries too old or too new for the code. Install the versions from `requirements-kaggle.txt`. |
| `LoRA adapters not found: models/legionaer-final` | Run `src/02_train.py` first. On Colab, files are gone after the runtime restarts. |
| Download of the base model fails | The model is ~15 GB. Check the internet connection and the free disk space; the download resumes when you start again. |
| `No Hugging Face token found` | Run `hf auth login` or set `HF_TOKEN` in `.env`. |
| Upload fails with 401/403 | The token needs **Write** access. |
| Answers are in the normal assistant style | 17 examples are very few. Add more examples or train for more epochs, and compare with the base model. |
| Package installation on Colab times out | Run the cell again. |
