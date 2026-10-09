# TODO

Open tasks. See also [Known issues](README.md#known-issues).

## 1. Clean-up

- [x] Remove the seven extra guides that repeated the README (backed up outside the repository)
- [x] Translate code, notebook and scripts to English; training data stays German
- [x] Fix training and inference for trl 1.x and transformers 5.x
- [x] Make `config.yaml` valid and read it in the scripts
- [x] Move `setup.py` to `scripts/setup_project.py`
- [x] Add Ruff, replace the CI workflow, tidy `.gitignore`
- [x] Consolidate the documentation in the README and `docs/`
- [ ] Declare `psutil` in `pixi.toml` (it is only in the lockfile as a dependency of Jupyter); needs `pixi lock`

## 2. Verification

- [x] Smoke test of training, inference and export on a CPU with a tiny Mistral test model and the locked versions
- [ ] Full training run on a T4 GPU (Colab) with Mistral-7B, then update the example answer in the README with a
  real output of the fine-tuned model
- [ ] Run the Colab notebook from top to bottom

## 3. Publishing

- [x] Choose and add a license (MIT)
- [x] Check for secrets in the files and the git history
- [ ] Optional: upload the trained adapters to the Hugging Face Hub and link them in the README
