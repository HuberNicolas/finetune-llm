# Google Colab and Kaggle

Training needs an NVIDIA GPU with about 16 GB of VRAM. The free T4 GPUs on Google Colab and Kaggle are enough.

## Google Colab

1. Open [`colab_legionaer_finetuning.ipynb`](../colab_legionaer_finetuning.ipynb) in Colab with the badge at its top,
   or via **File → Open notebook → GitHub** and `HuberNicolas/finetune-llm`.
2. **Runtime → Change runtime type → T4 GPU → Save**.
3. Run the cells from top to bottom:

| Step | What it does |
|---|---|
| 1 | Checks the GPU |
| 2 | Installs the libraries in the versions from `pixi.lock` (PyTorch is preinstalled) |
| 3 | Clones the repository to `/content/finetune-llm` |
| 4 | Writes the training data and shows the number of examples |
| 5 | Runs `src/02_train.py` (about 20–30 minutes) |
| 6 | Lists the saved adapters and checkpoints |
| 7 | Loads the model with the adapters and answers the test questions |
| 8 | Optional: uploads the adapters to a private model repository on the Hugging Face Hub |

Files in `/content` are deleted when the runtime stops. Upload the adapters (step 8) or download
`models/legionaer-final/` before you close the session.

## Kaggle

Not tested. The scripts run on any machine with a CUDA GPU, so a Kaggle notebook with the GPU accelerator enabled
should work with these cells:

```python
!git clone https://github.com/HuberNicolas/finetune-llm
%cd finetune-llm
!pip install -q -r requirements-kaggle.txt
!python src/01_dataset.py
!python src/02_train.py
```

## Save notebook changes to GitHub

To keep changes you made to the notebook in Colab, use **File → Save a copy in GitHub**. Colab asks for access to your
GitHub account and commits the notebook to the repository and branch you choose.

For changes to other files in the cloned repository, commit them in a notebook cell and push with a
[personal access token](https://github.com/settings/tokens) (fine-grained, *Contents: read and write* on your fork).
Do not paste the token into a cell that you save; use Colab's **Secrets** panel (🔑) and read it with
`google.colab.userdata.get(...)`.
