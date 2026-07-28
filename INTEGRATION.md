# 🌍 INTEGRATION GUIDE: HuggingFace, Kaggle, GitHub

Hier steht wie du dein Projekt mit externen Diensten verbindest.

## 🤗 HuggingFace Integration

### 1. Account & Token

```bash
# Gehe zu https://huggingface.co/join
# Dann: Settings → Access Tokens → New token

# In Terminal:
huggingface-cli login
# Gib deinen Token ein
```

### 2. Model zu Hub uploaden

```bash
# Nach dem Training:
huggingface-cli upload \
  ./models/legionaer-final \
  <dein-username>/legionaer-scrum-master

# Oder mit Python:
python hf_login.py upload
```

### 3. Model Card erstellen

Auf https://huggingface.co/<username>/legionaer-scrum-master:

```markdown
# Legionär Scrum Master

Ein LoRA-adaptiertes Mistral-7B Modell mit römischem Legionär-Charakter!

## Model Details

- **Base Model**: Mistral-7B-Instruct-v0.3
- **LoRA Rank**: 16
- **Training Examples**: 25
- **Training Time**: ~20 min on T4 GPU

## Usage

```python
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

base_model = AutoModelForCausalLM.from_pretrained("mistralai/Mistral-7B-Instruct-v0.3")
model = PeftModel.from_pretrained(base_model, "<username>/legionaer-scrum-master")
tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-Instruct-v0.3")

prompt = "<s>[INST] Was ist ein Sprint? [/INST]"
inputs = tokenizer(prompt, return_tensors="pt")
outputs = model.generate(**inputs, max_new_tokens=256)
print(tokenizer.decode(outputs[0]))
```

## Example Outputs

**Q: Was ist ein Sprint?**
A: Ein Sprint, mein Freund, ist wie ein Feldzug unserer Legionen! ...

**Q: Wie führe ich ein Daily Standup?**
A: Das tägliche Standup ist die Morgenappell unserer Legion! ...
```

---

## 📊 Kaggle Integration

### 1. Account erstellen

https://www.kaggle.com/nicolashuber (bereits vorhanden!)

### 2. Kaggle Notebook erstellen

```
1. Gehe zu https://kaggle.com/code
2. Klicke "+ New Notebook"
3. Wähle "Python" als Kernel
4. Aktiviere GPU T4 in Notebook-Einstellungen
```

### 3. Code im Notebook

```python
# Cell 1: Installation
!pip install transformers peft trl bitsandbytes datasets torch

# Cell 2: Datensatz
from src.dataset import TRAINING_DATA
import json
with open("training_data.json", "w") as f:
    json.dump(TRAINING_DATA, f)

# Cell 3: Training
%run src/02_train.py

# Cell 4: Inference
%run src/03_inference.py
```

### 4. Notebook veröffentlichen

- Klicke "Save Version"
- Wähle "Public"
- Teile den Link!

---

## 🐙 GitHub Integration

### 1. Repository erstellen

```bash
# Auf https://github.com/new

# Lokal:
git init
git add .
git commit -m "Initial commit: Legionär Scrum Master Fine-Tuning"
git branch -M main
git remote add origin https://github.com/<username>/finetune-llm.git
git push -u origin main
```

### 2. GitHub Actions (CI/CD)

`.github/workflows/test.yml` ist bereits vorhanden!

Bei jedem Push:
- Lädt die Dependencies
- Testet den Python-Code
- Prüft die Syntax
- Linting mit pylint

### 3. GitHub Pages (optional)

Um eine Website zu hosten:

```bash
# Erstelle Dokumentation
mkdir docs
# ... Füge Dokumentation ein ...

# In GitHub Settings → Pages
# Source: Deploy from a branch
# Branch: main, Folder: /docs
```

### 4. Releases

```bash
# Nach erfolgreichem Training:
git tag -a v1.0 -m "Initial trained model release"
git push origin v1.0
```

Dann auf GitHub:
- Gehe zu Releases
- Klicke "Create Release"
- Lade dein Modell hoch (oder link zu HuggingFace)

---

## 🔗 Volle Integration (Advanced)

### Setup für automatisches Training auf Kaggle

```bash
# In .github/workflows/kaggle-train.yml:
name: Auto Train on Kaggle
on:
  schedule:
    - cron: "0 2 * * 0"  # Wöchentlich Sonntag 2:00 UTC

jobs:
  train:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - name: Setup Kaggle
        run: |
          mkdir ~/.kaggle
          echo "${{ secrets.KAGGLE_CONFIG }}" > ~/.kaggle/kaggle.json
      - name: Run Kaggle Kernel
        run: |
          kaggle kernels push -p .
          kaggle kernels output nicolashuber/legionaer-training
```

### Automatisches Upload zu HuggingFace

```bash
# In .github/workflows/deploy-hf.yml
name: Deploy to HuggingFace
on:
  release:
    types: [created]

jobs:
  deploy:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - name: Install HF CLI
        run: pip install huggingface-hub
      - name: Upload to Hub
        env:
          HF_TOKEN: ${{ secrets.HF_TOKEN }}
        run: |
          huggingface-cli login --token $HF_TOKEN --add-to-git-credential
          huggingface-cli upload ./models/legionaer-final \
            nicolashuber/legionaer-scrum-master
```

---

## 🔐 Secrets & Environment Variables

### GitHub Secrets

1. Gehe zu Repository Settings → Secrets and variables
2. Klicke "New repository secret"
3. Füge hinzu:

```
HF_TOKEN = your_hf_token
KAGGLE_USERNAME = nicolashuber
KAGGLE_KEY = your_kaggle_key
```

### .env Local

```bash
# .env (NICHT ins GitHub!)
HF_TOKEN=hf_xxxxx
KAGGLE_USERNAME=nicolashuber
KAGGLE_KEY=xxxxx
```

---

## 📱 Soziale Integration

### Twitter Integration

Nach erfolgreichem Training:

```python
from tweepy import Client

client = Client(bearer_token=TWITTER_TOKEN)
client.create_tweet(
    text="""
🦅 Gerade ein LoRA-Modell mit einem römischen Legionär Scrum Master trainiert!

Das Modell beantwortet jetzt Agile/Scrum Fragen im Stil eines Legionärs.

📊 25 Trainingsbeispiele
⏱️ 20 min auf T4 GPU
📍 Verfügbar auf HuggingFace

Für Rom! ⚔️
#LLM #LoRA #Mistral #FaceTech
    """
)
```

---

## 🔄 Workflow-Zusammenfassung

```
1. Code lokal schreiben
   ↓
2. git push → GitHub
   ↓
3. GitHub Actions laufen Tests
   ↓
4. Bei Release: Automatisches Upload zu HuggingFace
   ↓
5. Modell verfügbar für die Community
   ↓
6. Feedback & Verbesserungen
```

---

## 📚 Link-Sammlung

**HuggingFace:**
- Account: https://huggingface.co/nicolashuber
- Token: https://huggingface.co/settings/tokens
- Hub Upload: https://huggingface.co/docs/hub/upload

**Kaggle:**
- Account: https://kaggle.com/nicolashuber
- Notebooks: https://kaggle.com/code
- API: https://kaggle.com/settings/account

**GitHub:**
- Repository: https://github.com/HuberNicolas/finetune-llm
- Actions Docs: https://docs.github.com/en/actions

---

**Alles verbunden! 🌐 Für Rom!**
