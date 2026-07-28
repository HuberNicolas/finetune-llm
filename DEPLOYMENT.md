# 🚀 DEPLOYMENT GUIDE

Hier sind verschiedene Wege um dein trainiertes Modell zu deployen.

## 🌍 Option 1: HuggingFace Hub (Empfohlen!)

Das ist die easiest Methode. Dein Modell wird in der Community verfügbar.

### Schritt 1: Account erstellen
https://huggingface.co/join

### Schritt 2: Access Token generieren
1. Gehe zu https://huggingface.co/settings/tokens
2. Klicke "New token"
3. Gib einen Namen ein (z.B. "legionaer-upload")
4. Wähle "Write" als Rolle
5. Kopiere den Token

### Schritt 3: Login & Upload

```bash
python hf_login.py upload
```

Dann:
```bash
huggingface-cli upload ./models/legionaer-final <dein-username>/legionaer-scrum-master
```

### Schritt 4: Share mit der Welt!

Dein Modell ist jetzt verfügbar unter:
```
https://huggingface.co/<dein-username>/legionaer-scrum-master
```

Jeder kann es jetzt nutzen:
```python
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

base_model = "mistralai/Mistral-7B-Instruct-v0.3"
lora_model = "<dein-username>/legionaer-scrum-master"

model = AutoModelForCausalLM.from_pretrained(base_model)
model = PeftModel.from_pretrained(model, lora_model)
```

---

## 🐳 Option 2: Docker Container

Deploye dein Modell in einem Docker Container.

### Erstelle ein Dockerfile

```dockerfile
FROM nvidia/cuda:11.8.0-runtime-ubuntu22.04

WORKDIR /app

# Installiere Python & Dependencies
RUN apt-get update && apt-get install -y python3 python3-pip
COPY requirements-kaggle.txt .
RUN pip install -r requirements-kaggle.txt

# Kopiere Modell-Dateien
COPY ./models/legionaer-final ./models/legionaer-final
COPY ./src ./src

# Starte API
EXPOSE 8000
CMD ["python3", "src/fastapi_app.py"]
```

### Build & Run

```bash
docker build -t legionaer-scrum-master .
docker run --gpus all -p 8000:8000 legionaer-scrum-master
```

---

## 🌐 Option 3: FastAPI Server

Erstelle einen REST API Server für dein Modell.

### fastapi_app.py

```python
from fastapi import FastAPI
from pydantic import BaseModel
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

app = FastAPI()

# Lade Modell beim Start
model = AutoModelForCausalLM.from_pretrained("mistralai/Mistral-7B-Instruct-v0.3")
model = PeftModel.from_pretrained(model, "./models/legionaer-final")
tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-Instruct-v0.3")

class Question(BaseModel):
    question: str

@app.post("/generate")
async def generate(item: Question):
    prompt = f"<s>[INST] {item.question} [/INST]"
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            input_ids=inputs["input_ids"],
            max_new_tokens=256,
            temperature=0.7,
            top_p=0.9,
        )

    antwort = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return {"answer": antwort}

@app.get("/health")
async def health():
    return {"status": "ok"}
```

### Start

```bash
pip install fastapi uvicorn
uvicorn fastapi_app:app --host 0.0.0.0 --port 8000
```

### Test

```bash
curl -X POST "http://localhost:8000/generate" \
  -H "Content-Type: application/json" \
  -d '{"question": "Was ist ein Sprint?"}'
```

---

## 🎨 Option 4: Streamlit Web UI

Eine schöne Web-Oberfläche für dein Modell!

### streamlit_app.py

```python
import streamlit as st
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import torch

st.set_page_config(page_title="Legionär Scrum Master", layout="wide")

st.title("🦅 Legionär Scrum Master")
st.write("Ein römischer Legionär beantwortet deine Scrum-Fragen!")

# Lade Modell (mit Caching)
@st.cache_resource
def lade_modell():
    model = AutoModelForCausalLM.from_pretrained("mistralai/Mistral-7B-Instruct-v0.3")
    model = PeftModel.from_pretrained(model, "./models/legionaer-final")
    tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-Instruct-v0.3")
    return model, tokenizer

model, tokenizer = lade_modell()

# Input
frage = st.text_input("Stelle eine Frage:")

if frage:
    st.write("🦅 Marcus denkt...\n")

    prompt = f"<s>[INST] {frage} [/INST]"
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            input_ids=inputs["input_ids"],
            max_new_tokens=256,
            temperature=0.7,
            top_p=0.9,
        )

    antwort = tokenizer.decode(outputs[0], skip_special_tokens=True)
    st.write(f"**Legionär Marcus:**\n\n{antwort}")
```

### Run

```bash
pip install streamlit
streamlit run streamlit_app.py
```

---

## ☁️ Option 5: Kaggle Inference API

Deploye auf Kaggle als Inference API.

1. Erstelle ein Notebook mit Inference-Code
2. Klicke "⋯" → "Create a data connector"
3. Wähle "Python script"
4. Dein Modell wird als API verfügbar!

---

## 🌌 Option 6: HuggingFace Spaces

Deploye deine Streamlit/Gradio App kostenlos auf HF Spaces!

### 1. Erstelle einen Space

https://huggingface.co/spaces

### 2. Wähle "Streamlit" als SDK

### 3. Upload die App

```bash
git clone https://huggingface.co/spaces/<dein-username>/legionaer
cd legionaer
# Kopiere streamlit_app.py hier hin
git add streamlit_app.py
git commit -m "Add legionaer app"
git push
```

### 4. Space lädt automatisch! 🚀

---

## 🔑 Environment Variables

Wichtig: Speichere sensitive Daten NICHT im Code!

### .env Datei

```
HF_TOKEN=hf_xxxxxxx
KAGGLE_USERNAME=nicolashuber
KAGGLE_KEY=xxxxx
```

### Python

```python
from dotenv import load_dotenv
import os

load_dotenv()
hf_token = os.getenv("HF_TOKEN")
```

---

## 📊 Performance Tipps

Für bessere Performance:

### 1. Merge LoRA zu Basis-Modell

```python
model = model.merge_and_unload()
model.save_pretrained("./models/legionaer-merged")
```

Vorteil: Schneller, weil keine LoRA-Operationen mehr.
Nachteil: Größer (~30GB statt ~1MB für LoRA)

### 2. Quantisierung

Nutze 4-bit Quantisierung für RAM-Sparen (siehe 02_train.py)

### 3. Batch Processing

Verarbeite mehrere Anfragen gleichzeitig:

```python
prompts = [
    "<s>[INST] Frage 1 [/INST]",
    "<s>[INST] Frage 2 [/INST]",
]
inputs = tokenizer(prompts, return_tensors="pt", padding=True)
outputs = model.generate(input_ids=inputs["input_ids"])
```

---

## ✅ Deployment Checkliste

- [ ] Modell auf HuggingFace Hub hochgeladen
- [ ] README mit Nutzungsbeispielen
- [ ] .env ist in .gitignore (keine Secrets im Git!)
- [ ] Teste dein Modell in der Production-Umgebung
- [ ] Documentiere die API/UI
- [ ] Teile mit der Community!

---

**Viel Erfolg beim Deployment! 🦅**
