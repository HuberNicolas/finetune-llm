# 📋 QUICK START GUIDE

Alles was du wissen musst um zu starten!

## ⚡ 5-Minuten Start

### 1. Repository klonen

```bash
git clone https://github.com/HuberNicolas/finetune-llm.git
cd finetune-llm
```

### 2. Setup

```bash
# Verzeichnisse erstellen
python setup.py

# Mit Pixi (empfohlen)
pixi install
pixi shell

# Oder mit pip
pip install -r requirements-kaggle.txt
```

### 3. Trainingsbeispiele generieren

```bash
python src/01_dataset.py
```

Output:
```
✅ Datensatz gespeichert: data/legionaer_training_data.json
📊 Anzahl Trainingsbeispiele: 25
💪 Mit 3 Epochen = 75 Trainingschritte
```

### 4. Trainieren (braucht GPU!)

```bash
python src/02_train.py
```

**Dauer:**
- Mit GPU (T4): ~20 Minuten ✅
- Mit CPU: nicht praktikabel ❌

### 5. Testen

```bash
python src/03_inference.py
```

Wähle Option "1" für interaktive Chat-Session!

---

## 🖥️ GPU-Optionen

### Option A: Google Colab (KOSTENLOS!) 🚀

```python
# In Colab:
!git clone https://github.com/HuberNicolas/finetune-llm
%cd finetune-llm
!pip install -r requirements-kaggle.txt

# Dann einfach:
!python src/01_dataset.py
!python src/02_train.py
!python src/03_inference.py
```

### Option B: Kaggle Notebook

1. Gehe zu https://kaggle.com
2. Erstelle ein neues Notebook
3. Aktiviere "GPU T4" in den Einstellungen
4. Kopiere den Code von 01_dataset.py, 02_train.py, 03_inference.py

### Option C: Lokale GPU (falls vorhanden)

```bash
# Check GPU
python -c "import torch; print(torch.cuda.is_available())"

# Starten
pixi shell
python src/02_train.py
```

### Option D: CPU (nicht empfohlen, aber möglich)

```bash
# Wird ~3 Stunden dauern!
python src/02_train.py
```

---

## 📊 Dateistruktur verstehen

```
finetune-llm/
│
├── 01_dataset.py          ← Generiert die Trainingsdaten
├── 02_train.py            ← Das Haupttraining
├── 03_inference.py        ← Teste das Modell
│
├── pixi.toml              ← Abhängigkeiten (Pixi-Format)
├── requirements-*.txt     ← Abhängigkeiten (pip-Format)
│
├── config.yaml            ← Konfigurationsdatei
├── hf_login.py            ← HuggingFace Upload
├── setup.py               ← Projekt-Setup
│
├── data/                  ← Trainingsbeispiele
│   └── legionaer_training_data.json
│
└── models/                ← Trainierte Modelle
    ├── legionaer-sft/     ← Training Checkpoints
    ├── legionaer-final/   ← Finale LoRA-Adapter
    └── legionaer-deployed/← Für Deployment
```

---

## 🤔 Häufige Fragen

### F: Warum brauche ich eine GPU?

A: Das Modell hat 7 Milliarden Parameter. Training auf CPU dauert Tage.
Mit GPU (T4): 20 Minuten. Mit A100: 5 Minuten.

### F: Kann ich das Modell ohne Training nutzen?

A: Ja! Das Basis-Mistral-Modell funktioniert, ist aber nicht "römisch".
Mit dem Training bekommt es die Persönlichkeit des Legionärs.

### F: Wie viel kostet das?

A: **Kostenlos!**
- Colab: Kostenlos (mit Limits)
- Kaggle: Kostenlos (30h GPU/Woche)
- HuggingFace Spaces: Kostenlos (zum Deployen)

### F: Kann ich die Trainingsbeispiele ändern?

A: **JA!** Das ist der ganze Punkt! 😄
In `01_dataset.py` kannst du deine eigenen Beispiele hinzufügen.
Das Modell lernt dann einen anderen Stil/Charakter.

### F: Wie teile ich mein Modell?

A: Upload zu HuggingFace Hub:
```bash
python hf_login.py upload
```
Dann kann jeder dein Modell nutzen!

### F: Kann ich ein anderes Basis-Modell nutzen?

A: JA! In `02_train.py` ändere die `MODELL_ID`:
```python
MODELL_ID = "meta-llama/Llama-2-7b-chat-hf"  # oder eine andere
```

---

## 🛠️ Troubleshooting

### Problem: CUDA Out of Memory

**Lösung:**
- Reduziere `per_device_train_batch_size` von 4 zu 2
- Oder nutze 4-bit Quantisierung (ist bereits aktiviert)

### Problem: ModuleNotFoundError

**Lösung:**
```bash
pixi install
# oder
pip install -r requirements-kaggle.txt
```

### Problem: Model nicht gefunden

**Lösung:**
- Stelle sicher dass HuggingFace Internetzugang hat
- Oder lade das Modell manuell vorher:
```python
from transformers import AutoModelForCausalLM
model = AutoModelForCausalLM.from_pretrained("mistralai/Mistral-7B-Instruct-v0.3")
```

---

## 🎯 Nächste Schritte nach Training

### 1. Teste das Modell
```bash
python src/03_inference.py
```

### 2. Upload zu HuggingFace
```bash
python hf_login.py upload
```

### 3. Deploye die App (Optional)
```bash
# Streamlit
streamlit run app.py

# FastAPI
uvicorn fastapi_app:app --reload
```

### 4. Teile mit der Community
- Poste auf HuggingFace
- Teile auf Reddit/Twitter
- Zeige Freunden/Familie!

---

## 📚 Weitere Ressourcen

- [HuggingFace Docs](https://huggingface.co/docs/transformers)
- [PEFT/LoRA Tutorial](https://github.com/huggingface/peft)
- [TRL Docs](https://github.com/huggingface/trl)
- [Mistral Modell](https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.3)

---

## 💬 Fragen / Issues?

- Erstelle ein Issue auf GitHub
- Kommentiere auf dem HuggingFace Model
- Frag in der Community!

---

**Happy Fine-Tuning! 🦅 Für Rom!**
