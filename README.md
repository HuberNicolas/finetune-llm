# 🦅 Legionär Scrum Master - Fine-Tuning Projekt

Ein vollständiges Fine-Tuning-Projekt für einen **römischen Legionär Scrum Master** mit Mistral-7B.

## 📚 Was ist dieses Projekt?

Dieses Projekt zeigt wie du ein großes Sprachmodell (Mistral-7B) mit nur **~50 Trainingsbeispielen** auf eine spezifische Persönlichkeit trainierst (einen römischen Legionär, der Scrum Master ist).

**Kern-Technologien:**
- 🔦 **torch** - Die Berechnungs-Engine
- 🤗 **transformers** - HuggingFace Modelle
- ⚡ **bitsandbytes** - 4-bit Quantisierung (RAM-Sparen)
- 🎯 **peft** - LoRA (trainiert nur ~1% der Parameter!)
- 🏋️ **trl** - SFTTrainer (der Trainer)

## ⚡ Quick Start

### 1. Umgebung Setup mit Pixi

```bash
# Pixi installieren (falls noch nicht installiert)
# Siehe: https://prefix.dev/

# Pixi Environment erstellen
pixi install

# Environment aktivieren
pixi shell
```

### 2. Trainingsbeispiele generieren

```bash
python src/01_dataset.py
```

Das erstellt `data/legionaer_training_data.json` mit 25 Trainingsbeispielen.

### 3. Modell trainieren

```bash
python src/02_train.py
```

**Dauer:**
- Mit GPU (T4): ~20-30 Minuten
- Mit CPU: ~3+ Stunden (nicht empfohlen)

Das trainierte Modell wird in `models/legionaer-final/` gespeichert.

### 4. Inferenz & Testing

```bash
python src/03_inference.py
```

Dann:
1. Wähle "Chat-Session" für interaktive Tests
2. Oder "Test-Fragen" um vordefinierte Fragen durchzulaufen

## 🚀 GPU-Zugriff für Anfänger

### Option 1: Google Colab (Kostenlos!)

```bash
# In einer neuen Colab Zelle:
!git clone https://github.com/HuberNicolas/finetune-llm
%cd finetune-llm

# Pixi im Colab aktivieren
!pip install pixi-project

# Danach einfach die Python-Dateien ausführen
!python src/01_dataset.py
!python src/02_train.py
!python src/03_inference.py
```

**Vorteile:**
- Kostenlose T4 GPU (12 GB VRAM)
- Keine Installation nötig
- Perfekt für Anfänger

### Option 2: Kaggle (Kostenlos mit GPU)

1. Erstelle einen Account: https://www.kaggle.com/nicolashuber
2. Forke dieses Notebook oder erstelle ein Neues
3. Aktiviere "GPU T4" in Notebook-Einstellungen
4. Führe den Code aus

```python
# Kaggle-Zelle 1: Installiere Abhängigkeiten
!pip install transformers peft trl bitsandbytes datasets torch

# Kaggle-Zelle 2: Klone das Repo
!git clone https://github.com/HuberNicolas/finetune-llm
%cd finetune-llm

# Kaggle-Zelle 3: Trainiere
%run src/01_dataset.py
%run src/02_train.py
```

### Option 3: HuggingFace Spaces

Kommt bald! Hier kannst du eine Web-UI für Inference deployen.

## 📂 Projektstruktur

```
finetune-llm/
├── pixi.toml                           # Abhängigkeiten (Pixi-Format)
├── .gitignore                          # Git-Ignores
├── README.md                           # Diese Datei
│
├── src/
│   ├── 01_dataset.py                   # Generiere Trainingsdaten
│   ├── 02_train.py                     # Fine-Tune mit LoRA
│   └── 03_inference.py                 # Teste das trainierte Modell
│
├── data/
│   └── legionaer_training_data.json    # ~50 Trainingsbeispiele
│
├── models/
│   ├── legionaer-sft/                  # Training Checkpoints
│   ├── legionaer-final/                # Finale LoRA-Adapter
│   └── legionaer-deployed/             # Exportiertes Modell
│
└── notebooks/
    └── legionaer_tutorial.ipynb        # Jupyter Notebook Tutorial
```

## 🎓 Wie funktioniert Fine-Tuning? (Vereinfacht)

### Das Problem
Mistral-7B ist ein großes Modell mit 7 Milliarden Parametern. Alles neu zu trainieren würde Wochen dauern und Tausende von Euro kosten.

### Die Lösung: LoRA (Low-Rank Adaptation)

Statt alle 7B Parameter zu trainieren, fügen wir kleine **"Adapter-Matrizen"** ein:

```
Original Gewicht:    [4096 × 4096] = ~16M Parameter (eingefroren!)
LoRA Adapter:        [4096 × 16] × [16 × 4096] = ~131K Parameter ← trainieren!

Ergebnis: 99% weniger zu trainieren! 🚀
```

### Was passiert beim Training?

1. **Datensatz laden**: 50 Fragen + römische Antworten
2. **Forward Pass**: Modell liest die Frage
3. **Loss berechnen**: Wie weit war die Vorhersage von der wahren Antwort weg?
4. **Backpropagation**: Berechne Gradienten für die LoRA-Adapter
5. **Update**: Adapter werden minimal angepasst (learning_rate = 0.0002)
6. **Repeat**: 50 Beispiele × 3 Epochen = 150 Lernschritte

Nach ~30 Minuten hat das Modell gelernt: "Diese Fragen sollen römisch-dramatisch beantwortet werden!" 🦅

## 📊 Hyperparameter Erklärung

Diese sind in `02_train.py` definiert:

| Parameter | Wert | Erklärung |
|-----------|------|-----------|
| `r` (Rank) | 16 | Breite der LoRA-Adapter. 8 = schneller, 64 = besser |
| `lora_alpha` | 32 | Scaling-Faktor. Meist 2× des Rank |
| `learning_rate` | 2e-4 | Wie große Lernschritte? Klein = stabil, Groß = schnell |
| `num_train_epochs` | 3 | Wie oft durchs Datensatz? 3 = ideal für kleine Datensätze |
| `per_device_batch_size` | 4 | Wieviele Beispiele parallel? 4 passt auf T4 GPU |
| `warmup_ratio` | 0.1 | Erst langsam aufwärmen (10% der Steps) |

## 🌍 Upload zu HuggingFace

Damit andere dein Modell nutzen können:

```bash
# 1. HuggingFace Login
huggingface-cli login
# Gib dein Token ein: https://huggingface.co/settings/tokens

# 2. Repo erstellen
huggingface-cli repo create legionaer-scrum-master

# 3. Clone und Upload
huggingface-cli upload ./models/legionaer-deployed <username>/legionaer-scrum-master

# Danach kann jeder dein Modell nutzen:
# from transformers import pipeline
# pipe = pipeline("text-generation", model="<username>/legionaer-scrum-master")
```

## 📤 GitHub Integration

```bash
# 1. Repository auf GitHub erstellen
# https://github.com/new

# 2. Klone dieses Repo und push zu deinem
git clone https://github.com/HuberNicolas/finetune-llm
cd finetune-llm

git remote set-url origin https://github.com/<dein-username>/legionaer-scrum-master
git push -u origin main
```

## 🎯 Nächste Schritte

### Level 1: Verstehen
- [ ] Lese den Code und die Kommentare
- [ ] Führe `01_dataset.py` aus und schaue die Trainingsbeispiele an
- [ ] Führe `02_train.py` auf Colab/Kaggle aus (20 Minuten)
- [ ] Teste das Modell mit `03_inference.py`

### Level 2: Experimentieren
- [ ] Ändere die Trainingsbeispiele (eigene Persönlichkeit trainieren!)
- [ ] Experimentiere mit LoRA-Hyperparametern
- [ ] Trainiere länger (mehr Epochen)
- [ ] Teste verschiedene Modelle (Llama-2, Mistral-Large)

### Level 3: Produktionalisieren
- [ ] Deploye das Modell zu HuggingFace Hub
- [ ] Erstelle eine FastAPI für Inference
- [ ] Baue eine Web-UI (Streamlit/Gradio)
- [ ] Nutze das Modell in einer echten Anwendung

## 🔗 Ressourcen

- **HuggingFace Transformers**: https://huggingface.co/docs/transformers
- **PEFT (LoRA)**: https://github.com/huggingface/peft
- **TRL (Trainer)**: https://github.com/huggingface/trl
- **Mistral-7B**: https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.3
- **Pixi**: https://prefix.dev/

## 🦅 Über Marcus Productivus Maximus

Unser trainiertes Modell ist ein fiktiver römischer Legionär, der:
- 📜 Lateinische Ausdrücke benutzt
- ⚔️ Militärische Metaphern liebt
- 🎯 Agile/Scrum-Konzepte "erklärt"
- 🏛️ Dramatisch und ehrfurchtsvoll spricht
- 🔥 Mit Feuer und Verve antwortet

Beispiel-Antwort auf "Was ist ein Sprint?":

> Hark! Ein Sprint, mein Freund, ist wie ein Feldzug unserer Legionen! Zwanzig Tage der konzentrierten Kampfkraft – nicht länger, nicht kürzer. Wir marschieren mit klarer Mission, schlagen unser Ziel, und kehren triumphierend zurück. Für Rom! ⚔️

## ✨ Lizenz

Dieses Projekt ist MIT-lizensiert. Nutze es frei!

## 🤝 Beiträge

Findest du einen Bug oder hast eine Idee? Erstelle ein Issue oder einen Pull Request!

---

**Viel Spaß beim Fine-Tuning! 🦅 Für Rom!**

*Erstellt mit ❤️ für Learning & Understanding*
