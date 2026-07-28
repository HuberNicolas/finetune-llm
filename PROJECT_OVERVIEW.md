# 🦅 PROJEKTÜBERSICHT - Legionär Scrum Master Fine-Tuning

Willkommen zum kompletten Fine-Tuning Projekt für einen römischen Legionär Scrum Master!

## 📋 Was ist in diesem Projekt?

```
🎓 LEARNING FOCUS - Alles ist kommentiert und erklärt!
├── 01_dataset.py       - Trainingsdaten generieren (25 Beispiele)
├── 02_train.py         - Fine-Tuning mit LoRA (~20 min auf T4)
├── 03_inference.py     - Modell testen & Chat
└── Ausführliche Kommentare auf Deutsch!

💻 TECHNOLOGIE STACK
├── PyTorch (torch)           - Berechnungs-Engine
├── HuggingFace (transformers) - Modelle & Pipeline
├── PEFT (peft)               - LoRA Fine-Tuning
├── TRL (trl)                 - SFTTrainer
└── BitsAndBytes (4-bit)      - RAM-Optimierung

☁️ CLOUD-READY
├── Google Colab (kostenlos!)
├── Kaggle Notebooks (kostenlos!)
├── HuggingFace Hub (für Deployment)
└── GitHub (für Versionskontrolle)

📚 DOKUMENTATION
├── README.md           - Komplettes Handbuch
├── QUICKSTART.md       - 5-Minuten Start
├── DEPLOYMENT.md       - 6 Deployment-Optionen
└── INTEGRATION.md      - GitHub, Kaggle, HuggingFace
```

## 🚀 START IN 5 SCHRITTEN

### 1️⃣ Repo klonen
```bash
git clone https://github.com/HuberNicolas/finetune-llm.git
cd finetune-llm
```

### 2️⃣ Dependencies installieren
```bash
# Mit Pixi (empfohlen)
pixi install
pixi shell

# Oder mit pip
pip install -r requirements-kaggle.txt
```

### 3️⃣ Trainingsdaten generieren
```bash
python src/01_dataset.py
```
Output: `data/legionaer_training_data.json` (25 Beispiele)

### 4️⃣ Training starten (braucht GPU!)
```bash
python src/02_train.py
```
Dauer: ~20 min auf T4, ~5 min auf A100, 🔴 nicht auf CPU!

### 5️⃣ Testen!
```bash
python src/03_inference.py
```
Dann: Wähle Option "1" für interaktive Chat-Session!

---

## 🖥️ Wo trainieren?

### Option 1: Google Colab ⭐ (EMPFOHLEN)
- **Kostenlos**: Ja!
- **GPU**: NVIDIA T4 (12GB) - perfekt für 7B Modelle
- **Setup**: Keine Installation nötig
- **Dauer**: ~20 Minuten
- **Link**: https://colab.research.google.com/

```python
# In Colab direkt:
!git clone https://github.com/HuberNicolas/finetune-llm
%cd finetune-llm
!python src/01_dataset.py
!python src/02_train.py
```

### Option 2: Kaggle
- **Kostenlos**: Ja! (30h GPU/Woche)
- **GPU**: T4 oder P100
- **Link**: https://kaggle.com/nicolashuber

### Option 3: Lokale GPU
- **Kostenlos**: Falls du eine GPU hast 😄
- **GPU**: NVIDIA RTX 3060+ (8GB+)
- **Dauer**: Abhängig von deiner Hardware

### Option 4: HuggingFace Spaces
- **Kostenlos**: Ja!
- **GPU**: A10 (24GB)
- **Setup**: Einfach
- **Dauer**: ~5 Minuten

---

## 📂 Datei-Übersicht

| Datei | Zweck |
|-------|-------|
| `pixi.toml` | Abhängigkeiten (Pixi-Format) |
| `requirements-kaggle.txt` | Abhängigkeiten (pip-Format) |
| `src/01_dataset.py` | Generiere Trainingsdaten |
| `src/02_train.py` | **HAUPTDATEI** - Das Training |
| `src/03_inference.py` | Teste das trainierte Modell |
| `config.yaml` | Zentrale Konfiguration |
| `hf_login.py` | HuggingFace Upload-Helper |
| `setup.py` | Projekt-Setup |
| `Makefile` | Hilfreiche Make-Befehle |
| `.github/workflows/test.yml` | GitHub CI/CD |
| `README.md` | Komplettes Manual |
| `QUICKSTART.md` | Schneller Start |
| `DEPLOYMENT.md` | 6 Deployment-Optionen |
| `INTEGRATION.md` | GitHub/Kaggle/HF Integration |

---

## 🎯 Wie funktioniert das Projekt?

### Das Problem
Mistral-7B ist ein großes Modell (7 Milliarden Parameter).
Training von Grund auf braucht Wochen und kostet tausende Euro.

### Die Lösung: LoRA (Low-Rank Adaptation)

```
Statt:    Trainiere alle 7B Parameter ❌ (Zu teuer & langsam)

Besser:   Trainiere nur kleine "Adapter" (~1% der Parameter) ✅
```

**Was passiert:**
```
Schritt 1: Lade Basis-Mistral (7B Parameter, EINGEFROREN)
Schritt 2: Füge kleine LoRA-Adapter hinzu (~1MB)
Schritt 3: Trainiere NUR die Adapter mit deinen Beispielen
Schritt 4: Nach ~20 min: Fertig! 🎉
```

**Resultat:**
- Das Modell hat den Basis-Wissen von Mistral
- + Die neue Persönlichkeit vom Training
- = Legionär Scrum Master! 🦅

---

## 💡 Wie man dieses Projekt erweitert

### Variante 1: Andere Persönlichkeit trainieren

Ändere die Trainingsbeispiele in `01_dataset.py`:

```python
TRAINING_DATA = [
    {
        "instruction": "Was ist ein Sprint?",
        "output": "Deine neue Antwort hier..."
    },
    # ... mehr Beispiele
]
```

### Variante 2: Anderes Basis-Modell

In `02_train.py`:
```python
MODELL_ID = "meta-llama/Llama-2-7b-chat-hf"  # oder eine andere
```

### Variante 3: Länger trainieren

In `02_train.py`:
```python
"num_epochs": 5,  # Statt 3 (länger = besser, aber Overfitting-Risiko)
```

### Variante 4: Mehr LoRA Rank

In `02_train.py`:
```python
"r": 32,  # Statt 16 (größer = mehr Parameter, aber auch RAM)
```

---

## 🌍 Deployment-Optionen

Nach dem Training kannst du das Modell:

1. **HuggingFace Hub uploaden** - Die einfachste Lösung!
2. **FastAPI Server** - REST API
3. **Streamlit App** - Web UI
4. **Docker** - Container
5. **Ollama** - Lokal laufen
6. **HuggingFace Spaces** - Kostenlos hosten

Siehe `DEPLOYMENT.md` für Details!

---

## 🔗 Ressourcen & Links

**Wichtige Links:**
- 🤗 [HuggingFace Hub](https://huggingface.co)
- 📘 [Transformers Dokumentation](https://huggingface.co/docs/transformers)
- 🎯 [PEFT/LoRA](https://github.com/huggingface/peft)
- 🦅 [Mistral Modell](https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.3)
- 📊 [Kaggle](https://kaggle.com)
- 🐙 [GitHub](https://github.com)

**Meine Accounts:**
- HuggingFace: https://huggingface.co/NicolasHuber
- Kaggle: https://kaggle.com/nicolashuber
- GitHub: https://github.com/HuberNicolas/

---

## 📚 Lernpfad

### Level 1: Verstehen (2-3 Stunden)
- [ ] Lese README.md
- [ ] Schaue auf die Code-Kommentare
- [ ] Führe `01_dataset.py` aus
- [ ] Verstehe die Trainingsbeispiele
- [ ] Führe Training auf Colab durch (~20 min)
- [ ] Teste das Modell mit `03_inference.py`

### Level 2: Experimentieren (2-4 Stunden)
- [ ] Ändere die Trainingsbeispiele
- [ ] Trainiere mit anderen Hyperparametern
- [ ] Teste verschiedene Basis-Modelle
- [ ] Generiere mehr Trainingsbeispiele
- [ ] Vergleiche Qualität bei unterschiedlichen Settings

### Level 3: Produktionalisieren (1-2 Tage)
- [ ] Deploye zu HuggingFace Hub
- [ ] Erstelle eine FastAPI
- [ ] Baue eine Streamlit UI
- [ ] Deploye zu HuggingFace Spaces
- [ ] Teile mit der Community!

---

## ❓ Häufige Fragen

**Q: Kann ich das ohne GPU machen?**
A: Technisch ja, aber es dauert Stunden. GPU ist stark empfohlen!

**Q: Brauche ich ein HuggingFace Account?**
A: Nicht zum Training. Nur wenn du das Modell uploaden möchtest.

**Q: Wie viel kostet das?**
A: Komplett kostenlos mit Colab/Kaggle!

**Q: Kann ich meine eigenen Daten verwenden?**
A: JA! Das ist genau der Punkt! 😄

**Q: Kann ich das Modell offline laufen?**
A: JA! Mit Ollama oder einer ähnlichen App.

---

## 🎓 Lernziele

Nach diesem Projekt wirst du verstehen:

✅ Wie moderne LLMs funktionieren
✅ Was LoRA ist und warum es nützlich ist
✅ Wie man Fine-Tuning macht
✅ Wie man GPUs nutzt
✅ Wie man Modelle deployt
✅ Wie man mit HuggingFace arbeitet
✅ Wie man Code auf GitHub versioniert
✅ Wie man auf Cloud-Plattformen trainiert

---

## 🎉 Fertig!

Du hast jetzt alles was du brauchst um:
1. Ein großes Sprachmodell zu verstehen
2. Es mit eigenen Daten zu trainieren
3. Mit wenig Ressourcen zu arbeiten (LoRA)
4. Das Modell in die Welt zu bringen

**Viel Spaß beim Fine-Tuning! 🦅**

---

*Projekt erstellt mit ❤️ für Learning & Understanding*

*"Für Rom!" - Marcus Productivus Maximus* ⚔️
