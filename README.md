# 📰 Text Summarization System
### An AI-powered deep learning project that reads a long news article and returns a short, accurate summary.

---

## 📌 What Is This Project?

We are building an end-to-end AI system that takes a **long news article as input** and automatically generates a **concise 2–3 sentence summary** as output.

```
INPUT:  "The European Central Bank announced on Thursday that it would raise
         interest rates by 50 basis points, the largest increase in over a
         decade. The decision came amid mounting concerns over inflation across
         the eurozone, with prices rising at their fastest pace since the 1980s..."
         (500+ words)

OUTPUT: "The ECB raised interest rates by 50 basis points, its largest hike in
         a decade, in response to surging eurozone inflation."
         (1–2 sentences)
```

The AI model behind this is a **Large Language Model (LLM)** that has been
**fine-tuned** (specifically trained) on thousands of real news articles
so it learns what a good summary looks like.

---

## 🧑‍💻 Who Does What?

This project is split between **two people** with clearly defined roles:

| | Person A | Person B |
|---|---|---|
| **Role** | Data Engineer & Deployment | ML Engineer & Model Expert |
| **Responsible for** | Collecting & cleaning data, evaluating quality, building the web app | Choosing the model, fine-tuning it, deploying it as an API |
| **Guide file** | `PERSON_A_GUIDE.md` | `PERSON_B_GUIDE.md` |

---

## 🗂️ Project Folder Structure

```
text-summarization/
│
├── README.md                 ← You are here
├── PERSON_A_GUIDE.md         ← Full task guide for Person A
├── PERSON_B_GUIDE.md         ← Full task guide for Person B
│
├── data/
│   ├── raw/                  ← Original downloaded dataset
│   ├── cleaned/              ← After text cleaning
│   └── processed/
│       ├── train.jsonl       ← Training data (formatted for LLM)
│       ├── val.jsonl         ← Validation data
│       └── test.jsonl        ← Test data (for evaluation)
│
├── model/
│   ├── config.py             ← Model hyperparameters
│   ├── fine_tune.py          ← Fine-tuning script
│   ├── Modelfile             ← Ollama model definition
│   └── checkpoints/          ← Saved model weights
│
├── evaluation/
│   ├── evaluate.py           ← ROUGE scoring script
│   ├── baseline.py           ← Lead-3 baseline script
│   └── results/              ← Saved score files
│
├── app/
│   └── app.py                ← Streamlit web interface
│
└── predict.py                ← Inference: article → summary
```

---

## 🔄 How the Project Works — Step by Step

Here is the full pipeline from raw data to a working web app:

```
┌─────────────────────────────────────────────────────────────────┐
│                         FULL PIPELINE                           │
│                                                                 │
│  [Dataset]  ──►  [Clean & Format]  ──►  [Fine-tune LLM]        │
│                        │                       │                │
│                   Person A                Person B             │
│                        │                       │                │
│                        └──────────┬────────────┘                │
│                                   ▼                             │
│                        [Evaluate with ROUGE]                    │
│                                   │                             │
│                                   ▼                             │
│                        [Web App - Streamlit]                    │
│                         User pastes article                     │
│                         → Gets summary                          │
└─────────────────────────────────────────────────────────────────┘
```

### Phase 1 — Data (Person A)
Person A downloads the **CNN/DailyMail** dataset, which contains ~300,000
real news articles paired with human-written summaries. They clean the text
(remove HTML, URLs, etc.) and format it into a prompt structure that the
LLM can learn from:
```
### Instruction:
Summarize the following news article in 2-3 sentences.

### Article:
{long article here}

### Summary:
{short summary here}
```
This formatted data is saved as `train.jsonl`, `val.jsonl`, `test.jsonl`.

### Phase 2 — Model Setup & Fine-Tuning (Person B)
Person B receives the JSONL files from Person A and uses them to **fine-tune**
a pre-trained LLM. "Fine-tuning" means taking a model that already understands
language (e.g., Mistral-7B) and training it further on summarization examples
so it gets very good at *this specific task*.

The technique used is called **LoRA** (Low-Rank Adaptation) — it only trains
a small fraction of the model's weights, making the process fast and
memory-efficient even on a laptop GPU.

After training, Person B exports the model and deploys it **locally** using
**Ollama** — a tool that runs LLMs as a local REST API.

### Phase 3 — Evaluation (Person A + Person B)
Person A writes an evaluation script using **ROUGE scores** — a standard
metric in NLP that measures how much the generated summary overlaps with
the human reference summary.

Person A also runs a **Lead-3 Baseline** (just taking the first 3 sentences
of the article as the "summary") to set a minimum score. Person B's fine-tuned
model must beat this baseline to prove the AI is actually useful.

### Phase 4 — Inference & Web App (Person A)
Person A writes `predict.py` — a script that sends any article to Person B's
Ollama model and gets back the summary. Then they build a **Streamlit web app**
where any user can paste an article and click "Summarize".

---

## 🤝 Collaboration & Dependencies

The two people share data at **two key points**:

```
PERSON A                                    PERSON B
────────────────────────────────────────────────────────────
Step 1: Downloads & cleans data
Step 2: Formats data as JSONL
                          ── delivers JSONL files ──►
                                                    Step 3: Fine-tunes model
                                                    Step 4: Exports to Ollama
                          ◄── delivers Ollama model ──
Step 5: Builds evaluate.py
                          ── delivers evaluate.py ──►
                                                    Step 6: Runs final scores
Step 7: Builds web app
        (connects to Ollama)
────────────────────────────────────────────────────────────
             FINAL RESULT: Working web app + ROUGE scores
```

### The 2 Sync Points
1. **Person A → Person B:** Share the `data/processed/` JSONL files. Person B cannot start real training without these.
2. **Person B → Person A:** Share the Ollama model name (`summarizer`). Person A needs this to finish the web app.

> 💡 **Tip for independent work:** Person A can build the entire web app using
> a **mock summarizer** (a dummy function that returns a fake summary). When
> Person B delivers the real Ollama model, Person A just swaps **one line** of
> code to connect to the real model. No waiting required.

---

## 🛠️ Tech Stack

| Tool | What it's for |
|------|--------------|
| **Python** | Main programming language |
| **Hugging Face `datasets`** | Easily download the CNN/DailyMail dataset |
| **Unsloth** | Fast and memory-efficient LLM fine-tuning |
| **LoRA / PEFT** | Fine-tune only part of the model (saves memory) |
| **Ollama** | Run the fine-tuned LLM as a local API |
| **Hugging Face `evaluate`** | Calculate ROUGE scores |
| **Streamlit** | Build the web interface in pure Python |
| **NLTK** | Text tokenization for baseline evaluation |

### Recommended Models (Person B picks one)
| Model | Size | RAM Needed | Notes |
|-------|------|-----------|-------|
| `mistral-7b` ⭐ | 7B | 8GB GPU | Best quality |
| `llama3.2` | 3B | 4GB GPU | Good balance |
| `phi3-mini` | 3.8B | 4GB GPU | Very efficient |
| `gemma:2b` | 2B | CPU ok | Lowest resource |

---

## 🚀 How to Run the Final System

Once both people have completed their work:

**Step 1 — Start the model server (Person B's machine or shared)**
```bash
ollama serve
```

**Step 2 — Run the web app (Person A's work)**
```bash
streamlit run app/app.py
```

**Step 3 — Open your browser**
```
http://localhost:8501
```

Paste any news article → click **Summarize** → get your AI-generated summary. ✅

---

## 📅 Suggested Timeline

| Week | Person A | Person B |
|------|----------|----------|
| **Week 1** | Download dataset, clean & format data | Set up environment, choose model |
| **Week 2** | Write ROUGE evaluation script, baseline | Receive JSONL data, start fine-tuning |
| **Week 3** | Build web app with mock summarizer | Finish fine-tuning, export to Ollama |
| **Week 4** | Connect web app to real model, final testing | Run final evaluation with ROUGE scores |

---

## 📖 Glossary (Quick Reference)

| Term | Simple Explanation |
|------|--------------------|
| **Fine-tuning** | Taking an existing AI model and training it more on your specific task |
| **LLM** | Large Language Model — an AI that understands and generates text (e.g., Mistral, LLaMA) |
| **LoRA** | A technique to fine-tune only a small part of the model, saving time and memory |
| **Ollama** | A tool that runs LLMs locally on your computer as a simple API |
| **ROUGE** | A metric that measures how similar a generated summary is to a human-written one |
| **JSONL** | A file format where each line is a separate JSON object — used for training data |
| **Streamlit** | A Python library for building web apps in minutes without HTML/CSS |
| **Baseline** | A simple benchmark result (e.g., taking first 3 sentences) — the AI must beat this |
