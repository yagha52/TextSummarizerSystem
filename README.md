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

|                           | Yara                                                                 | Gaelle                                                     |
| ------------------------- | -------------------------------------------------------------------- | ---------------------------------------------------------- |
| **Role**            | Data Engineer & Deployment                                           | ML Engineer & Model Expert                                 |
| **Responsible for** | Collecting & cleaning data, evaluating quality, building the web app | Choosing the model, fine-tuning it, deploying it as an API |

---

## 🗂️ Project Folder Structure

```
text-summarization/
│
├── README.md                 ← You are here
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
│   ├── fine_tune.py          ← Fine-tuning script (Transformers + PEFT)
│   └── checkpoints/
│       └── final/            ← Final fine-tuned weights (adapters)
│
├── evaluation/
│   ├── rouge_utils.py        ← ROUGE scoring logic
│   ├── baseline.py           ← Lead-3 baseline logic
│   └── results/              ← Saved ROUGE scores & comparisons
│
├── app/
│   └── app.py                ← Streamlit web interface
│
├── predict.py                ← Direct inference (Loads model via Transformers)
├── run_evaluation.py         ← Main evaluation runner
└── TECHNICAL_SUMMARY.md      ← Detailed project technical report
```

---

## 🔄 How the Project Works — Step by Step

Here is the full pipeline from raw data to a working web app:

```
┌─────────────────────────────────────────────────────────────────┐
│                         FULL PIPELINE                           │
│                                                                 │
│  [Dataset]  ──►  [Clean & Format]  ──►  [Fine-tune LLM]         │
│                        │                       │                │
│                      Yara                   Gaelle              │
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

### Phase 1 — Data (Yara)

Yara downloads the **CNN/DailyMail** dataset, which contains ~300,000
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

### Phase 2 — Model Setup & Fine-Tuning (Gaelle)

Gaelle receives the JSONL files from Yara and uses them to **fine-tune**
a pre-trained LLM. "Fine-tuning" means taking a model that already understands
language (e.g., Mistral-7B) and training it further on summarization examples
so it gets very good at *this specific task*.

The technique used is called **LoRA** (Low-Rank Adaptation) — it only trains
a small fraction of the model's weights, making the process fast and
memory-efficient even on a laptop GPU.

After training, Gaelle exports the model weights to `model/checkpoints/final`. The project uses the **Transformers** library for inference, allowing for direct integration into Python scripts without needing external model servers.

### Phase 3 — Evaluation (Yara + Gaelle)

Yara writes an evaluation script using **ROUGE scores** — a standard
metric in NLP that measures how much the generated summary overlaps with
the human reference summary.

Yara also runs a **Lead-3 Baseline** (just taking the first 3 sentences
of the article as the "summary") to set a minimum score. Gaelle's fine-tuned
model must beat this baseline to prove the AI is actually useful.

### Phase 4 — Inference & Web App (Yara)

Yara writes `predict.py` — a script that loads the fine-tuned model and generates summaries. Then they build a **Streamlit web app** where any user can paste an article and click "Summarize".

---

## 🤝 Collaboration & Dependencies

The two people share data at **two key points**:

```
Yara                                            Gaelle
────────────────────────────────────────────────────────────
Step 1: Downloads & cleans data
Step 2: Formats data as JSONL
                          ── delivers JSONL files ──►
                                                    Step 3: Fine-tunes model
                                                    Step 4: Saves weights to `checkpoints/final`
                          ── model is ready to load ──►
Step 5: Builds rouge_utils.py
                          ── delivers rouge_utils.py ──►
                                                     Step 6: Runs final scores
Step 7: Builds web app
        (connects to predict.py)
────────────────────────────────────────────────────────────
             FINAL RESULT: Working web app + ROUGE scores
```

### The 2 Sync Points

1. **Yara → Gaelle:** Share the `data/processed/` JSONL files. Gaelle cannot start real training without these.
2. **Gaelle → Yara:** Share the final model directory path (`model/checkpoints/final`). Yara needs this for `predict.py` to work.

> 💡 **Tip for independent work:** Yara can build the entire web app using
> a **mock summarizer** (a dummy function that returns a fake summary). When
> Gaelle delivers the real model weights, Yara just swaps **one line** of
> code to connect to the real model. No waiting required.

---

## 🛠️ Tech Stack

| Tool                                | What it's for                                   |
| ----------------------------------- | ----------------------------------------------- |
| **Python**                    | Main programming language                       |
| **Hugging Face `datasets`** | Easily download the CNN/DailyMail dataset       |
| **BitsAndBytes**              | 4-bit quantization to save GPU memory           |
| **LoRA / PEFT**               | Fine-tune only part of the model (saves memory) |
| **PyTorch / Transformers**    | Core libraries for model loading and inference  |
| **Hugging Face `evaluate`** | Calculate ROUGE scores                          |
| **Streamlit**                 | Build the web interface in pure Python          |
| **NLTK**                      | Text tokenization for baseline evaluation       |

### Recommended Models (Gaelle picks one)

| Model             | Size | RAM Needed | Notes           |
| ----------------- | ---- | ---------- | --------------- |
| `mistral-7b` ⭐ | 7B   | 8GB GPU    | Best quality    |
| `llama3.2`      | 3B   | 4GB GPU    | Good balance    |
| `phi3-mini`     | 3.8B | 4GB GPU    | Very efficient  |
| `gemma:2b`      | 2B   | CPU ok     | Lowest resource |

---

## 🚀 How to Run the Final System

Once both people have completed their work:

**Step 1 — Create and activate a virtual environment**

```bash
# Create venv
python -m venv .venv

# Activate (Windows)
.\.venv\Scripts\activate

# Activate (Mac/Linux)
source .venv/bin/activate
```

**Step 2 — Install dependencies**

```bash
pip install -r requirements.txt
```

**Step 2 — Run the web app**

```bash
streamlit run app/app.py
```

**Step 3 — Run Evaluation (Optional)**

```bash
python run_evaluation.py --quick
```

**Step 3 — Open your browser**

```
http://localhost:8501
```

Paste any news article → click **Summarize** → get your AI-generated summary. ✅

---

## 📅 Suggested Timeline

| Week             | Yara                                         | Gaelle                                 |
| ---------------- | -------------------------------------------- | -------------------------------------- |
| **Week 1** | Download dataset, clean & format data        | Set up environment, choose model       |
| **Week 2** | Write ROUGE evaluation script, baseline      | Receive JSONL data, start fine-tuning  |
| **Week 3** | Build web app with mock summarizer           | Finish fine-tuning, save model weights |
| **Week 4** | Connect web app to real model, final testing | Run final evaluation with ROUGE scores |

---

## 📖 Glossary (Quick Reference)

| Term                   | Simple Explanation                                                                       |
| ---------------------- | ---------------------------------------------------------------------------------------- |
| **Fine-tuning**  | Taking an existing AI model and training it more on your specific task                   |
| **LLM**          | Large Language Model — an AI that understands and generates text (e.g., Mistral, LLaMA) |
| **LoRA**         | A technique to fine-tune only a small part of the model, saving time and memory          |
| **Transformers** | The library used to load and run the Llama model directly in Python                      |
| **ROUGE**        | A metric that measures how similar a generated summary is to a human-written one         |
| **JSONL**        | A file format where each line is a separate JSON object — used for training data        |
| **Streamlit**    | A Python library for building web apps in minutes without HTML/CSS                       |
| **Baseline**     | A simple benchmark result (e.g., taking first 3 sentences) — the AI must beat this      |
