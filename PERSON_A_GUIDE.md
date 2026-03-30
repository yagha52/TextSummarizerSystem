# 👤 Person A — Data Engineer & Deployment Lead
## Project: Text Summarization System

> **Your Role:** You are responsible for everything that happens *before* and *after* the model.
> You feed it clean data, measure how good it performs, and bring it to the end-user via a web interface.

---

## 🗂️ Your Task Overview

| # | Task | Phase | Deliverable |
|---|------|-------|-------------|
| 1 | Dataset Collection & Exploration | Phase 1 | Raw dataset downloaded |
| 2 | Data Cleaning & Preprocessing | Phase 1 | Cleaned CSV/JSON files |
| 3 | Data Formatting for Fine-Tuning | Phase 1 | `train.jsonl`, `val.jsonl`, `test.jsonl` |
| 4 | Evaluation Script (ROUGE Metrics) | Phase 4 | `evaluate.py` script |
| 5 | Baseline Model Evaluation | Phase 4 | Baseline ROUGE scores |
| 6 | Inference Pipeline | Phase 5 | `predict.py` script |
| 7 | Web Interface (Streamlit/Gradio) | Phase 6 | Working web app |

---

## 📁 Folder Structure You Own

```
text-summarization/
├── data/
│   ├── raw/                  ← You put downloaded datasets here
│   ├── cleaned/              ← Cleaned text files
│   └── processed/
│       ├── train.jsonl       ← Ready for fine-tuning
│       ├── val.jsonl
│       └── test.jsonl
├── evaluation/
│   ├── evaluate.py           ← Your ROUGE scoring script
│   └── results/              ← Output scores saved here
├── app/
│   └── app.py                ← Your Streamlit/Gradio web app
├── predict.py                ← Your inference pipeline
└── requirements_A.txt        ← Your Python dependencies
```

---

## ✅ Task 1 — Dataset Collection & Exploration

### Goal
Download a real news article dataset that contains (article → summary) pairs.

### Steps

1. Choose one of the following datasets:
   - **CNN/DailyMail** (most popular for news summarization)
   - **XSum** (more abstractive summaries)

2. Install the Hugging Face `datasets` library:
   ```bash
   pip install datasets
   ```

3. Download and explore the dataset:
   ```python
   # explore_data.py
   from datasets import load_dataset

   # Load CNN/DailyMail dataset
   dataset = load_dataset("cnn_dailymail", "3.0.0")

   print(dataset)
   print("--- Example Article ---")
   print(dataset["train"][0]["article"][:500])
   print("--- Example Summary ---")
   print(dataset["train"][0]["highlights"])
   ```

4. Note the column names — for CNN/DailyMail they are:
   - `article` → the long news article (INPUT)
   - `highlights` → the summary (TARGET OUTPUT)

### Deliverable
- A script `explore_data.py` that runs without errors and prints sample data.
- A short note (even just a comment in the file) about the dataset size.

---

## ✅ Task 2 — Data Cleaning & Preprocessing

### Goal
Remove noise and standardize text so the model receives consistent input.

### Steps

1. Create `data/preprocess.py`:
   ```python
   import re

   def clean_text(text):
       # Remove HTML tags
       text = re.sub(r"<.*?>", "", text)
       # Remove URLs
       text = re.sub(r"http\S+|www\S+", "", text)
       # Remove special characters but keep punctuation
       text = re.sub(r"[^a-zA-Z0-9\s.,!?'\"-]", "", text)
       # Normalize whitespace
       text = re.sub(r"\s+", " ", text).strip()
       return text
   ```

2. Apply the cleaning function to all articles and summaries in the dataset.

3. Filter out entries where:
   - The article is shorter than 100 words.
   - The summary is shorter than 10 words.

### Deliverable
- `data/preprocess.py` with the clean_text function.
- Cleaned data saved to `data/cleaned/`.

---

## ✅ Task 3 — Data Formatting for Fine-Tuning

### Goal
Format data into the exact prompt format that will be used to fine-tune the LLM.

### Steps

1. The format should match what Person B will use:
   ```
   ### Instruction:
   Summarize the following news article in 2-3 sentences.

   ### Article:
   {article text here}

   ### Summary:
   {summary text here}
   ```

2. Save to JSONL format (one JSON object per line):
   ```python
   import json

   def format_sample(article, summary):
       prompt = f"### Instruction:\nSummarize the following news article in 2-3 sentences.\n\n### Article:\n{article}\n\n### Summary:\n{summary}"
       return {"text": prompt}

   with open("data/processed/train.jsonl", "w") as f:
       for article, summary in zip(train_articles, train_summaries):
           sample = format_sample(article, summary)
           f.write(json.dumps(sample) + "\n")
   ```

3. Create separate files: `train.jsonl`, `val.jsonl`, `test.jsonl`.
   - Suggested split: **80% train / 10% val / 10% test**.

### Deliverable
- `data/processed/train.jsonl`, `val.jsonl`, `test.jsonl`
- These files are your **handoff to Person B** — they need these to start fine-tuning.

> ⚠️ **SYNC POINT:** Share the JSONL files with Person B before they can begin training.

---

## ✅ Task 4 — Evaluation Script (ROUGE Metrics)

### Goal
Write a script that takes generated summaries and reference summaries and computes ROUGE scores.

### What is ROUGE?
ROUGE measures overlap between the model's generated summary and the human reference summary. Higher is better.

### Steps

1. Install the evaluation library:
   ```bash
   pip install evaluate rouge_score
   ```

2. Create `evaluation/evaluate.py`:
   ```python
   import evaluate
   import json

   rouge = evaluate.load("rouge")

   def compute_rouge(predictions: list, references: list):
       results = rouge.compute(
           predictions=predictions,
           references=references,
           use_stemmer=True
       )
       return results

   if __name__ == "__main__":
       # Test with dummy data
       preds = ["The cat sat on the mat.", "The dog ran in the park."]
       refs  = ["A cat was sitting on a mat.", "A dog was running in the park."]
       scores = compute_rouge(preds, refs)
       print(scores)
   ```

3. The script should save results to `evaluation/results/rouge_scores.json`.

### Deliverable
- `evaluation/evaluate.py` that accepts predictions and references and prints/saves ROUGE-1, ROUGE-2, ROUGE-L scores.

---

## ✅ Task 5 — Baseline Model Evaluation

### Goal
Create a *Lead-3 Baseline*: simply take the first 3 sentences of the article as the "summary". This gives us a minimum score the fine-tuned model must beat.

### Steps

1. Create `evaluation/baseline.py`:
   ```python
   import nltk
   nltk.download("punkt")
   from nltk.tokenize import sent_tokenize

   def lead3_baseline(article: str) -> str:
       sentences = sent_tokenize(article)
       return " ".join(sentences[:3])
   ```

2. Run the baseline on your test set and call `compute_rouge()` from Task 4.
3. Save your baseline ROUGE scores to `evaluation/results/baseline_scores.json`.

### Deliverable
- `evaluation/baseline.py`
- `evaluation/results/baseline_scores.json` with the actual numbers. *(This is the bar Person B's model must beat!)*

---

## ✅ Task 6 — Inference Pipeline

> ⚠️ **Dependency:** Wait for Person B to deliver the fine-tuned model before finalizing this task.

### Goal
Write a `predict.py` script that:
1. Loads the fine-tuned model.
2. Takes a raw news article string as input.
3. Returns the generated summary.

### Steps

1. Create `predict.py`:
   ```python
   # predict.py — works with Ollama locally deployed model
   import requests

   OLLAMA_URL = "http://localhost:11434/api/generate"
   MODEL_NAME = "summarizer"  # Name Person B gives the fine-tuned model

   def summarize(article: str) -> str:
       prompt = f"### Instruction:\nSummarize the following news article in 2-3 sentences.\n\n### Article:\n{article}\n\n### Summary:\n"
       payload = {
           "model": MODEL_NAME,
           "prompt": prompt,
           "stream": False
       }
       response = requests.post(OLLAMA_URL, json=payload)
       response.raise_for_status()
       return response.json()["response"].strip()

   if __name__ == "__main__":
       test_article = input("Paste your article here:\n")
       summary = summarize(test_article)
       print("\n--- Generated Summary ---")
       print(summary)
   ```

2. Test with a sample article from the test set.

### Deliverable
- `predict.py` that runs successfully and returns a summary.

---

## ✅ Task 7 — Web Interface

> ⚠️ **Dependency:** `predict.py` must work before building this.

### Goal
Build a clean, user-friendly web app where anyone can paste an article and get a summary.

### Steps

1. Install Streamlit:
   ```bash
   pip install streamlit
   ```

2. Create `app/app.py`:
   ```python
   import streamlit as st
   import sys
   sys.path.append("..")
   from predict import summarize

   st.set_page_config(page_title="📰 Text Summarizer", layout="wide")
   st.title("📰 AI News Article Summarizer")
   st.markdown("Paste a long news article below and get a concise summary instantly.")

   article = st.text_area("📄 Article", height=300, placeholder="Paste your article here...")

   if st.button("✨ Summarize"):
       if article.strip():
           with st.spinner("Generating summary..."):
               summary = summarize(article)
           st.subheader("📝 Summary")
           st.success(summary)
       else:
           st.warning("Please paste an article first.")
   ```

3. Run the app:
   ```bash
   streamlit run app/app.py
   ```

### Deliverable
- A running Streamlit web app accessible at `http://localhost:8501`

---

## 📦 Your `requirements_A.txt`

```
datasets
evaluate
rouge_score
nltk
streamlit
requests
```

Install all at once:
```bash
pip install -r requirements_A.txt
```

---

## 📅 Your Timeline & Sync Points

| Week | Action |
|------|--------|
| Week 1 | Complete Tasks 1, 2, 3 — Share JSONL files with Person B |
| Week 2 | Complete Tasks 4 & 5 — Share `evaluate.py` with Person B |
| Week 3 | Receive model from Person B, complete Task 6 |
| Week 4 | Complete Task 7, integrate and test the full system |
