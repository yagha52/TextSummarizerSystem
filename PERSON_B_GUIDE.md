# 👤 Person B — ML Engineer & Model Lead
## Project: Text Summarization System

> **Your Role:** You are the brain of the project. You are responsible for choosing
> the right model, fine-tuning it on the dataset provided by Person A, optimizing
> it for quality summaries, and delivering the final trained model.

---

## 🗂️ Your Task Overview

| # | Task | Phase | Deliverable |
|---|------|-------|-------------|
| 1 | Model Research & Selection | Phase 2 | Documented model choice |
| 2 | Environment Setup (Ollama + Unsloth) | Phase 2 | Working local LLM environment |
| 3 | Load & Verify Dataset | Phase 2 | Confirmed data format |
| 4 | Fine-Tuning the Model | Phase 3 | Fine-tuned model weights |
| 5 | Hyperparameter Tuning | Phase 3 | Optimized model config |
| 6 | Export Model for Ollama | Phase 3 | `Modelfile` + deployed model |
| 7 | Final Evaluation | Phase 4 | ROUGE scores vs. baseline |

---

## 📁 Folder Structure You Own

```
text-summarization/
├── model/
│   ├── fine_tune.py          ← Your main fine-tuning script
│   ├── config.py             ← Hyperparameters and settings
│   ├── Modelfile             ← Ollama model definition file
│   └── checkpoints/          ← Saved model weights during training
├── notebooks/
│   └── experiment_log.ipynb  ← Track your experiments here
└── requirements_B.txt        ← Your Python dependencies
```

---

## ✅ Task 1 — Model Research & Selection

### Goal
Choose the best LLM for text summarization that can run locally with minimal hardware requirements.

### Recommended Models

| Model | Size | Best For | Hardware Needed |
|-------|------|----------|-----------------|
| **Mistral-7B** ⭐ | 7B params | General summarization, great quality | 8GB+ VRAM |
| **LLaMA 3.2 3B** | 3B params | Lightweight, faster training | 4GB+ VRAM |
| **Phi-3 Mini** | 3.8B params | Very efficient, low resource usage | 4GB+ VRAM |
| **Gemma 2B** | 2B params | Smallest option, CPU-friendly | No GPU needed |

### Recommendation
> ✅ Use **Mistral-7B** if you have a decent GPU (8GB+ VRAM).
> ✅ Use **LLaMA 3.2 3B** or **Phi-3 Mini** if on a laptop or limited hardware.

### Steps
1. Check your GPU memory:
   ```bash
   nvidia-smi
   ```
2. Pick your model from the table above based on your hardware.
3. Document your choice in `model/config.py`:
   ```python
   # config.py
   MODEL_NAME    = "unsloth/mistral-7b-bnb-4bit"  # or your chosen model
   MAX_SEQ_LEN   = 2048
   BATCH_SIZE    = 2
   GRAD_ACCUM    = 4
   LEARNING_RATE = 2e-4
   EPOCHS        = 3
   LORA_RANK     = 16
   ```

### Deliverable
- `model/config.py` filled in with your chosen model name and settings.

---

## ✅ Task 2 — Environment Setup (Ollama + Unsloth)

### Goal
Set up your local development environment with all the tools needed for loading and fine-tuning LLMs.

### Steps

#### 2a — Install Ollama (for running the model locally)
1. Download Ollama from: https://ollama.com/download
2. Install and verify:
   ```bash
   ollama --version
   ```
3. Pull your chosen base model:
   ```bash
   # For Mistral
   ollama pull mistral

   # For LLaMA 3.2
   ollama pull llama3.2
   ```

#### 2b — Install Python Fine-Tuning Libraries
```bash
pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"
pip install --no-deps trl peft accelerate bitsandbytes
pip install torch transformers datasets
```

> 💡 **Tip:** If you don't have a GPU, you can use **Google Colab** (free GPU) to run
> the fine-tuning, then download the weights and deploy locally with Ollama.

#### 2c — Verify setup
```python
# test_setup.py
from unsloth import FastLanguageModel
print("✅ Unsloth installed correctly")

import torch
print(f"✅ GPU available: {torch.cuda.is_available()}")
print(f"   Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
```

### Deliverable
- All libraries installed without errors.
- `test_setup.py` runs and confirms GPU/CPU availability.

---

## ✅ Task 3 — Load & Verify Dataset

> ⚠️ **Dependency:** You need the `train.jsonl`, `val.jsonl`, `test.jsonl` files from **Person A** before proceeding.

### Goal
Load Person A's processed dataset and confirm it matches the expected format.

### Steps

1. Once you receive the `.jsonl` files from Person A, place them in `data/processed/`.

2. Verify the format:
   ```python
   # verify_data.py
   import json

   with open("data/processed/train.jsonl", "r") as f:
       for i, line in enumerate(f):
           sample = json.loads(line)
           if i == 0:
               print("✅ Keys found:", list(sample.keys()))
               print("✅ Sample text preview:")
               print(sample["text"][:300])
               break
   ```

3. Confirm the `"text"` field contains the expected format:
   ```
   ### Instruction:
   Summarize the following news article in 2-3 sentences.

   ### Article:
   {article}

   ### Summary:
   {summary}
   ```

4. Count samples:
   ```python
   with open("data/processed/train.jsonl") as f:
       count = sum(1 for _ in f)
   print(f"Training samples: {count}")
   ```

### Deliverable
- Confirmed the dataset loads correctly and has the right format.
- Know the number of training, val, and test samples.

---

## ✅ Task 4 — Fine-Tuning the Model

### Goal
Fine-tune your chosen LLM on the summarization dataset using **LoRA** (Low-Rank Adaptation), which fine-tunes only a small fraction of the model weights — making it fast and memory-efficient.

### Steps

Create `model/fine_tune.py`:

```python
# fine_tune.py
from unsloth import FastLanguageModel
from trl import SFTTrainer
from transformers import TrainingArguments
from datasets import load_dataset
import config

# ── 1. Load the base model ──────────────────────────────────────────────────
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name   = config.MODEL_NAME,
    max_seq_length = config.MAX_SEQ_LEN,
    load_in_4bit = True,  # Use 4-bit quantization to save memory
)

# ── 2. Apply LoRA adapters ──────────────────────────────────────────────────
model = FastLanguageModel.get_peft_model(
    model,
    r              = config.LORA_RANK,
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                      "gate_proj", "up_proj", "down_proj"],
    lora_alpha     = 16,
    lora_dropout   = 0,
    bias           = "none",
    use_gradient_checkpointing = True,
)

# ── 3. Load dataset ─────────────────────────────────────────────────────────
dataset = load_dataset("json", data_files={
    "train": "data/processed/train.jsonl",
    "validation": "data/processed/val.jsonl"
})

# ── 4. Training configuration ───────────────────────────────────────────────
trainer = SFTTrainer(
    model        = model,
    tokenizer    = tokenizer,
    train_dataset = dataset["train"],
    eval_dataset  = dataset["validation"],
    dataset_text_field = "text",
    max_seq_length = config.MAX_SEQ_LEN,
    args = TrainingArguments(
        per_device_train_batch_size   = config.BATCH_SIZE,
        gradient_accumulation_steps   = config.GRAD_ACCUM,
        warmup_steps                  = 50,
        num_train_epochs              = config.EPOCHS,
        learning_rate                 = config.LEARNING_RATE,
        fp16                          = True,
        logging_steps                 = 10,
        evaluation_strategy           = "epoch",
        save_strategy                 = "epoch",
        output_dir                    = "model/checkpoints",
        load_best_model_at_end        = True,
        report_to                     = "none",
    ),
)

# ── 5. Train! ───────────────────────────────────────────────────────────────
print("🚀 Starting fine-tuning...")
trainer.train()
print("✅ Fine-tuning complete!")

# ── 6. Save the fine-tuned model ────────────────────────────────────────────
model.save_pretrained("model/checkpoints/final")
tokenizer.save_pretrained("model/checkpoints/final")
print("✅ Model saved to model/checkpoints/final")
```

Run the training:
```bash
python model/fine_tune.py
```

### What to Monitor
- **Training Loss** should decrease over time.
- **Validation Loss** should also decrease (if it increases, the model is overfitting — stop early).

### Deliverable
- `model/checkpoints/final/` — the saved fine-tuned model weights.
- Training logs showing loss curves.

---

## ✅ Task 5 — Hyperparameter Tuning

### Goal
Experiment with different settings to get the best possible summary quality.

### What to Experiment With

| Parameter | Default | Try Also |
|-----------|---------|----------|
| `LEARNING_RATE` | `2e-4` | `1e-4`, `5e-4` |
| `LORA_RANK` | `16` | `8`, `32` |
| `EPOCHS` | `3` | `2`, `5` |
| `BATCH_SIZE` | `2` | `4`, `8` |

### Steps
1. Create a copy of `config.py` for each experiment (e.g., `config_exp1.py`).
2. Run training with each config.
3. Log all experiment results in `notebooks/experiment_log.ipynb`:

| Experiment | LR | LoRA Rank | Epochs | Val Loss | ROUGE-1 |
|------------|-----|-----------|--------|----------|---------|
| Exp 1 | 2e-4 | 16 | 3 | ? | ? |
| Exp 2 | 1e-4 | 32 | 3 | ? | ? |

4. Keep the best model checkpoint as your final model.

### Deliverable
- `notebooks/experiment_log.ipynb` with results of at least 2 experiments.
- Best model checkpoint identified.

---

## ✅ Task 6 — Export Model for Ollama

### Goal
Convert your fine-tuned model to GGUF format and deploy it with Ollama so Person A can use it in their `predict.py` script.

### Steps

#### 6a — Convert to GGUF format
```python
# In fine_tune.py, append this after saving:
model.save_pretrained_gguf(
    "model/gguf_model",
    tokenizer,
    quantization_method = "q4_k_m"  # Good balance of size vs quality
)
print("✅ GGUF model saved!")
```

#### 6b — Create an Ollama Modelfile
Create `model/Modelfile`:
```
FROM ./gguf_model/model-q4_k_m.gguf

SYSTEM """
You are a helpful AI assistant specialized in summarizing news articles.
Always respond with a concise, accurate summary in 2-3 sentences.
"""

PARAMETER temperature 0.3
PARAMETER top_p 0.9
PARAMETER num_predict 200
```

#### 6c — Register the model with Ollama
```bash
ollama create summarizer -f model/Modelfile
```

#### 6d — Test it works
```bash
ollama run summarizer "### Instruction:\nSummarize this article.\n\n### Article:\nOpenAI released GPT-4 today, marking a major milestone in AI development...\n\n### Summary:"
```

### Deliverable
- Model registered with Ollama under the name `summarizer`.
- Person A can now use `predict.py` to call it.

> ✅ **SYNC POINT:** Notify Person A that the model is deployed and ready. They just need to run `ollama serve` and their `predict.py` will work.

---

## ✅ Task 7 — Final Evaluation

> ⚠️ **Dependency:** You need `evaluation/evaluate.py` from **Person A** to run this task.

### Goal
Measure how good your fine-tuned model is using ROUGE scores and compare it against the baseline.

### Steps

1. Generate summaries for the entire test set:
   ```python
   # run_evaluation.py
   import json
   import requests

   OLLAMA_URL  = "http://localhost:11434/api/generate"
   MODEL_NAME  = "summarizer"

   def generate_summary(article):
       prompt = f"### Instruction:\nSummarize the following news article in 2-3 sentences.\n\n### Article:\n{article}\n\n### Summary:\n"
       resp = requests.post(OLLAMA_URL, json={
           "model": MODEL_NAME, "prompt": prompt, "stream": False
       })
       return resp.json()["response"].strip()

   predictions, references = [], []

   with open("data/processed/test.jsonl") as f:
       for line in f:
           sample = json.loads(line)
           # Extract article and reference summary from the text field
           parts   = sample["text"].split("### Summary:\n")
           article = parts[0].split("### Article:\n")[1].strip()
           ref     = parts[1].strip()
           pred    = generate_summary(article)
           predictions.append(pred)
           references.append(ref)

   # Use Person A's evaluation script
   import sys
   sys.path.append("evaluation")
   from evaluate import compute_rouge

   scores = compute_rouge(predictions, references)
   print("📊 Final ROUGE Scores:", scores)
   ```

2. Compare your scores against Person A's baseline scores from `evaluation/results/baseline_scores.json`.

3. Write a short summary report.

### What Good Looks Like

| Metric | Baseline (Lead-3) | Target (Fine-tuned) |
|--------|-------------------|---------------------|
| ROUGE-1 | ~0.30 | > 0.38 |
| ROUGE-2 | ~0.11 | > 0.15 |
| ROUGE-L | ~0.20 | > 0.25 |

### Deliverable
- `evaluation/results/final_rouge_scores.json`
- Confirmation that fine-tuned model beats the baseline.

---

## 📦 Your `requirements_B.txt`

```
torch
transformers
datasets
trl
peft
accelerate
bitsandbytes
requests
jupyter
```

Install:
```bash
pip install -r requirements_B.txt
```

> ⚠️ Install Unsloth separately (see Task 2 instructions).

---

## 📅 Your Timeline & Sync Points

| Week | Action |
|------|--------|
| Week 1 | Complete Tasks 1 & 2 — Set up environment, choose model |
| Week 1-2 | Task 3 — Wait for JSONL files from Person A, verify them |
| Week 2-3 | Tasks 4 & 5 — Fine-tune and tune hyperparameters |
| Week 3 | Task 6 — Export & deploy model via Ollama, notify Person A |
| Week 4 | Task 7 — Run final evaluation with Person A's scripts |

---

## 🤝 How You Connect With Person A

```
Person A                              Person B
─────────────────────────────────────────────────────────────
Delivers:  train/val/test.jsonl  ──►  You use to fine-tune
           evaluate.py           ──►  You use for final scores

You deliver: Ollama "summarizer"  ──►  Person A uses in predict.py
             final ROUGE scores   ──►  Person A displays in web app
```
