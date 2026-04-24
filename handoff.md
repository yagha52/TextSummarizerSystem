# 🤝 Handoff Guide for Person B (ML Engineer)

Welcome to the modeling phase! Person A (Data Engineer) has successfully built the data pipeline, the evaluation metrics, and the web interface. 

This document explains what has been built, where the files are, and exactly how they connect to your upcoming tasks.

## 🗂️ What Person A Delivered

Person A has populated the repository with several critical files. Here is what you need to know about them:

### 1. The Dataset (`data/processed/*.jsonl`)
Person A downloaded the CNN/DailyMail news dataset, cleaned out HTML and URLs, and formatted it into three JSONL files (`train.jsonl`, `val.jsonl`, `test.jsonl`).
- **Format:** Each line is a JSON object containing a `"text"` key.
- **Prompt Structure:** The text is already perfectly injected into the Instruction/Article/Summary structure required for your model training. You don't need to do any string formatting!
- **Size:** To prevent endless training times, the dataset was limited to 10,000 training samples and 1,000 validation/test samples.

### 2. The Evaluation Scripts (`evaluation/`)
- `evaluation/baseline.py`: Person A wrote a script that takes the first 3 sentences of every article as a "dumb" baseline summary. 
- `evaluation/evaluate.py`: This contains a `score_model()` function that computes ROUGE scores using Hugging Face's `evaluate` library.
> [!IMPORTANT]
> When you finish fine-tuning your AI model in **Task 7**, you will use the `score_model()` function from `evaluate.py` to compare your AI's summaries against the Human summaries in `test.jsonl`. Your AI must beat Person A's baseline!

### 3. The Web Interface (`app/app.py` & `predict.py`)
Person A built a full Streamlit application to demonstrate the final project.
- `predict.py` expects your fine-tuned model to be running locally on Ollama under the name `"summarizer"`.
- `app/app.py` provides the user interface. 
> [!NOTE]
> You do not need to edit these files. Once you successfully export your model to Ollama in your **Task 6**, Person A can simply run `streamlit run app/app.py` and the whole system will magically work end-to-step.

---

## 🚀 How You Should Start (Your Tasks)

Your job is now entirely focused on deep learning, model selection, and fine-tuning. Here is your roadmap:

### 1. Environment Setup (Tasks 1 & 2)
You need to install your deep learning libraries (`torch`, `transformers`, `unsloth`, `peft`). 
> [!TIP]
> If you are training this on a laptop, you will definitely want to use **Unsloth** and **LoRA** to fine-tune your model in 4-bit quantization, otherwise it will run out of VRAM instantly. 

### 2. Fine-Tuning (Tasks 3 & 4)
Load the `data/processed/train.jsonl` file that Person A generated. Write a training script (e.g., `model/fine_tune.py`) that feeds this data into a lightweight LLM (like Mistral-7B or LLaMA 3.2 3B). Monitor your validation loss carefully!

### 3. Export to Ollama (Task 6)
Once training is complete, convert your model weights to GGUF format and create an Ollama `Modelfile`. Register the model to your local machine:
```bash
ollama create summarizer -f model/Modelfile
```
*As soon as you do this, Person A's `predict.py` script will instantly start working!*

### 4. Final Scoring (Task 7)
Write a Python script that asks your Ollama model to summarize all the articles in `data/processed/test.jsonl`. Put those AI summaries in a list, put the real human summaries in a second list, and pass both lists to Person A's `score_model()` function in `evaluation/evaluate.py`. Record the ROUGE scores for your final report!

---

Good luck! The data is clean and ready. You are clear for takeoff! 🛫
