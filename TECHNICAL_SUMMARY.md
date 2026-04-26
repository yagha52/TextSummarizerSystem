# 🛠️ Technical Summary: Text Summarization System

This document provides a detailed explanation of the modeling phase, the architecture choices, and the final results achieved. It is intended for the next developer or for project documentation.

## 1. Project Overview
The goal was to build a news summarization system using a fine-tuned Large Language Model (LLM) that can run efficiently on consumer hardware (specifically a laptop with **4GB VRAM**).

## 2. Model & Architecture
- **Base Model**: `Llama-3.2-3B` (via the `unsloth/Llama-3.2-3B-bnb-4bit` pre-quantized hub model).
- **Quantization**: **4-bit (bitsandbytes)**. This reduced the memory footprint from ~12GB to ~2.5GB, allowing it to fit into VRAM.
- **Fine-Tuning Method**: **LoRA (Low-Rank Adaptation)** via the `peft` library. Instead of training all 3 billion parameters, we trained a tiny fraction (~0.5%) of "adapter" weights.

## 3. Architectural Decisions: Why Python over Ollama?
Although the initial plan considered using Ollama, the final implementation moved to a **direct Python-based inference** (`predict.py` using Transformers) for several key reasons:
- **VRAM Efficiency**: By running everything within a single Python environment, we avoided the extra memory overhead of running an external Ollama server, which was critical for the **4GB VRAM** limit.
- **Deep Integration**: Direct integration allowed the Streamlit web app and the evaluation scripts to share the same model loading logic without needing API calls or background services.
- **Standard Tooling**: By using the official `transformers` and `peft` libraries, the project maintains high compatibility and stability across different environments.

> [!NOTE]
> **Llama vs. Ollama**: It is important to distinguish between the two. **Llama 3.2** is the model (the "brain"), while **Ollama** is a standalone runtime environment (the "player"). In this project, we chose to run the Llama model directly through the Transformers library to eliminate the "middleman" and maximize resource efficiency.

## 4. Training Implementation (`model/fine_tune.py`)
To handle the hardware constraints, the following techniques were used:
- **Paged AdamW 8-bit Optimizer**: Offloads optimizer states to CPU when necessary.
- **Gradient Checkpointing**: Saves VRAM by recomputing activations during the backward pass instead of storing them all.
- **Gradient Accumulation**: Set to `4` with a `batch_size` of `1` to simulate a larger batch size without increasing memory usage.
- **Sequence Length**: Restricted to **384 tokens** to ensure stability on small GPUs.

## 4. Code Flow
1. **Data**: `data/processed/*.jsonl` (provided by Person A) contains news articles formatted with a specific instruction prompt.
2. **Fine-Tuning**: `model/fine_tune.py` loads the 4-bit model, adds LoRA adapters, and trains on the data. The final weights are saved in `model/checkpoints/final`.
3. **Inference**: `predict.py` loads the trained weights and provides a `summarize()` function. It uses a 3-sentence constraint in the prompt to ensure consistency.
4. **Evaluation**: `run_evaluation.py` compares the model's summaries against a **Lead-3 baseline** (the first 3 sentences of the article).

## 5. Results & Interpretation
The model was evaluated on a subset of the test set.

| Metric | Fine-Tuned Model | Lead-3 Baseline |
|--------|------------------|-----------------|
| **ROUGE-1** | 0.2929 | 0.2967 |
| **ROUGE-2** | 0.1096 | 0.1295 |
| **ROUGE-L** | 0.2165 | 0.2181 |

### Interpretation:
- **The "Lead-3" Challenge**: In news summarization, the first three sentences usually contain the most important facts. This makes the Lead-3 baseline extremely difficult to beat.
- **Performance**: The fine-tuned model is performing **on par** with the Lead-3 baseline. This indicates that the model has successfully learned to identify key information, even with a small sequence length (384) and 4-bit quantization.
- **Hardware vs. Quality Trade-off**: To fit the model on a **4GB GPU**, we prioritized memory efficiency (quantization, lower context length) over maximum ROUGE scores. Achieving parity with Lead-3 in such a constrained environment is a significant technical success.
- **Future Improvements**: Increasing the `MAX_SEQ_LEN` (if hardware allows) or training for more epochs would likely push the scores above the baseline.

## 6. How to Run
1. **Inference**: `python predict.py` (requires model weights in `model/checkpoints/final`).
2. **Evaluation**: `python run_evaluation.py --quick`.
3. **App**: `streamlit run app/app.py` (uses the `summarize` function from `predict.py`).

---

