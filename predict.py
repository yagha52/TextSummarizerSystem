import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# -------------------------
# 1. Load trained model ONCE
# -------------------------
MODEL_PATH = "model/checkpoints/final"

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    dtype=torch.float16,
    device_map="auto"
)

model.eval()

# -------------------------
# 2. Summarize function
# -------------------------
def summarize(article: str) -> str:
    # ✅ SIMPLE prompt (very important)
    prompt = (
        f"Summarize the following news article in exactly 3 sentences:\n"
        f"1) main idea\n"
        f"2) key details\n"
        f"3) impact or risks\n\n"
        f"{article}\n\nSummary:"
    )

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=120,
            temperature=0.4,          # less rigid → better text
            top_p=0.9,
            repetition_penalty=1.1,
            do_sample=True,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.eos_token_id,  # silences max_length conflict warning
        )

    # ✅ decode ONLY the generated part (VERY IMPORTANT)
    generated_tokens = outputs[0][inputs["input_ids"].shape[-1]:]
    result = tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()

    # ✅ clean accidental leftovers
    result = result.replace("Summary:", "").strip()

    # ✅ force max 3 sentences
    sentences = [s.strip() for s in result.split(".") if s.strip()]
    result = ". ".join(sentences[:3])

    if not result.endswith("."):
        result += "."

    return result