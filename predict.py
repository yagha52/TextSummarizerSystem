import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# -------------------------
# 1. Load trained model ONCE
# -------------------------
MODEL_PATH = "model/checkpoints/final"

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    torch_dtype=torch.float16,
    device_map="auto"
)

model.eval()

# -------------------------
# 2. Summarize function
# -------------------------
def summarize(article: str) -> str:
    prompt = (
        "You are a strict news summarizer.\n"
        "Follow instructions EXACTLY.\n\n"
        "RULES:\n"
        "- Output EXACTLY 3 sentences\n"
        "- Do not add analysis or opinions\n"
        "- Do not add extra commentary\n"
        "- Do not exceed 3 sentences under any condition\n\n"
        "ARTICLE:\n"
        f"{article}\n\n"
        "SUMMARY (3 sentences only):\n"
    )

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            temperature=0.2,
            top_p=0.8,
            repetition_penalty=1.3,
            max_new_tokens=120,
            do_sample=True
        )

    result = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # extract only after SUMMARY
    if "SUMMARY:" in result:
        result = result.split("SUMMARY:")[-1].strip()
    
    sentences = result.split(".")
    result = ".".join(sentences[:3]).strip()
    if not result.endswith("."):
        result += "."

    return result