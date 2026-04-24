import re
import os
import json
from datasets import load_dataset
from tqdm import tqdm

# ---- SAFE LENGTH LIMITS (word-based approximation) ----
# Keep SMALLER than model max to avoid tokenizer overflow
MAX_ARTICLE_WORDS = 220
MAX_SUMMARY_WORDS = 60


def clean_text(text):
    if not isinstance(text, str):
        return ""

    text = re.sub(r"<.*?>", "", text)               # HTML
    text = re.sub(r"http\S+|www\S+", "", text)     # URLs
    text = re.sub(r"[^a-zA-Z0-9\s.,!?'\"-]", "", text)
    text = re.sub(r"\s+", " ", text).strip()

    return text


def process_and_save(split_name, output_jsonl, max_samples=None):
    print(f"Processing {split_name} split...")

    os.makedirs(os.path.dirname(output_jsonl), exist_ok=True)

    dataset = load_dataset(
        "cnn_dailymail",
        "3.0.0",
        split=split_name,
        streaming=True
    )

    if max_samples:
        dataset = dataset.take(max_samples)

    count = 0

    with open(output_jsonl, "w", encoding="utf-8") as f:

        for row in tqdm(dataset, desc=split_name):

            article = clean_text(row["article"])
            summary = clean_text(row["highlights"])

            # HARD LIMITS (prevents overflow)
            article = " ".join(article.split()[:MAX_ARTICLE_WORDS])
            summary = " ".join(summary.split()[:MAX_SUMMARY_WORDS])

            # FILTER QUALITY
            if len(article.split()) < 50 or len(summary.split()) < 5:
                continue

            prompt = (
                "### Instruction:\n"
                "Summarize the following news article in 2-3 sentences.\n\n"
                f"### Article:\n{article}\n\n"
                f"### Summary:\n{summary}"
            )

            f.write(json.dumps({"text": prompt}) + "\n")
            count += 1

    print(f"Saved {count} samples to {output_jsonl}")


if __name__ == "__main__":

    process_and_save("train", "data/processed/train.jsonl", max_samples=10000)
    process_and_save("validation", "data/processed/val.jsonl", max_samples=1000)
    process_and_save("test", "data/processed/test.jsonl", max_samples=1000)

    print("Done.")