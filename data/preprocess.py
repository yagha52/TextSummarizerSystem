import re
import os
import json
from datasets import load_dataset
from tqdm import tqdm

def clean_text(text):
    if not isinstance(text, str):
        return ""
    # Remove HTML tags
    text = re.sub(r"<.*?>", "", text)
    # Remove URLs
    text = re.sub(r"http\S+|www\S+", "", text)
    # Remove special characters but keep punctuation
    text = re.sub(r"[^a-zA-Z0-9\s.,!?'\"-]", "", text)
    # Normalize whitespace
    text = re.sub(r"\s+", " ", text).strip()
    return text

def process_and_save(dataset, split_name, output_jsonl, max_samples=None):
    """Clean data and save directly to formatted jsonl."""
    print(f"Processing {split_name} split...")
    os.makedirs(os.path.dirname(output_jsonl), exist_ok=True)
    
    count = 0
    with open(output_jsonl, "w", encoding="utf-8") as f:
        samples = dataset[split_name]
        if max_samples:
            # For streaming datasets, we use .take() instead of .select()
            samples = samples.take(max_samples)
            
        for row in tqdm(samples, desc=split_name):
            article = clean_text(row["article"])
            summary = clean_text(row["highlights"])
            
            # Filter condition: Article > 100 char and Summary > 10 char roughly approximates the words limit, 
            # but let's do a simple proper word count check.
            if len(article.split()) >= 100 and len(summary.split()) >= 10:
                prompt = (
                    "### Instruction:\nSummarize the following news article in 2-3 sentences.\n\n"
                    f"### Article:\n{article}\n\n### Summary:\n{summary}"
                )
                sample = {"text": prompt}
                f.write(json.dumps(sample) + "\n")
                count += 1
                
    print(f"Saved {count} valid samples to {output_jsonl}")

if __name__ == "__main__":
    print("Loading dataset...")
    dataset = load_dataset("cnn_dailymail", "3.0.0", streaming=True)
    
    # Process train, val, test
    # We will process a subset for train to make it feasible for local fine-tuning unless they have a massive GPU.
    # We output them to the required paths for Person B.
    process_and_save(dataset, "train", "data/processed/train.jsonl", max_samples=10000) # taking 10k samples for speed
    process_and_save(dataset, "validation", "data/processed/val.jsonl", max_samples=1000)
    process_and_save(dataset, "test", "data/processed/test.jsonl", max_samples=1000)
    
    print("Data preparation complete! JSONL files are ready in data/processed/")
