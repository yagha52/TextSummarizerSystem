# run_evaluation.py
import json
import requests
import sys
import os

# Connect to Person A's evaluation metric script
sys.path.append("evaluation")
from evaluate import compute_rouge

OLLAMA_URL  = "http://localhost:11434/api/generate"
MODEL_NAME  = "summarizer"

def generate_summary(article):
    prompt = f"### Instruction:\nSummarize the following news article in 2-3 sentences.\n\n### Article:\n{article}\n\n### Summary:\n"
    try:
        resp = requests.post(OLLAMA_URL, json={
            "model": MODEL_NAME, "prompt": prompt, "stream": False
        })
        resp.raise_for_status()
        return resp.json()["response"].strip()
    except requests.exceptions.RequestException as e:
        print(f"Failed to connect to Ollama: {e}")
        return ""

def run_eval():
    predictions, references = [], []
    
    print(f"⏳ Evaluating the test set using Ollama model '{MODEL_NAME}'...")
    
    try:
        with open("data/processed/test.jsonl", "r", encoding="utf-8") as f:
            for i, line in enumerate(f):
                sample = json.loads(line)
                
                # The data is pre-formatted as Instruction/Article/Summary.
                # We split it up to feed just the article to our model.
                parts   = sample["text"].split("### Summary:\n")
                article = parts[0].split("### Article:\n")[1].strip()
                ref     = parts[1].strip()
                
                print(f"Generating summary {i+1}...")
                pred = generate_summary(article)
                
                if not pred:
                    print("Error: Empty prediction returned. Is your Ollama server running?")
                    return
                
                predictions.append(pred)
                references.append(ref)
                
                # To test quickly without waiting for 1,000 articles, uncomment the next line:
                # if i == 9: break

    except FileNotFoundError:
        print("Error: Couldn't find test.jsonl! Make sure you are in the root directory and Person A delivered the data.")
        return

    print("\n📊 Computing final ROUGE metrics vs baseline...")
    scores = compute_rouge(predictions, references)
    
    print("\n✅ Final ROUGE Scores:")
    print(json.dumps(scores, indent=4))
    
    # Save results to the evaluation folder for Person A
    os.makedirs("evaluation/results", exist_ok=True)
    with open("evaluation/results/final_rouge_scores.json", "w") as f:
        json.dump(scores, f, indent=4)
    print("\n💾 Saved final scores to 'evaluation/results/final_rouge_scores.json'")

if __name__ == "__main__":
    run_eval()
