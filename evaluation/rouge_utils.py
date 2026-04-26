import evaluate
import json
import os

# Load rouge
rouge = evaluate.load("rouge")

def compute_rouge(predictions: list, references: list):
    """Compute ROUGE scores given predictions and references lists."""
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
    print("Dummy test scores:", scores)
    
    # Save the dummy results to demonstrate
    os.makedirs("evaluation/results", exist_ok=True)
    with open("evaluation/results/dummy_scores.json", "w") as f:
        json.dump(scores, f, indent=4)
        
    print("Saved dummy scores to evaluation/results/dummy_scores.json")
