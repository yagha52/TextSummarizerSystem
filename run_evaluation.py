# run_evaluation.py
"""
Evaluates the fine-tuned model (model/checkpoints/final) against the
held-out test set using ROUGE scores, and compares them to the Lead-3 baseline.

Usage:
    # Full evaluation (all ~1000 test samples — slow, ~10-20 min on CPU)
    python run_evaluation.py

    # Quick smoke-test (first 50 samples — much faster)
    python run_evaluation.py --quick
"""
import json
import argparse
import os
import sys

# -----------------------------------------------------------------------
# Paths
# -----------------------------------------------------------------------

from evaluation.rouge_utils import compute_rouge           # renamed from evaluate.py to avoid circular import

TEST_FILE  = "data/processed/test.jsonl"
RESULTS_DIR = "evaluation/results"

# -----------------------------------------------------------------------
# Import predict.py summarise function (loads model once)
# -----------------------------------------------------------------------
print("⏳ Loading fine-tuned model from model/checkpoints/final …")
from predict import summarize
print("✅ Model loaded.\n")


# -----------------------------------------------------------------------
# Lead-3 baseline (no model needed)
# -----------------------------------------------------------------------
def lead3(article: str) -> str:
    """First 3 sentences of the article — classic extractive baseline."""
    import re
    sentences = re.split(r'(?<=[.!?])\s+', article.strip())
    return " ".join(sentences[:3])


# -----------------------------------------------------------------------
# Main evaluation loop
# -----------------------------------------------------------------------
def run_eval(quick: bool = False):
    if not os.path.exists(TEST_FILE):
        print(f"❌ Error: '{TEST_FILE}' not found.")
        print("   Make sure you are running from the project root and that")
        print("   data/preprocess.py has already been run by Person A.")
        return

    model_preds, baseline_preds, references = [], [], []
    limit = 50 if quick else None

    print(f"{'⚡ QUICK MODE — ' if quick else ''}Evaluating on "
          f"{'first 50' if quick else 'all'} test samples …\n")

    with open(TEST_FILE, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if limit and i >= limit:
                break

            sample = json.loads(line)
            text   = sample["text"]

            # ── Parse article & reference ──────────────────────────────
            # Format (from preprocess.py):
            #   ### Instruction:\n...\n\n### Article:\n{article}\n\n### Summary:\n{summary}
            try:
                article = text.split("### Article:\n")[1].split("\n\n### Summary:\n")[0].strip()
                ref     = text.split("### Summary:\n")[1].strip()
            except IndexError:
                print(f"  ⚠️  Skipping malformed sample {i+1}")
                continue

            # ── Model prediction ───────────────────────────────────────
            print(f"  [{i+1:>4}] Generating summary …", end="\r")
            pred = summarize(article)

            if not pred or pred.startswith("Could not"):
                print(f"\n  ⚠️  Empty prediction for sample {i+1}, skipping.")
                continue

            model_preds.append(pred)
            baseline_preds.append(lead3(article))
            references.append(ref)

    if not model_preds:
        print("❌ No valid predictions were produced. Aborting.")
        return

    print(f"\n\n📊 Computing ROUGE scores for {len(model_preds)} samples …\n")

    model_scores    = compute_rouge(model_preds,    references)
    baseline_scores = compute_rouge(baseline_preds, references)

    # ── Pretty-print comparison ────────────────────────────────────────
    header = f"{'Metric':<12} {'Fine-tuned':>12} {'Lead-3 Base':>12}  {'Δ':>8}"
    print("=" * len(header))
    print(header)
    print("=" * len(header))
    for metric in ["rouge1", "rouge2", "rougeL", "rougeLsum"]:
        m = model_scores.get(metric, 0)
        b = baseline_scores.get(metric, 0)
        delta = m - b
        sign  = "+" if delta >= 0 else ""
        print(f"{metric:<12} {m:>12.4f} {b:>12.4f}  {sign}{delta:>7.4f}")
    print("=" * len(header))

    # ── Save results ───────────────────────────────────────────────────
    os.makedirs(RESULTS_DIR, exist_ok=True)

    tag = "quick" if quick else "final"

    model_path    = f"{RESULTS_DIR}/{tag}_rouge_scores.json"
    baseline_path = f"{RESULTS_DIR}/{tag}_baseline_scores.json"

    combined = {
        "fine_tuned_model": model_scores,
        "lead3_baseline":   baseline_scores,
        "delta":            {k: round(model_scores.get(k, 0) - baseline_scores.get(k, 0), 6)
                             for k in model_scores}
    }
    combined_path = f"{RESULTS_DIR}/{tag}_comparison.json"

    with open(model_path,    "w") as f: json.dump(model_scores,    f, indent=4)
    with open(baseline_path, "w") as f: json.dump(baseline_scores, f, indent=4)
    with open(combined_path, "w") as f: json.dump(combined,        f, indent=4)

    print(f"\n💾 Results saved to '{RESULTS_DIR}/':")
    print(f"   • {tag}_rouge_scores.json    — fine-tuned model scores")
    print(f"   • {tag}_baseline_scores.json — Lead-3 baseline scores")
    print(f"   • {tag}_comparison.json      — side-by-side comparison\n")


# -----------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate the fine-tuned summarisation model.")
    parser.add_argument(
        "--quick", action="store_true",
        help="Run on only the first 50 test samples for a fast sanity-check."
    )
    args = parser.parse_args()
    run_eval(quick=args.quick)
