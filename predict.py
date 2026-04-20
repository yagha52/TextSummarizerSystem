import requests

OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL_NAME = "summarizer"  # Person B's fine-tuned model name

def summarize(article: str) -> str:
    """Takes an article text and returns the summary via Ollama API."""
    prompt = (
        "### Instruction:\nSummarize the following news article in 2-3 sentences.\n\n"
        f"### Article:\n{article}\n\n### Summary:\n"
    )
    
    payload = {
        "model": MODEL_NAME,
        "prompt": prompt,
        "stream": False
    }
    
    try:
        response = requests.post(OLLAMA_URL, json=payload)
        response.raise_for_status()
        return response.json().get("response", "").strip()
    except requests.exceptions.RequestException as e:
        return f"Error connecting to Ollama: {e}"

if __name__ == "__main__":
    print("--- Text Summarizer Inference ---")
    test_article = input("Paste your article here (press Enter when done, note multi-line might need tweaking depending on terminal):\n")
    if test_article.strip():
        print("\n⏳ Generating Summary...")
        summary = summarize(test_article)
        print("\n--- Generated Summary ---")
        print(summary)
    else:
        print("Empty article provided.")
