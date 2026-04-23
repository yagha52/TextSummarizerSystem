import json

with open("data/processed/train.jsonl", "r") as f:
    for i, line in enumerate(f):
        sample = json.loads(line)
        if i == 0:
            print("✅ Keys found:", list(sample.keys()))
            print("✅ Sample text preview:")
            print(sample["text"][:300])
            break