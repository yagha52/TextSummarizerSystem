# explore_data.py
from datasets import load_dataset

def explore():
    print("Loading CNN/DailyMail dataset...")
    # Load CNN/DailyMail dataset (version 3.0.0)
    dataset = load_dataset("cnn_dailymail", "3.0.0")

    print(f"Dataset splits: {dataset.keys()}")
    print(f"Number of training samples: {dataset['train'].num_rows}")
    print(f"Number of validation samples: {dataset['validation'].num_rows}")
    print(f"Number of test samples: {dataset['test'].num_rows}")
    
    print("\n--- Example Article ---")
    print(dataset["train"][0]["article"][:500])
    
    print("\n--- Example Summary ---")
    print(dataset["train"][0]["highlights"])

if __name__ == "__main__":
    explore()
