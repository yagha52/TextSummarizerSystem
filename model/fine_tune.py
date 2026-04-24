import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    BitsAndBytesConfig,
    Trainer,
    DataCollatorForLanguageModeling
)
from peft import LoraConfig, get_peft_model, TaskType
import config

# -------------------------
# 1. Tokenizer
# -------------------------
tokenizer = AutoTokenizer.from_pretrained(config.MODEL_NAME, use_fast=True)
tokenizer.pad_token = tokenizer.eos_token

# -------------------------
# 2. Model (4-bit safe)
# -------------------------
quant_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_quant_type="nf4",
)

model = AutoModelForCausalLM.from_pretrained(
    config.MODEL_NAME,
    quantization_config=quant_config,
    device_map="auto",
)

# (important for stability on small GPU)
model.gradient_checkpointing_enable()

# -------------------------
# 3. LoRA
# -------------------------
lora_config = LoraConfig(
    r=config.LORA_RANK,
    lora_alpha=16,
    lora_dropout=0.05,
    bias="none",
    task_type=TaskType.CAUSAL_LM,
    target_modules=[
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj"
    ]
)

model = get_peft_model(model, lora_config)

# -------------------------
# 4. Dataset
# -------------------------
dataset = load_dataset("json", data_files={
    "train": "data/processed/train.jsonl",
    "validation": "data/processed/val.jsonl"
})

# -------------------------
# 5. Tokenization (FIXED: no max padding)
# -------------------------
def tokenize(batch):
    return tokenizer(
        batch["text"],
        truncation=True,
        max_length=config.MAX_SEQ_LEN
    )

tokenized = dataset.map(tokenize, batched=True)

# remove raw text
tokenized = tokenized.remove_columns(["text"])

# -------------------------
# 6. Data collator (handles labels automatically)
# -------------------------
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False
)

# -------------------------
# 7. Training arguments (FIXED)
# -------------------------
training_args = TrainingArguments(
    output_dir="model/checkpoints",

    per_device_train_batch_size=config.BATCH_SIZE,
    gradient_accumulation_steps=config.GRAD_ACCUM,

    learning_rate=config.LEARNING_RATE,
    num_train_epochs=config.EPOCHS,

    fp16=True,
    logging_steps=10,

    save_strategy="epoch",
    eval_strategy="epoch",

    report_to="none",

    optim="paged_adamw_8bit",
)

# -------------------------
# 8. Trainer
# -------------------------
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized["train"],
    eval_dataset=tokenized["validation"],
    data_collator=data_collator,
)

# -------------------------
# 9. Train
# -------------------------
print("🚀 Starting training...")
trainer.train()
print("✅ Done!")

# -------------------------
# 10. Save
# -------------------------
model.save_pretrained("model/checkpoints/final")
tokenizer.save_pretrained("model/checkpoints/final")

print("💾 Model saved")