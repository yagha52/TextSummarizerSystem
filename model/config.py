# Base model
MODEL_NAME = "unsloth/Llama-3.2-3B-bnb-4bit"

# MUST match actual model limit
MAX_SEQ_LEN = 384

# Training hyperparameters
BATCH_SIZE = 1
GRAD_ACCUM = 4
LEARNING_RATE = 2e-4
EPOCHS = 1
LORA_RANK = 16