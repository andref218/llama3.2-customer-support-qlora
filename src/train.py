"""
Fine-tune Llama 3.2 using QLoRA on the Bitext Customer Support dataset.

This script:
- Loads the customer support dataset.
- Formats conversations using the Llama 3.2 chat template.
- Loads the model in 4-bit precision.
- Applies QLoRA adapters.
- Fine-tunes the model.
- Saves the trained LoRA adapters.
"""

# ==========================================================
# Import Libraries
# ==========================================================

import os

import torch
import wandb

# Hugging Face
from datasets import DatasetDict, load_dataset
from huggingface_hub import login
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
)

# PEFT and TRL
from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
)
from trl import SFTConfig, SFTTrainer

# Project utilities
from src.formatting import format_example


# ==========================================================
# Configuration
# ==========================================================

BASE_MODEL = "meta-llama/Llama-3.2-3B-Instruct"

PROJECT_NAME = "llm_customer_support_finetuning"

DATASET_NAME = "bitext/Bitext-customer-support-llm-chatbot-training-dataset"


# ==========================================================
# Authentication
# ==========================================================

HF_TOKEN = os.getenv("HF_TOKEN")
WANDB_API_KEY = os.getenv("WANDB_API_KEY")

if HF_TOKEN is None:
    raise ValueError("HF_TOKEN environment variable is not set.")

if WANDB_API_KEY is None:
    raise ValueError("WANDB_API_KEY environment variable is not set.")

login(HF_TOKEN, add_to_git_credential=True)

wandb.login(key=WANDB_API_KEY)

# ==========================================================
# Configure Weights & Biases
# ==========================================================

os.environ["WANDB_PROJECT"] = PROJECT_NAME
os.environ["WANDB_LOG_MODEL"] = "false"
os.environ["WANDB_WATCH"] = "false"


# ==========================================================
# Load Dataset
# ==========================================================

dataset = load_dataset(DATASET_NAME)

# ==========================================================
# Configure 4-bit Quantization
# ==========================================================

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
)

# ==========================================================
# Load Tokenizer
# ==========================================================

tokenizer = AutoTokenizer.from_pretrained(
    BASE_MODEL,
    trust_remote_code=True,
)

tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

# ==========================================================
# Load Base Model
# ==========================================================

base_model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    quantization_config=bnb_config,
    device_map="auto",
)

base_model.generation_config.pad_token_id = tokenizer.pad_token_id

print(f"Memory footprint: {base_model.get_memory_footprint() / 1e6:.1f} MB")

# ==========================================================
# Split Dataset

# The original dataset only provides a training split.
# We divide it into training, validation, and test sets to:
# - Train the model,
# - Monitor validation performance during fine-tuning,
# - Evaluate generalization on unseen conversations.

# ==========================================================

train_test = dataset["train"].train_test_split(
    test_size=0.2,
    seed=42,
)

validation_test = train_test["test"].train_test_split(
    test_size=0.5,
    seed=42,
)

dataset_split = DatasetDict(
    {
        "train": train_test["train"],
        "validation": validation_test["train"],
        "test": validation_test["test"],
    }
)

print(dataset_split)

# ==========================================================
# Format Dataset

# Convert each customer support conversation into the
# Llama 3.2 chat template expected during supervised fine-tuning.

# ==========================================================

formatted_dataset = dataset_split.map(
    lambda example: format_example(example, tokenizer)
)

print(formatted_dataset)
print(formatted_dataset["train"].column_names)

# ==========================================================
# Prepare the quantized model for QLoRA fine-tuning
# ==========================================================

base_model = prepare_model_for_kbit_training(base_model)

# ==========================================================
# Configure LoRA

# Configure the Low-Rank Adaptation (LoRA) modules that
# will be trained while keeping the base model frozen.

# ==========================================================

lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)

# ==========================================================
# Apply LoRA Adapters to the base model
# ==========================================================

base_model = get_peft_model(
    base_model,
    lora_config,
)

print(base_model)

# ==========================================================
# Verify Trainable Parameters
# ==========================================================

trainable = sum(
    p.numel() for p in base_model.parameters()
    if p.requires_grad
)

total = sum(
    p.numel() for p in base_model.parameters()
)

print(f"Trainable parameters: {trainable:,}")
print(f"Total parameters: {total:,}")
print(f"Trainable percentage: {100 * trainable / total:.4f}%")