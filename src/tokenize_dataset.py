"""
Tokenize the customer support dataset for Llama 3.2 fine-tuning.

This script:
- Loads the Bitext customer support dataset.
- Formats each instruction-response pair using the Llama 3.2 chat template.
- Tokenizes the formatted conversations into input IDs and attention masks.
- Applies padding and truncation to create fixed-length training examples.

The resulting tokenized dataset is ready to be used for supervised fine-tuning (SFT).
"""

from datasets import load_dataset
from transformers import AutoTokenizer
from dotenv import load_dotenv
import os

load_dotenv()

HF_TOKEN = os.getenv("HF_TOKEN")

MODEL_NAME = "meta-llama/Llama-3.2-3B-Instruct"

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME,
    token=HF_TOKEN,
)
tokenizer.pad_token = tokenizer.eos_token

dataset = load_dataset(
    "bitext/Bitext-customer-support-llm-chatbot-training-dataset"
)

def format_example(example):
    messages = [
        {"role": "user", "content": example["instruction"]},
        {"role": "assistant", "content": example["response"]},
    ]

    example["text"] = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
    )

    return example

formatted_dataset = dataset.map(format_example)

# Tokenize the formatted conversations
def tokenize_function(example):
    return tokenizer(
        example["text"],
        truncation=True,
        padding="max_length",
        max_length=512,
    )

# Apply tokenization to the dataset
tokenized_dataset = formatted_dataset.map(tokenize_function)

print(tokenized_dataset["train"][0])