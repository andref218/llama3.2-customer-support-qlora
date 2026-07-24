"""
Demonstrates how to format a customer support conversation using the
Llama 3.2 chat template.

This script:
- Loads the Llama 3.2 tokenizer.
- Loads a sample from the Bitext Customer Support dataset.
- Converts the instruction-response pair into a chat conversation.
- Applies the model's chat template.
- Prints the formatted conversation before tokenization.
"""

import os

from dotenv import load_dotenv
from transformers import AutoTokenizer
from datasets import load_dataset

load_dotenv()

HF_TOKEN = os.getenv("HF_TOKEN")

if HF_TOKEN is None:
    raise ValueError("HF_TOKEN not found in environment variables.")

MODEL_NAME = "meta-llama/Llama-3.2-3B-Instruct"

# Load the Llama 3.2 tokenizer
tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME,
    token=HF_TOKEN
)

# Load dataset
dataset = load_dataset(
    "bitext/Bitext-customer-support-llm-chatbot-training-dataset",
)

example = dataset["train"][0]

# Convert the dataset example into a chat conversation
messages = [
    {
        "role": "user",
        "content": example["instruction"],
    },
    {
        "role": "assistant",
        "content": example["response"],
    },
]

# Apply the Llama 3.2 chat template
formatted_chat = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
)

# Display the formatted conversation
print(formatted_chat)