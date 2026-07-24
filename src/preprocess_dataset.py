"""
Preprocesses the Bitext customer support dataset for supervised fine-tuning with
Llama 3.2.

This script:
- Loads the customer support dataset.
- Converts each instruction-response pair into a chat conversation.
- Applies the Llama 3.2 chat template.
- Adds a new `text` field containing the formatted conversation, ready for
  tokenization.
"""

import os

from datasets import load_dataset
from dotenv import load_dotenv
from transformers import AutoTokenizer

# Load environment variables
load_dotenv()

HF_TOKEN = os.getenv("HF_TOKEN")

if HF_TOKEN is None:
    raise ValueError("HF_TOKEN not found in environment variables.")

MODEL_NAME = "meta-llama/Llama-3.2-3B-Instruct"

# Load the tokenizer
tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME,
    token=HF_TOKEN,
)

# Load the dataset
dataset = load_dataset(
    "bitext/Bitext-customer-support-llm-chatbot-training-dataset"
)


# Format a dataset example using the Llama chat template
def format_example(example):
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

    example["text"] = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
    )

    return example


# Apply the formatting to the entire dataset
formatted_dataset = dataset.map(format_example)

# Display one processed example
print(formatted_dataset["train"][0]["text"])