"""
Utility functions for formatting datasets for supervised fine-tuning.

This module contains reusable functions for converting instruction-response
examples into the chat format expected by Llama models using the tokenizer's
chat template.
"""


def format_example(example, tokenizer):
    """
    Format a dataset example into a Llama chat conversation.

    Args:
        example (dict): Dataset example containing an instruction and response.
        tokenizer: Hugging Face tokenizer with a chat template.

    Returns:
        dict: Example with an additional 'text' field.
    """

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
        add_generation_prompt=False,
    )

    return example