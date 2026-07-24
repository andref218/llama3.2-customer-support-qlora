from datasets import load_dataset

# Load the dataset from Hugging Face
dataset = load_dataset("bitext/Bitext-customer-support-llm-chatbot-training-dataset")

train_dataset = dataset["train"]

# Print the first 5 examples
for i in range(5):
    print(f"\nExample {i}")
    print(train_dataset[i]) 