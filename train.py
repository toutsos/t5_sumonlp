import json
from tokenize_data import tokenize_data
from dataset import CustomDataset, collate_fn  # Ensure this matches your actual class name
from model import train_model
from torch.utils.data import DataLoader

# Paths to your data
input_file = 'input_sentences_500k.txt'
output_file = 'output_logical_500k.txt'
tokenized_output_file = 'tokenized_data_small.json'

# Step 1: Tokenize the data (uncomment if needed)
# tokenize_data(input_file, output_file, tokenized_output_file)

# Step 2: Load tokenized data
try:
    with open(tokenized_output_file, 'r') as f:
        tokenized_data = json.load(f)
    print("Tokenized data loaded successfully.")
except FileNotFoundError:
    print(f"Error: The file {tokenized_output_file} was not found.")
    exit(1)

# Step 3: Create dataset
dataset = CustomDataset(tokenized_data)  # Ensure this matches your actual class name
print("Dataset created successfully.")

# Step 4: Create DataLoader
train_loader = DataLoader(dataset, batch_size=32, collate_fn=collate_fn, shuffle=True)

# Step 5: Train the model
model = train_model(train_loader)
print("Model trained successfully.")


