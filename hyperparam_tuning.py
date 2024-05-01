from functions_text_model import *
import os


from transformers import BertTokenizer
from torch.utils.data import Dataset, DataLoader
import json
from transformers import BertForSequenceClassification, BertTokenizer
from torch.optim import AdamW
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
from torch.utils.data import random_split
from sklearn.model_selection import ParameterGrid


# Load the BERT tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)

# Function to encode the text
def encode_text(text):
    encoded_dict = tokenizer.encode_plus(
                        text,                      # Input text
                        add_special_tokens = True, # Add '[CLS]' and '[SEP]'
                        max_length = 64,           # Pad & truncate all sentences
                        truncation = True,
                        padding = 'max_length',
                        return_attention_mask = True,   # Construct attention masks
                        return_tensors = 'pt',     # Return pytorch tensors
                   )
    return encoded_dict['input_ids'], encoded_dict['attention_mask']

# PyTorch Dataset
class SarcasmDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        utterance = item['utterance']
        sarcasm = int(item['sarcasm'])
        input_ids, attention_mask = encode_text(utterance)
        return input_ids.flatten(), attention_mask.flatten(), sarcasm
# Create the DataLoader
# Load the data from the JSON file
with open('sarcasm_data.json') as f:
    data = json.load(f)

# Convert the data to a list of dictionaries
data = list(data.values())

dataset = SarcasmDataset(data)


# Define the hyperparameters to tune
param_grid = {
    'lr': [1e-3, 1e-4, 1e-5],
    'num_epochs': [5],
    'batch_size': [8, 16, 32, 64]
}

# Set device
device = torch.device("cpu")

# Create a parameter grid
grid = ParameterGrid(param_grid)

# Initialize a list to store the results
results = []

# For each combination of hyperparameters
for params in grid:
    # Create a new model
    model = BertForSequenceClassification.from_pretrained(
        "bert-base-uncased",
        num_labels = 2,
        output_attentions = False,
        output_hidden_states = False,
    )

    model.to(device)

    # Create a new optimizer with the current learning rate
    optimizer = AdamW(model.parameters(), lr=params['lr'])
    # Create the optimizer  
    criterion = nn.CrossEntropyLoss()

    metrics = {'ACC': acc, 'F1-weighted': f1}
    # Define the size of the training set and the test set
    train_size = int(0.8 * len(dataset))  # 80% of the data for training
    test_size = len(dataset) - train_size  # 20% of the data for testing

    # Split the dataset
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

    # Create the DataLoaders
    train_dataloader = DataLoader(train_dataset, batch_size=params['batch_size'])
    test_dataloader = DataLoader(test_dataset, batch_size=params['batch_size'])


    # Train and evaluate the model for the current number of epochs
    for epoch in range(params['num_epochs']):
        train_loss, train_metrics = train_epoch(model, optimizer, criterion, metrics, train_dataloader, device)
        eval_loss, eval_metrics = evaluate(model, criterion, metrics, test_dataloader, device)

    # Store the results
    results.append({
        'lr': params['lr'],
        'batch_size': params['batch_size'],
        'num_epochs': params['num_epochs'],
        'eval_loss': eval_loss,
        'eval_metrics': eval_metrics
    })

# Save the results to a JSON file
with open('results.json', 'w') as f:
    json.dump(results, f)

print(results)