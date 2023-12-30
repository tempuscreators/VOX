import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
import pandas as pd
from sklearn.model_selection import train_test_split
import logging
import numpy as np
import logging

# Load model and tokenizer
model_name = "mistralai/Mistral-7B-Instruct-v0.1"
model = AutoModelForSequenceClassification.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Debugging: Print tokenizer's initial configuration
print("Initial tokenizer config:", tokenizer)

# Set padding token explicitly
if tokenizer.pad_token is None:
    print("Setting new padding token.")
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})
else:
    print("Tokenizer already has padding token:", tokenizer.pad_token)

# Resize model's token embeddings
model.resize_token_embeddings(len(tokenizer))

# Set model's padding token
model.config.pad_token_id = tokenizer.pad_token_id

# Debugging: Print tokenizer's updated configuration
print("Updated tokenizer config:", tokenizer)

# Rest of your script...


# Basic logging setup for file
logging.basicConfig(filename='training_log.txt', level=logging.INFO, 
                    format='%(asctime)s - %(levelname)s - %(message)s')


# Tokenization function
def tokenize_data(text):
    return tokenizer(text, padding='max_length', truncation=True, max_length=512)

# Custom Dataset class
class MyDataset(Dataset):
    def __init__(self, texts, labels=None):
        self.encodings = tokenizer(texts, padding=True, truncation=True, max_length=512, return_tensors="pt")
        self.labels = torch.tensor(labels) if labels is not None else torch.zeros(len(texts), dtype=torch.long)

    def __getitem__(self, idx):
        item = {key: val[idx] for key, val in self.encodings.items()}
        item['labels'] = self.labels[idx]
        return item

    def __len__(self):
        return len(self.labels)



# Load CSV data
csv_file = 'C:\\Users\\ejhaw\\VoxPromtTuning.csv'  # Replace with your CSV file path
data = pd.read_csv(csv_file)

# Extract 'GeneratedPrompt' and optionally 'Labels' if available
prompts = data['GeneratedPrompt'].tolist()
labels = [0] * len(prompts)  # Assuming dummy labels

# Split dataset into training and validation sets
train_prompts, val_prompts, train_labels, val_labels = train_test_split(prompts, labels, test_size=0.2)

# Create instances of MyDataset for training and validation
train_dataset = MyDataset(train_prompts, train_labels)
val_dataset = MyDataset(val_prompts, val_labels)

# Creating DataLoaders
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=16)

# Define training parameters
learning_rate = 1e-5
num_epochs = 3
optimizer = AdamW(model.parameters(), lr=learning_rate)

# Early stopping parameters
early_stopping_patience = 2
best_val_loss = np.inf
no_improvement_epochs = 0

# Basic logging setup
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Training Loop with Early Stopping
model.train()
for epoch in range(num_epochs):
    total_train_loss = 0
    for batch in train_loader:
        optimizer.zero_grad()
        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask']
        labels = batch['labels']
        outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        total_train_loss += loss.item()
        loss.backward()
        optimizer.step()

    avg_train_loss = total_train_loss / len(train_loader)
    logging.info(f'Epoch {epoch + 1} Average Training Loss: {avg_train_loss}')

    # Validation Loop
    model.eval()
    total_val_loss = 0
    with torch.no_grad():
        for batch in val_loader:
            input_ids = batch['input_ids']
            attention_mask = batch['attention_mask']
            labels = batch['labels']
            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            total_val_loss += loss.item()

    avg_val_loss = total_val_loss / len(val_loader)
    logging.info(f'Epoch {epoch + 1} Average Validation Loss: {avg_val_loss}')

    # Early Stopping Check
    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        no_improvement_epochs = 0
    else:
        no_improvement_epochs += 1
        if no_improvement_epochs >= early_stopping_patience:
            logging.info(f'Early stopping triggered after epoch {epoch + 1}')
            break

# Save model
model.save_pretrained('./my_mistral_model')
tokenizer.save_pretrained('./my_mistral_model')
