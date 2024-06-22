#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 14 22:46:58 2024

@author: saiful
"""
#%%  Load dataset
import os
os.environ["CUDA_VISIBLE_DEVICES"]="3 "
import torch
from transformers import AutoModel, AutoTokenizer
from datasets import load_dataset
from torch.utils.data import DataLoader, Dataset

# Load the BERT model and tokenizer
model = AutoModel.from_pretrained("bert-base-uncased")
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

# Example input
inputs = tokenizer(['Hello world', 'Hi how are you'], padding=True, truncation=True, return_tensors='pt')
inputs
inputs.keys()

# Pass input through the model
output = model(**inputs)
output

# Load dataset
emotions = load_dataset('SetFit/emotion')
print('flag 1.1 emotions', emotions)

# Print the features of the 'train' dataset
train_features = emotions['train'].features
print("Features of 'train' dataset:", train_features)

emotions['train'].features['text']
emotions['train'][:3]['text']

emotions['train'][:3]['label']
emotions['train'][:3]['label_text']

# Print some keys and values of the nested dictionary
for split in emotions.keys():
    print(f"Split: {split}")
    print("Features:", emotions[split].features)
    print("Example data:", emotions[split][0])
    
# Print keys and values for the 'train' split
print(f"Keys in 'train' split: {emotions['train'].features}")
for i in range(3):  # printing first 3 examples for brevity
    print(f"Example {i} in 'train' split: {emotions['train'][i]}")
    
# Tokenization function
def tokenize2(batch):
    return tokenizer(batch["text"], padding=True, truncation=True)

# Apply tokenization and add the new fields to the dataset
emotions_encoded = emotions.map(tokenize2, batched=True, remove_columns=["text"])
print('flag 1.2 emotions_encoded', emotions_encoded)
print('flag 1.2 emotions_encoded.keys()', emotions_encoded.keys())

print(emotions_encoded['train'][:3]['label'])
print(emotions_encoded['train'][:3]['label_text'])
print(emotions_encoded['train'][:3]['input_ids'])
print(emotions_encoded['train'][:3]['token_type_ids'])
print(emotions_encoded['train'][:3]['attention_mask'])

#%%
# Custom dataset class for PyTorch
class EmotionDataset(Dataset):
    def __init__(self, encoded_dataset):
        self.encoded_dataset = encoded_dataset

    def __len__(self):
        return len(self.encoded_dataset)
    
    def __getitem__(self, idx):
        item = {
            "input_ids": torch.tensor(self.encoded_dataset["input_ids"][idx]),
            "attention_mask": torch.tensor(self.encoded_dataset["attention_mask"][idx]),
            "token_type_ids": torch.tensor(self.encoded_dataset["token_type_ids"][idx]),
            "label": torch.tensor(self.encoded_dataset["label"][idx])
        }
        return item

# Assuming 'emotions_encoded' is your dataset after tokenization and encoding
train_dataset = EmotionDataset(emotions_encoded['train'])
test_dataset = EmotionDataset(emotions_encoded['test'])

# Custom collate function for DataLoader
def collate_fn(batch):
    # Stack input_ids, attention_mask, and token_type_ids, pad them to the same length
    input_ids = [item['input_ids'] for item in batch]
    attention_mask = [item['attention_mask'] for item in batch]
    token_type_ids = [item['token_type_ids'] for item in batch]
    labels = [item['label'] for item in batch]

    input_ids = [torch.tensor(seq) for seq in input_ids]
    attention_mask = [torch.tensor(seq) for seq in attention_mask]
    token_type_ids = [torch.tensor(seq) for seq in token_type_ids]

    input_ids_padded = torch.nn.utils.rnn.pad_sequence(input_ids, batch_first=True, padding_value=tokenizer.pad_token_id)
    attention_mask_padded = torch.nn.utils.rnn.pad_sequence(attention_mask, batch_first=True, padding_value=0)
    token_type_ids_padded = torch.nn.utils.rnn.pad_sequence(token_type_ids, batch_first=True, padding_value=0)
    labels = torch.tensor(labels, dtype=torch.long)

    return {
        'input_ids': input_ids_padded,
        'attention_mask': attention_mask_padded,
        'token_type_ids': token_type_ids_padded,
        'labels': labels
    }

BATCH_SIZE = 3
train_loader = DataLoader(emotions_encoded['train'], batch_size=8, shuffle=True, collate_fn=collate_fn)
test_loader = DataLoader(emotions_encoded['test'], batch_size=8, shuffle=False, collate_fn=collate_fn)

# Test the DataLoader
for batch in train_loader:
    print(batch)
    break

for batch in test_loader:
    print(batch)
    break

#%%
# Custom model for classification
class BERTForClassification(torch.nn.Module):
    def __init__(self, bert_model, num_classes):
        super(BERTForClassification, self).__init__()
        self.bert = bert_model
        self.fc = torch.nn.Linear(bert_model.config.hidden_size, num_classes)

    def forward(self, input_ids, attention_mask=None, token_type_ids=None):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        x = outputs[1]
        return self.fc(x)

# Instantiate the model
classifier = BERTForClassification(model, num_classes=6)

# Training setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
classifier.to(device)
optimizer = torch.optim.Adam(classifier.parameters(), lr=1e-5)
criterion = torch.nn.CrossEntropyLoss()

#%%

# Training loop
num_epochs = 3
for epoch in range(num_epochs):
    classifier.train()
    for batch in train_loader:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        token_type_ids = batch['token_type_ids'].to(device)
        labels = batch['labels'].to(device)

        optimizer.zero_grad()
        outputs = classifier(input_ids, attention_mask, token_type_ids)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

    print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {loss.item()}")

# Evaluation
classifier.eval()
correct = 0
total = 0
with torch.no_grad():
    for batch in test_loader:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        token_type_ids = batch['token_type_ids'].to(device)
        labels = batch['labels'].to(device)

        outputs = classifier(input_ids, attention_mask, token_type_ids)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(f'Accuracy: {100 * correct / total}%')

#%%

# Training loop
num_epochs = 3
for epoch in range(num_epochs):
    classifier.train()
    train_loss = 0.0
    correct_train = 0
    total_train = 0

    for batch in train_loader:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        token_type_ids = batch['token_type_ids'].to(device)
        labels = batch['labels'].to(device)

        optimizer.zero_grad()
        outputs = classifier(input_ids, attention_mask, token_type_ids)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total_train += labels.size(0)
        correct_train += (predicted == labels).sum().item()

    train_accuracy = 100 * correct_train / total_train
    avg_train_loss = train_loss / len(train_loader)
    print(f"Epoch {epoch + 1}/{num_epochs}, Training Accuracy: {train_accuracy}%")

    # Evaluation on the test set
    classifier.eval()
    correct_test = 0
    total_test = 0
    test_loss = 0.0
    with torch.no_grad():
        for batch in test_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            token_type_ids = batch['token_type_ids'].to(device)
            labels = batch['labels'].to(device)

            outputs = classifier(input_ids, attention_mask, token_type_ids)
            loss = criterion(outputs, labels)
            test_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total_test += labels.size(0)
            correct_test += (predicted == labels).sum().item()

    test_accuracy = 100 * correct_test / total_test
    avg_test_loss = test_loss / len(test_loader)
    print(f"Epoch {epoch + 1}/{num_epochs}, Testing Accuracy: {test_accuracy}%")

print("Training and evaluation completed.")
