import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel, AdamW
from torch.utils.data import DataLoader
from torch.nn.utils import clip_grad_norm_

# Set device (GPU or CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Tokenizer
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')

# Set custom pad token
tokenizer.pad_token = tokenizer.eos_token

# Load data directly from your dataset file
dataset_file_path = "C:\\Users\\hp\\OneDrive\\Desktop\\dialogs.txt"
dialogues = []

with open(dataset_file_path, 'r', encoding='utf-8') as file:
    dialogues = file.read().splitlines()

# Tokenize the dialogues
inputs = tokenizer(dialogues, padding=True, truncation=True, max_length=128, return_tensors='pt')
inputs.to(device)

# Initialize model
model = GPT2LMHeadModel.from_pretrained('gpt2')
model.to(device)

# Initialize optimizer using AdamW from PyTorch
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)

# Training loop
epochs = 3
for epoch in range(epochs):
    optimizer.zero_grad()
    
    # Forward pass
    outputs = model(input_ids=inputs['input_ids'], attention_mask=inputs['attention_mask'], labels=inputs['input_ids'])
    loss = outputs.loss
    loss.backward()
    
    # Gradient clipping
    clip_grad_norm_(model.parameters(), max_norm=1.0)
    
    optimizer.step()
    print(f'Epoch {epoch + 1}, Loss: {loss.item()}')

# Save the trained model
model.save_pretrained("fine_tuned_chatbot_model")
tokenizer.save_pretrained("fine_tuned_chatbot_model")
