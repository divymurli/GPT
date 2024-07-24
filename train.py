import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from dataset import get_data, GPTDataset
from model import GPT
import collections
from tqdm import tqdm
import ipdb

# basic hparams
train_batch_size = 2048
eval_batch_size = 32
block_size = 32
num_epochs = 5
lr = 1e-3
criterion = nn.CrossEntropyLoss()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

text = get_data()

train_text = text[:int(0.9*len(text))]
val_text = text[int(0.9*len(text)):]

train_dataset = GPTDataset(train_text, block_size=block_size)
val_dataset = GPTDataset(val_text, block_size=block_size)

train_dataloader = DataLoader(train_dataset, batch_size=train_batch_size, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=eval_batch_size, shuffle=False)

model = GPT(num_blocks=4,
              vocab_size=len(train_dataset.chars),
              seq_len=block_size,
              num_heads=4,
              head_dim=16,
              dropout=0.0,
              embedding_dim=64
            )

model = model.to(device)
# print the number of parameters in the model
print(sum(p.numel() for p in model.parameters())/1e6, 'M parameters')

# create a PyTorch optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

epochs = 10

# training loop
for epoch in range(epochs):
  for batch in tqdm(train_dataloader):
    encoded_inputs = batch['encoded_input'].to(device)
    encoded_targets = batch['encoded_target'].to(device)

    logits = model(encoded_inputs)
    # add logic for 
    loss = criterion(logits.view(logits.shape[0]*logits.shape[1], logits.shape[2]), encoded_targets.view(-1))
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()



