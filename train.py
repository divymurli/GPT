import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader
from dataset import get_data, GPTDataset
from model import GPT
from tqdm import tqdm
import os

# basic hparams
train_batch_size = 64
eval_batch_size = 32
block_size = 32
epochs = 5
save_every = 1
check_output_every = 1
lr = 5e-4

save_dir = "./checkpoints"
if not os.path.exists(save_dir):
  os.makedirs(save_dir)

criterion = nn.CrossEntropyLoss()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

text = get_data()
chars = sorted(list(set(text)))
vocab_size = len(chars)
stoi = { ch:i for i,ch in enumerate(chars) }
itos = { i:ch for i,ch in enumerate(chars) }
encode = lambda s: [stoi[c] for c in s] # encoder: take a string, output a list of integers
decode = lambda l: ''.join([itos[i] for i in l]) # decoder: take a list of integers, output a string

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
print("total model params:")
print(sum(p.numel() for p in model.parameters())/1e6, 'M parameters')

# create a PyTorch optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

# get a model output before any training
print("untrained output:")
start_token = torch.zeros((1, 1), dtype=torch.long, device=device)
print(decode(model.generate(start_token, max_lookback_tokens=block_size, max_new_tokens=100)[0].tolist()))

# training loop
for epoch in range(epochs):
  losses = []
  for batch in tqdm(train_dataloader):
    encoded_inputs = batch['encoded_input'].to(device)
    encoded_targets = batch['encoded_target'].to(device)
    logits = model(encoded_inputs)

    loss = criterion(logits.view(logits.shape[0]*logits.shape[1], logits.shape[2]), encoded_targets.view(-1))
    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1)
    optimizer.step()
    losses.append(loss.item())
  mean_epoch_train_loss = np.mean(losses)

  if epoch % check_output_every == 0:
    print("what're we outputting?")
    model.eval()
    start_token = torch.zeros((1, 1), dtype=torch.long, device=device)
    print(decode(model.generate(start_token, max_lookback_tokens=block_size, max_new_tokens=100)[0].tolist()))

  if epoch % save_every == 0:
    torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': mean_epoch_train_loss,
            }, f"{save_dir}/rudimentary_gpt_epoch_{epoch}")

  print(f"Epoch {epoch} train loss: {mean_epoch_train_loss}")




