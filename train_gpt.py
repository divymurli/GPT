import torch
import torch.nn as nn
import numpy as np
from model import GPT
from torch.utils.data import DataLoader
from sharded_dataset import ShardedTokenDataset
from tqdm import tqdm
import torch.optim.lr_scheduler as lr_scheduler
import tiktoken

import os

# first pass rudimentary training loop

grad_accum_steps = 8
train_batch_size = 64
eval_batch_size = 32
block_size = 512
epochs = 5
save_every = 1
check_output_every = 1
lr = 5e-4
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

save_dir = "./checkpoints"
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

criterion = nn.CrossEntropyLoss()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

train_dataset = ShardedTokenDataset("./openwebtext_abridged_sharded", 500000, block_size)
train_dataloader = DataLoader(train_dataset, batch_size=train_batch_size, shuffle=True)



model = GPT(num_blocks=2,
              vocab_size=50257,
              seq_len=block_size,
              num_heads=4,
              head_dim=4,
              dropout=0.0,
              embedding_dim=768
            )

enc = tiktoken.get_encoding("gpt2")
text = "Hey, I'm a language model. Ask me a question about "
encoded_tokens = enc.encode_ordinary(text)
print("========= GENERATED OUTPUT ============")
print(enc.decode(model.generate(torch.tensor([encoded_tokens]), 60, 100).numpy()[0].tolist()))

model = model.to(device)
# print the number of parameters in the model
print("total model params:")
print(sum(p.numel() for p in model.parameters())/1e6, 'M parameters')

# create a PyTorch optimizer
optimizer = model.configure_optimizers(weight_decay=0.1, learning_rate=6e-4, device=device)
scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)

for epoch in range(epochs):
    model.train()
    running_loss = 0.0
    progress_bar = tqdm(enumerate(train_dataloader), total=len(train_dataloader), desc=f"Epoch {epoch+1}")
    optimizer.zero_grad()  # Start with zeroed gradients
    losses = []
    for i, batch in progress_bar:

        if i % 1000 == 0:
            print("========= GENERATED OUTPUT ============")
            print(enc.decode(model.generate(torch.tensor([encoded_tokens]), 60, 100).numpy()[0].tolist()))

        inputs = batch[0].to(device)
        targets = batch[1].to(device)

        logits =  model(inputs)
        loss = criterion(logits.view(logits.shape[0]*logits.shape[1], logits.shape[2]), targets.view(-1))
        
        loss /= grad_accum_steps
        loss.backward()

        if i % grad_accum_steps == 0:
            optimizer.step()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1)
            optimizer.zero_grad()

        running_loss += loss.item() * grad_accum_steps  # Revert loss normalization for display
        progress_bar.set_postfix({'loss': running_loss / (i + 1)})

    # Step the learning rate scheduler at the end of the epoch
    scheduler.step()

    print(f"Epoch {epoch+1} finished. Mean loss: {running_loss / len(train_dataloader):.4f}")

    if (epoch + 1) % save_every == 0:
        torch.save({
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
        }, f"{save_dir}/model_epoch_{epoch + 1}.pth")

import ipdb
ipdb.set_trace()












