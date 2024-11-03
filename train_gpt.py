import torch
import torch.nn as nn
from model import GPT, GPTConfig
from torch.utils.data import DataLoader
from sharded_dataset import ShardedTokenDataset
from tqdm import tqdm
import torch.optim.lr_scheduler as lr_scheduler
import tiktoken
import time
import ipdb
import os

# first pass rudimentary training loop
# TODO: ADD MODEL SAVING IF IT CRASHES
# TODO: ADD TENSORBOARD SHIT

def collate_fn(batch):
    input_tokens, target_tokens = zip(*batch)

    max_len = max([x.size(0) for x in input_tokens])  # Find the longest sequence in the batch

    padded_inputs = [torch.nn.functional.pad(x, (0, max_len - x.size(0))) for x in input_tokens]
    padded_targets = [torch.nn.functional.pad(y, (0, max_len - y.size(0))) for y in target_tokens]

    return torch.stack(padded_inputs), torch.stack(padded_targets)

grad_accum_steps = 32
train_batch_size = 32
eval_batch_size = 32
block_size = 512
epochs = 10
save_every = 1
check_output_every = 1
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.set_float32_matmul_precision('high')

save_dir = "./checkpoints"
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

criterion = nn.CrossEntropyLoss()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

train_dataset = ShardedTokenDataset("./openwebtext_abridged_sharded", 500000, block_size)
train_dataloader = DataLoader(train_dataset, batch_size=train_batch_size, shuffle=False, collate_fn=collate_fn)

model = GPT(GPTConfig(seq_len=block_size))
model.to(device)

enc = tiktoken.get_encoding("gpt2")
text = "Hey, I'm a language model. Ask me a question about "
encoded_tokens = enc.encode_ordinary(text)
print("========= GENERATED OUTPUT ============")
print(enc.decode(model.generate(torch.tensor([encoded_tokens]).to(device), 60, 100).cpu().numpy()[0].tolist()))

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
        tic = time.time()
        if i % 20000 == 0:
            print("========= GENERATED OUTPUT ============")
            print(enc.decode(model.generate(torch.tensor([encoded_tokens]).to(device), 60, 100, temperature=0.8).cpu().numpy()[0].tolist()))

        inputs = batch[0].to(device)
        targets = batch[1].to(device)

        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
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
        toc = time.time()
        token_throughput = inputs.shape[0]*inputs.shape[1] / (toc - tic)
        # progress_bar.set_postfix({'dt': toc - tic})
        progress_bar.set_postfix({'token throughput': token_throughput, 'loss': running_loss / (i + 1), 'dt': toc - tic})

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












