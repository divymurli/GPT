import math
import os
import time
from dataclasses import dataclass
import torch
import torch.nn as nn
from torch.nn import functional as F
from model import GPT, GPTConfig
import karpathy_model
from sharded_dataset_distinct_sequences import ShardedTokenDataset
from torch.utils.data import DataLoader
from tqdm import tqdm
import tiktoken

grad_accum_steps = 4
train_batch_size = 16
max_lr = 6e-4
min_lr = max_lr * 0.1
warmup_steps = 715
save_every = 10
max_steps = 83500 # 16*1024*4 = 65536, 65536*167 ~ 11M full tokens (one epoch)
epochs = 500 # therefore 167*500 = 83500 steps, assuming each step = grad_accum_steps MICRO steps

save_dir = "./checkpoints_opt_small_massive_overfit"
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

def get_lr(it):
    # 1) linear warmup for warmup_iters steps
    if it < warmup_steps:
        return max_lr * (it+1) / warmup_steps
    # 2) if it > lr_decay_iters, return min learning rate
    if it > max_steps:
        return min_lr
    # 3) in between, use cosine decay down to min learning rate
    decay_ratio = (it - warmup_steps) / (max_steps - warmup_steps)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio)) # coeff starts at 1 and goes to 0
    return min_lr + coeff * (max_lr - min_lr)

# set seed
torch.manual_seed(1337)
if torch.cuda.is_available():
    torch.cuda.manual_seed(1337)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.set_float32_matmul_precision('high')

train_dataset = ShardedTokenDataset("./openwebtext_abridged_sharded_2", 500000, 1024)
train_dataloader = DataLoader(train_dataset, batch_size=train_batch_size, shuffle=False)

criterion = nn.CrossEntropyLoss()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# My architecture causing weird repetitiveness in generative outputs
# model = GPT(GPTConfig())
# model.to(device)

# Run with his architecture, to see if I can overfit to my
# dataset
model = karpathy_model.GPT(karpathy_model.GPTConfig())
model.to(device)

if torch.cuda.is_available():
    optimizer = model.configure_optimizers(weight_decay=0.1, learning_rate=6e-4, device="cuda")
else:
    optimizer = model.configure_optimizers(weight_decay=0.1, learning_rate=6e-4, device="cpu")

overall_step = 0
for epoch in range(epochs):

    model.train()
    running_loss = 0.0
    progress_bar = tqdm(enumerate(train_dataloader), total=len(train_dataloader), desc=f"Epoch {epoch+1}")
    optimizer.zero_grad()  # Start with zeroed gradients
    losses = []
    for i, batch in progress_bar:
        
        # generate an output every 1000 major steps (NOT micro steps)
        if overall_step % 1000 == 0:
            model.eval()
            print(f"Overall step: {overall_step}")
            enc = tiktoken.get_encoding("gpt2")
            text = "A little less dark but no less harmful is a bully situation where a friend "
            encoded_tokens = enc.encode_ordinary(text)
            print("========= GENERATED OUTPUT ============")
            print(enc.decode(model.generate(torch.tensor([encoded_tokens]).to(device), 60, 100, 0.7).cpu().numpy()[0].tolist()))

        tic = time.time()
        inputs = batch[0].to(device)
        targets = batch[1].to(device)

        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            # my model
            # logits =  model(inputs)

            # his model
            logits, _ =  model(inputs)

            loss = criterion(logits.view(logits.shape[0]*logits.shape[1], logits.shape[2]), targets.view(-1))

        loss /= grad_accum_steps
        loss.backward()

        if i % grad_accum_steps == 0:
            
            # adjust LR here: one main step = grad_accum*num_gpu micro steps
            lr = get_lr(overall_step)
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr
            overall_step += 1

            optimizer.step()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1)
            optimizer.zero_grad()
        
        # # adjust LR here: adjusting it every MICRO step rather than every MAJOR step
        # lr = get_lr(overall_step)
        # for param_group in optimizer.param_groups:
        #     param_group['lr'] = lr

        running_loss += loss.item() * grad_accum_steps  # Revert loss normalization for display
        progress_bar.set_postfix({'loss': running_loss / (i + 1)})
        torch.cuda.synchronize() 
        toc = time.time()
        token_throughput = inputs.shape[0]*inputs.shape[1] / (toc - tic)
        progress_bar.set_postfix({'token throughput': token_throughput, 'loss': running_loss / (i + 1), 'dt': toc - tic, 'lr': lr})
        # overall_step += 1

    print(f"Epoch {epoch+1} finished. Mean loss: {running_loss / len(train_dataloader):.4f}")

    if (epoch + 1) % save_every == 0:
        torch.save({
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
        }, f"{save_dir}/model_epoch_{epoch + 1}.pth")





