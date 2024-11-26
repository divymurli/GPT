import torch
import torch.nn.functional as F
from model import GPT, GPTConfig
import tiktoken
import ipdb

# Adapted from build-nanogpt repo: https://github.com/karpathy/build-nanogpt/blob/master/train_gpt2.py

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = GPT(GPTConfig())
state_dict = torch.load("/teamspace/studios/this_studio/GPT/checkpoints/model_epoch_1.pth")
enc = tiktoken.get_encoding("gpt2")

model.load_state_dict(state_dict["model_state_dict"])
model.to(device)

model.eval()
num_return_sequences = 4
max_length = 100
tokens = enc.encode("Harry met Sally at the bar. What happens next?")
tokens = torch.tensor(tokens, dtype=torch.long)
tokens = tokens.unsqueeze(0).repeat(num_return_sequences, 1)
xgen = tokens.to(device)
sample_rng = torch.Generator(device=device)
sample_rng.manual_seed(42)
while xgen.size(1) < max_length:
    # forward the model to get the logits
    with torch.no_grad():
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            logits = model(xgen) # (B, T, vocab_size)
        # take the logits at the last position
        logits = logits[:, -1, :] # (B, vocab_size)
        # get the probabilities
        probs = F.softmax(logits, dim=-1)
        # do top-k sampling of 50 (huggingface pipeline default)
        # topk_probs here becomes (5, 50), topk_indices is (5, 50)
        topk_probs, topk_indices = torch.topk(probs, 50, dim=-1)
        # select a token from the top-k probabilities
        # note: multinomial does not demand the input to sum to 1
        ix = torch.multinomial(topk_probs, 1, generator=sample_rng) # (B, 1)
        # gather the corresponding indices
        xcol = torch.gather(topk_indices, -1, ix) # (B, 1)
        # append to the sequence
        xgen = torch.cat((xgen, xcol), dim=1)
# print the generated text
for i in range(num_return_sequences):
    tokens = xgen[i, :max_length].tolist()
    decoded = enc.decode(tokens)
    print(f"sample {i}: {decoded}")



