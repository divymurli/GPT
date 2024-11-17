import torch
import torch.nn as nn
import inspect
import torch.nn.functional as F
from dataclasses import dataclass
import ipdb

## My implementation of GPT, following Karpathy's video
# and taking tidbits of his implementation advce

#### MANUAL, SLOW IMPLEMENTATION OF ATTENTION ####
class Attention(nn.Module):

    def __init__(self, embedding_dim, seq_len, head_dim):
        super().__init__()
        self.embeding_dim = embedding_dim
        self.head_dim = head_dim

        self.W_q = nn.Linear(embedding_dim, head_dim, bias=False)
        self.W_k = nn.Linear(embedding_dim, head_dim, bias=False)
        self.W_v = nn.Linear(embedding_dim, head_dim, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(seq_len, seq_len)))

    # x has shape [bs, seq_len, embedding_dim]
    def forward(self, x):

        # q, k, v each have shape [bs, seq_len, head_dim]
        query = self.W_q(x)
        key = self.W_k(x)
        value = self.W_v(x)
        
        # edge case in case the batch seq_len is less than the prescribed max seq_len (block_size)
        _, curr_seq_len, _ = query.shape

        # [bs, seq_len, seq_len]
        attn_matrix = torch.bmm(query, key.transpose(1, 2))
        masked_attn_matrix = attn_matrix.masked_fill(self.tril[:curr_seq_len, :curr_seq_len] == 0, -float('inf'))
        softmaxed_masked_attn_matrix = torch.softmax(masked_attn_matrix / self.head_dim ** 0.5, dim=2)
       
        return torch.bmm(softmaxed_masked_attn_matrix, value)

class MultiHeadAttention(nn.Module):

    def __init__(self, num_heads, embedding_dim, seq_len, head_dim):

        super().__init__()
        self.all_heads = nn.ModuleList([Attention(embedding_dim, seq_len, head_dim) for _ in range(num_heads)])
        self.linear_out = nn.Linear(num_heads*head_dim, embedding_dim, bias=False)

    def forward(self, x):
        all_head_outputs = []
        for head in self.all_heads:
            all_head_outputs.append(head(x))
        
        out = torch.cat(all_head_outputs, dim=-1)

        return self.linear_out(out)
#### MANUAL, SLOW IMPLEMENTATION OF ATTENTION ####

@dataclass
class GPTConfig:
    num_blocks: int = 12
    vocab_size: int = 50257 
    seq_len: int = 1024 
    num_heads: int = 12 
    head_dim: int = 64
    embedding_dim: int = 768

class CausalSelfAttention(nn.Module):
    def __init__(self, 
                embedding_dim, 
                head_dim, 
                num_heads):
        super().__init__()

        self.embedding_dim = embedding_dim
        self.head_dim = head_dim
        self.num_heads = num_heads
        self.c_attn = nn.Linear(embedding_dim, 3*num_heads*head_dim)
        self.linear_out = nn.Linear(head_dim*num_heads, embedding_dim)
        self.linear_out.NANOGPT_SCALE_INIT = 1

    # input has shape (bs, seq_len, emb_dim)
    def forward(self, x):

        # get the shapes of the input
        bs, seq_len, _ = x.size()

        # (bs, seq_len, 3*num_heads*head_dim)
        qkv = self.c_attn(x)

        # (bs, seq_len, head_dim*num_heads)
        q, k, v = qkv.chunk(3, dim=2)
        
        # reshape to (bs, seq_len, num_heads, head_dim)
        # need to do this first before transposing so as to preserve tensor structure:
        # basically we break the final dimension (num_heads*head_dim) in to two dimensions and then we transpose (1, 2) dimensions
        q = q.view(bs, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(bs, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(bs, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        # faster attentions with flash attention, pre-implemented in pytorch
        attn_output = F.scaled_dot_product_attention(q, k, v, is_causal=True)

        attn_output = attn_output.transpose(1, 2).contiguous().view(bs, seq_len, self.num_heads * self.head_dim)
        out = self.linear_out(attn_output)

        return out

class FeedForward(nn.Module):
    """ simple linear layer sandwiched by a non-linearity """

    def __init__(self, embedding_dim):
        super().__init__()

        self.first_linear = nn.Linear(embedding_dim, 4 * embedding_dim)
        # I decided to just go with GPT's original implementation of 
        # gelu, but I'd like to understand why it's better than relu
        self.gelu    = nn.GELU(approximate='tanh')
        self.final_linear = nn.Linear(4 * embedding_dim, embedding_dim)
        self.final_linear.NANOGPT_SCALE_INIT = 1

    def forward(self, x):

        x = self.first_linear(x)
        x = self.gelu(x)
        out = self.final_linear(x)

        return out

class Block(nn.Module):

    def __init__(self, 
                num_heads, 
                embedding_dim, 
                head_dim):
        super().__init__()
        self.causal_self_attention = CausalSelfAttention(embedding_dim, head_dim, num_heads)
        self.ff = FeedForward(embedding_dim)
        self.layernorm_1 = nn.LayerNorm(embedding_dim)
        self.layernorm_2 = nn.LayerNorm(embedding_dim)

    def forward(self, x):

        # gpt2 implementation
        x = x + self.causal_self_attention(self.layernorm_1(x))
        x = x + self.ff(self.layernorm_2(x))
        
        # original transformers implementation
        # attn = self.causal_self_attention(x)
        # sub_block_1 = self.layernorm_1(x + attn)
        # sub_block_2 = self.layernorm_2(sub_block_1 + self.ff(sub_block_1))

        return x

class GPT(nn.Module):

    def __init__(self,
                config,
                ):
        super().__init__()

        self.config = config
        num_blocks = config.num_blocks
        vocab_size = config.vocab_size
        seq_len = config.seq_len
        num_heads = config.num_heads
        head_dim = config.head_dim
        embedding_dim = config.embedding_dim

        self.token_embedding = nn.Embedding(vocab_size, embedding_dim)
        self.positional_embedding = nn.Embedding(seq_len, embedding_dim)
        self.blocks = nn.Sequential(*[Block(num_heads, embedding_dim, head_dim) for _ in range(num_blocks)])
        self.layer_norm = nn.LayerNorm(embedding_dim)
        # last linear layer doesn't have a bias because of the weight tying scheme
        self.last_linear_layer = nn.Linear(embedding_dim, vocab_size, bias=False)

        # weight tying scheme laid out in the original transformers paper
        self.token_embedding.weight = self.last_linear_layer.weight

        self.apply(self._init_weights)

    # weight initialization scheme as applied in original gpt2 paper
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            std = 0.02
            if hasattr(module, 'NANOGPT_SCALE_INIT'):
                std *= (2 * self.config.num_blocks) ** -0.5
            torch.nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, x):
        
        # get the shape of x
        _, curr_seq_len = x.shape
        # x has shape[bs, seq_len]
        token_embedding = self.token_embedding(x)
        positional_embedding = self.positional_embedding(torch.arange(curr_seq_len, device=x.device))
        x = token_embedding + positional_embedding # [batch_size, seq_len, embedding_dim]
        # TRANSFORMER BLOCK STUFF #
        x = self.blocks(x)
        # TRANSFORMER BLOCK STUFF #
        x = self.layer_norm(x) # [batch_size, seq_len, embedding_dim]
        out = self.last_linear_layer(x) # [batch_size, seq_len, vocab_size] 

        return out
    
    def generate(self, sequence, max_lookback_tokens, max_new_tokens=1000, temperature=0.7):
        
        for _ in range(max_new_tokens):
            # sequence has shape [bs, seq_len]
            input_sequence = sequence[:, -max_lookback_tokens:]
            # [bs, seq_len, vocab_size]
            logits = self.forward(input_sequence)
            # apply temperature scaling
            logits = logits[:, -1, :] / temperature 
            # softmax the outputs from the very last sequence
            next_token_probabilities = torch.softmax(logits, dim=-1)
            # sample one token greedily (is it better to argmax or to multinomial sample?)
            sampled_next_token = torch.multinomial(next_token_probabilities, num_samples=1)
            # sampled_next_token = torch.argmax(next_token_probabilities, dim=-1, keepdim=True)
            # append to the input sequence
            # shape [bs, seq_len + 1]
            sequence = torch.cat((sequence, sampled_next_token), dim=1)
        
        return sequence

    def configure_optimizers(self, weight_decay, learning_rate, device):
        # start with all of the candidate parameters (that require grad)
        param_dict = {pn: p for pn, p in self.named_parameters()}
        param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}
        # create optim groups. Any parameters that is 2D will be weight decayed, otherwise no.
        # i.e. all weight tensors in matmuls + embeddings decay, all biases and layernorms don't.
        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
        optim_groups = [
            {'params': decay_params, 'weight_decay': weight_decay},
            {'params': nodecay_params, 'weight_decay': 0.0}
        ]
        num_decay_params = sum(p.numel() for p in decay_params)
        num_nodecay_params = sum(p.numel() for p in nodecay_params)
        # if master_process:
        print(f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters")
        print(f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters")
        # Create AdamW optimizer and use the fused version if it is available
        fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and "cuda" in device
        # if master_process:
        print(f"using fused AdamW: {use_fused}")
        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=(0.9, 0.95), eps=1e-8, fused=use_fused)
        return optimizer
    

