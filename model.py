import torch
import torch.nn as nn
import inspect
import ipdb

# self attention module
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

class FeedFoward(nn.Module):
    """ a simple linear layer followed by a non-linearity """

    def __init__(self, embedding_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(embedding_dim, 4 * embedding_dim),
            nn.ReLU(),
            nn.Linear(4 * embedding_dim, embedding_dim),
        )

    def forward(self, x):
        return self.net(x)

class Block(nn.Module):

    def __init__(self, num_heads, embedding_dim, seq_len, head_dim, dropout):
        super().__init__()
        self.mha = MultiHeadAttention(num_heads, embedding_dim, seq_len, head_dim)
        self.ff = FeedFoward(embedding_dim)
        self.layernorm_1 = nn.LayerNorm(embedding_dim)
        self.layernorm_2 = nn.LayerNorm(embedding_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):

        attn = self.mha(x)
        sub_block_1 = self.dropout(self.layernorm_1(x + attn))
        sub_block_2 = self.dropout(self.layernorm_2(sub_block_1 + self.ff(sub_block_1)))

        return sub_block_2

class GPT(nn.Module):

    def __init__(self, num_blocks, vocab_size, seq_len, num_heads, head_dim, dropout, embedding_dim=64):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, embedding_dim)
        self.positional_embedding = nn.Embedding(seq_len, embedding_dim)
        self.blocks = nn.Sequential(*[Block(num_heads, embedding_dim, seq_len, head_dim, dropout) for _ in range(num_blocks)])
        self.layer_norm = nn.LayerNorm(embedding_dim)
        self.linear_layer = nn.Linear(embedding_dim, vocab_size)

    def forward(self, x):
        
        # get the shape of x
        _, curr_seq_len = x.shape
        # x has shape[bs, seq_len]
        token_embedding = self.token_embedding(x)
        positional_embedding = self.positional_embedding(torch.arange(curr_seq_len, device=torch.device("cuda" if torch.cuda.is_available() else "cpu")))
        x = token_embedding + positional_embedding # [batch_size, seq_len, embedding_dim]
        # TRANSFORMER BLOCK STUFF #
        x = self.blocks(x)
        # TRANSFORMER BLOCK STUFF #
        x = self.layer_norm(x) # [batch_size, seq_len, embedding_dim]
        out = self.linear_layer(x) # [batch_size, seq_len, vocab_size] 

        return out
    
    def generate(self, sequence, max_lookback_tokens, max_new_tokens=1000, temperature=0.7):
        
        # self.eval()
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
        use_fused = fused_available and device == "cuda"
        # if master_process:
        print(f"using fused AdamW: {use_fused}")
        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=(0.9, 0.95), eps=1e-8, fused=use_fused)
        return optimizer

# if __name__ == "__main__":

#     attention = Attention(64, 5, 16)
    # mha = MultiHeadAttention(17, 64, 5, 16)
    # import ipdb
    # ipdb.set_trace()
#     block = Block(num_heads=4, embedding_dim=64, seq_len=5, head_dim=16, dropout=0.0)
    # gpt = GPT(num_blocks=12,
    #           vocab_size=50257,
    #           seq_len=1024,
    #           num_heads=12,
    #           head_dim=12,
    #           dropout=0.0,
    #           embedding_dim=768
    #         )
    # total_params = sum(p.numel() for p in gpt.parameters())
    # print(total_params)


#     # x = torch.randn(1, 5, 256)
#     x = torch.randint(low=0, high=255, size=(1, 2))

#     out = gpt(x)

#     seq = torch.zeros((1,1), dtype=torch.long)

#     output = gpt.generate(seq, max_lookback_tokens=5)


#     ipdb.set_trace()



