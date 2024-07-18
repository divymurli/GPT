import torch
import torch.nn as nn
import ipdb

# self attention module
class Attention(nn.Module):

    def __init__(self, embedding_dim, seq_len, head_dim):
        super().__init__()
        self.embeding_dim = embedding_dim
        self.head_dim = head_dim

        self.W_q = nn.Linear(embedding_dim, head_dim, bias=False)
        self.W_k = nn.Linear(embedding_dim, head_dim, bias=False)
        self.W_v = nn.Linear(embedding_dim,head_dim, bias=False)
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
    
    def generate(self, sequence, max_lookback_tokens, max_new_tokens=1000):
        
        for _ in range(max_new_tokens):
            # sequence has shape [bs, seq_len]
            input_sequence = sequence[:, -max_lookback_tokens:]
            # [bs, seq_len, vocab_size]
            logits = self.forward(input_sequence)
            
            # softmax the outputs from the very last sequence
            next_token_probabilities = torch.softmax(logits[:, -1, :], dim=-1)
            # sample one token greedily (is it better to argmax or to multinomial sample?)
            sampled_next_token = torch.multinomial(next_token_probabilities, num_samples=1)
            # append to the input sequence
            # shape [bs, seq_len + 1]
            sequence = torch.cat((sequence, sampled_next_token), dim=1)
        
        return sequence

if __name__ == "__main__":

    attention = Attention(64, 5, 16)
    mha = MultiHeadAttention(17, 64, 5, 16)
    block = Block(num_heads=4, embedding_dim=64, seq_len=5, head_dim=16, dropout=0.0)
    gpt = GPT(num_blocks=12,
              vocab_size=256,
              seq_len=5,
              num_heads=4,
              head_dim=16,
              dropout=0.0,
              embedding_dim=64
            )

    # x = torch.randn(1, 5, 256)
    x = torch.randint(low=0, high=255, size=(1, 2))

    out = gpt(x)

    seq = torch.zeros((1,1), dtype=torch.long)

    output = gpt.generate(seq, max_lookback_tokens=5)


    ipdb.set_trace()



