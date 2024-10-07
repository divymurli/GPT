import os
import numpy as np
import torch
from torch.utils.data import Dataset

import os
import numpy as np
import torch
from torch.utils.data import Dataset

class ShardedTokenDataset(Dataset):
    def __init__(self, shard_dir, shard_size, seq_len):
        """
        Args:
            shard_dir (str): Directory containing the .npy shards of token arrays.
            shard_size (int): The number of tokens in each shard.
            seq_len (int): The sequence length to return for each data point.
        """
        self.shard_dir = shard_dir
        self.shard_size = shard_size
        self.seq_len = seq_len
        self.shard_files = sorted([os.path.join(shard_dir, f) for f in os.listdir(shard_dir) if f.endswith('.npy')])
        self.num_shards = len(self.shard_files)
        
        # Calculate total number of tokens in the dataset
        total_tokens = self.num_shards * shard_size
        
        # Calculate total number of sequences
        # We subtract `seq_len` to account for the sequence and the autoregressive target shift
        self.total_sequences = total_tokens - seq_len
        self.reset()

    def reset(self):
        self.shard_idx = 0
        self.tokens = np.load(self.shard_files[self.shard_idx])

    def __len__(self):
        return self.total_sequences

    def __getitem__(self, start_point):

        shard_idx = start_point // self.shard_size
        self.tokens = np.load(self.shard_files[shard_idx])

        # if we are in the current shard 
        selected_input = self.tokens[start_point:(start_point + self.seq_len)]
        selected_target = self.tokens[(start_point + 1):(start_point + 1 + self.seq_len)]

        if len(selected_input) < self.seq_len:
            next_shard_idx = shard_idx
            next_shard_tokens = np.load(self.shard_files[shard_idx + 1])
            num_remaining_input_tokens = self.seq_len - len(selected_input)
            num_remaining_target_tokens = self.seq_len - len(selected_target)

            selected_input = np.concatenate((selected_input, next_shard_tokens[:num_remaining_input_tokens]), axis=0)
            selected_target = np.concatenate(selected_target, next_shard_tokens[:num_remaining_target_tokens], axis=0)

        input_tokens = torch.tensor(input_tokens, dtype=torch.long)
        target_tokens = torch.tensor(target_tokens, dtype=torch.long)

        return input_tokens, target_tokens   
