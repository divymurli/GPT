import os
import numpy as np
import torch
from torch.utils.data import Dataset

class ShardedTokenDataset(Dataset):
    def __init__(self, shard_dir, shard_size, seq_len, val=False):
        """
        Args:
            shard_dir (str): Directory containing the .npy shards of token arrays.
            shard_size (int): The number of tokens in each shard.
            seq_len (int): The sequence length to return for each data point.
        """
        self.shard_dir = shard_dir
        self.shard_size = shard_size
        self.seq_len = seq_len
        self.cached_shard_idx = -1
        self.shard_files = [os.path.join(shard_dir, f) for f in os.listdir(shard_dir) if f.startswith("shard_") and f.endswith('.npy')]
        if val:
            self.shard_files = os.path.join(shard_dir, "val_shard_22.npy")
        self.num_shards = len(self.shard_files)

        # Calculate total number of tokens in the dataset
        total_tokens = self.num_shards * shard_size
        if val:
            total_tokens = len(np.load(os.path.join(self.shard_dir, "val_shard_22.npy")))
        
        # Calculate total number of sequences
        # We subtract `seq_len` to account for the sequence and the autoregressive target shift
        # Add 1 because we need to take a total of [bs, seq_len + 1] in each pull 
        self.total_sequences = total_tokens // (seq_len + 1)

    def _load_shard(self, shard_idx):
        if self.cached_shard_idx != shard_idx:
            self.cached_shard = np.load(os.path.join(self.shard_dir, f"shard_{shard_idx}.npy"))
            self.cached_shard_idx = shard_idx
        return self.cached_shard

    def __len__(self):
        return self.total_sequences

    def __getitem__(self, start_point):
        
        if start_point < 0 or start_point >= self.total_sequences:
            raise IndexError(f"Index {start_point} out of bounds for dataset of size {self.total_sequences}")
        
        # grab the block of text in question
        index_start = start_point * (self.seq_len)
        shard_idx = index_start // self.shard_size
        within_shard_idx = index_start % self.shard_size

        # load the current tokens, from the cached shard
        self.tokens = self._load_shard(shard_idx)

        # if we are in the current shard 
        selected_input = self.tokens[within_shard_idx:(within_shard_idx + self.seq_len)]
        selected_target = self.tokens[(within_shard_idx + 1):(within_shard_idx + 1 + self.seq_len)]

        ### CROSS SHARD BOUNDARIES
        # test this edge case more carefully
        if len(selected_input) < self.seq_len:
            # quick fix for now
            if shard_idx + 1 < self.num_shards:
                next_shard_tokens = self._load_shard(shard_idx + 1)
                num_remaining_input_tokens = self.seq_len - len(selected_input)
                num_remaining_target_tokens = self.seq_len - len(selected_target)

                selected_input = np.concatenate((selected_input, next_shard_tokens[:num_remaining_input_tokens]), axis=0)
                selected_target = np.concatenate((selected_target, next_shard_tokens[:num_remaining_target_tokens]), axis=0)
            
            else:
                # If no next shard, pad the sequence to the required length
                pad_len = self.seq_len - len(selected_input)
                selected_input = np.pad(selected_input, (0, pad_len), constant_values=0)  # Padding value can be 0 or another token
                selected_target = np.pad(selected_target, (0, pad_len), constant_values=0)

        if len(selected_target) < self.seq_len:

            if shard_idx + 1 < self.num_shards:
                next_shard_tokens = self._load_shard(shard_idx + 1)
                num_remaining_input_tokens = self.seq_len - len(selected_input)
                num_remaining_target_tokens = self.seq_len - len(selected_target)
                selected_target = np.concatenate((selected_target, next_shard_tokens[:num_remaining_target_tokens]), axis=0)
            
            else:
                # If no next shard, pad the sequence to the required length
                pad_len = self.seq_len - len(selected_input)
                selected_target = np.pad(selected_target, (0, pad_len), constant_values=0)
        ### CROSS SHARD BOUNDARIES

        input_tokens = torch.tensor(selected_input, dtype=torch.long)
        target_tokens = torch.tensor(selected_target, dtype=torch.long)

        return input_tokens, target_tokens
