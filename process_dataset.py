import os
import numpy as np
import multiprocessing as mp
from tqdm import tqdm
from datasets import load_dataset
import tiktoken  # Ensure tiktoken is installed

# Initialize tiktoken tokenizer (e.g., for GPT-2)
tokenizer = tiktoken.get_encoding("gpt2")

# Function to tokenize a single document using tiktoken
def process_document(doc):
    enc = tiktoken.get_encoding("gpt2")
    tokens = enc.encode_ordinary(doc)

    # document delimiter
    tokens.append(enc._special_tokens['<|endoftext|>'])
    # recast as np uint16 to save on space 
    tokens = np.array(tokens).astype(np.uint16)
    return tokens

# Function to write token arrays to disk as a shard
def write_doc_tokens(shard_path, tokens):
    np.save(shard_path, tokens)

# Worker function for multiprocessing
def process_shard(shard_idx, dataset_chunk, shard_size, local_dir):
    all_tokens = np.zeros((shard_size,), dtype=np.uint16)
    start_delimiter = 0

    # Loop over documents and process them
    for encoded_tokens in map(process_document, dataset_chunk):

        # case 1: there is enough space left in the current shard
        if start_delimiter + len(encoded_tokens) <= shard_size:
            all_tokens[start_delimiter:start_delimiter + len(encoded_tokens)] = encoded_tokens
            start_delimiter += len(encoded_tokens)

        # case 2: if there isn't enough space left in the current shard, write out however much is left, 
        # and put the remainder in the next shard
        else:
            tokens_remaining_in_current_shard = shard_size - start_delimiter
            all_tokens[start_delimiter:] = encoded_tokens[:tokens_remaining_in_current_shard]

            # Write the completed shard to disk (keep the first shard to being the 'validation' shard)
            if shard_idx == 0:
                shard_path = os.path.join(local_dir, f"shard_val_{shard_idx}.npy")
            else:
                shard_path = os.path.join(local_dir, f"shard_{shard_idx}.npy")
            print(f"Writing Shard {shard_idx}...")  # For visibility
            write_doc_tokens(shard_path, all_tokens)

            # Progress bar for writing out the shard
            with tqdm(total=shard_size, unit="tokens", desc=f"Shard {shard_idx}") as pbar:
                pbar.update(shard_size)

            # Prepare for the next shard
            shard_idx += 1
            all_tokens = np.zeros((shard_size,), dtype=np.uint16)
            all_tokens[:len(encoded_tokens[tokens_remaining_in_current_shard:])] = encoded_tokens[tokens_remaining_in_current_shard:]
            start_delimiter = len(encoded_tokens) - tokens_remaining_in_current_shard

    # Write the last partially filled shard (if needed)
    if start_delimiter > 0:
        shard_path = os.path.join(local_dir, f"shard_{shard_idx}.npy")
        print(f"Writing Shard {shard_idx}...")  # For visibility
        write_doc_tokens(shard_path, all_tokens[:start_delimiter])

        # Progress bar for the last partially filled shard
        with tqdm(total=shard_size, unit="tokens", desc=f"Shard {shard_idx}") as pbar:
            pbar.update(start_delimiter)

if __name__ == '__main__':
    # Settings
    local_dir = "./openwebtext_10k_sample_val_split"   # Directory to save the shards
    shard_size = 500000                    # Size of each shard in tokens
    nprocs = max(1, os.cpu_count() // 2)   # Number of processes to use

    # Load dataset from Huggingface (OpenWebText)
    dataset = load_dataset("stas/openwebtext-10k", split="train")  # Split can be "train", "validation", etc.

    # Ensure output directory exists
    if not os.path.exists(local_dir):
        os.makedirs(local_dir)

    # Extract documents from the Huggingface dataset. Assuming it's under a 'text' field.
    documents = dataset['text']

    # Split dataset into chunks for each process
    dataset_chunks = np.array_split(documents, nprocs)

    # Create multiprocessing pool and distribute work
    processes = []
    for shard_idx, dataset_chunk in enumerate(dataset_chunks):
        p = mp.Process(target=process_shard, args=(shard_idx, dataset_chunk, shard_size, local_dir))
        processes.append(p)
        p.start()

    # Wait for all processes to finish
    for p in processes:
        p.join()

    print(f"Sharding completed. Check {local_dir} for the output shards.")
