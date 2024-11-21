import numpy as np
import os
from tqdm import tqdm
import tiktoken
from datasets import load_dataset
import ipdb

enc = tiktoken.get_encoding("gpt2")
def process_document(doc):

    """
    doc: a row in a huggingface dataset
    """
    
    document_delimiter = enc._special_tokens['<|endoftext|>']
    tokens = [document_delimiter]
    tokens.extend(enc.encode_ordinary(doc["text"]))
    # recast as np uint16 to save on disk space 
    tokens = np.array(tokens).astype(np.uint16)
    return tokens

# Function to write token arrays to disk as a shard
def write_doc_tokens(shard_path, tokens):

    """
    write tokens out to a shard
    """

    np.save(shard_path, tokens)

if __name__ == "__main__":
    local_dir = "edu_fineweb"
    DATA_CACHE_DIR = os.path.join(os.path.dirname(__file__), local_dir)
    # dataset = load_dataset("stas/openwebtext-10k", split="train")  # Split can be "train", "validation", etc.
    dataset = load_dataset("HuggingFaceFW/fineweb-edu", name="sample-10BT", split="train")
    shard_size = 100000000

    # Ensure output directory exists
    if not os.path.exists(DATA_CACHE_DIR):
        os.makedirs(DATA_CACHE_DIR)

    # initialize an empty buffer of tokens
    # use np.empty rather than np.zeros to save on memory
    all_tokens = np.zeros((shard_size,), dtype=np.uint16)
    start_delimiter = 0
    shard_idx = 0
    pbar = None

    for encoded_tokens in map(process_document, dataset):
        
        # case 1: there is enough space left in the current shard
        if start_delimiter + len(encoded_tokens) <= shard_size:
            
            if pbar is None and shard_idx > 0:
                pbar = tqdm(total=shard_size, unit="tokens", desc=f"Shard {shard_idx}")
                pbar.update(start_delimiter)

            all_tokens[start_delimiter:(start_delimiter + len(encoded_tokens))] = encoded_tokens
            start_delimiter += len(encoded_tokens)
            if pbar is None:
                pbar = tqdm(total=shard_size, unit="tokens", desc=f"Shard {shard_idx}")
            pbar.update(len(encoded_tokens))
        
        # case 2: if there isn't enough space left in the current shard, write out however much is left, 
        # and put the remainder in the next shard
        else:
            # Write the completed shard to disk (keep the first shard to being the 'validation' shard)
            shard_path = os.path.join(DATA_CACHE_DIR, f"shard_{shard_idx:06d}.npy")
            # Fill up what's left in the current shard and write it out
            num_tokens_remaining_in_current_shard = shard_size - start_delimiter
            # Finish out the progress bar
            pbar.update(num_tokens_remaining_in_current_shard)
            all_tokens[start_delimiter:] = encoded_tokens[:num_tokens_remaining_in_current_shard]            
            # print(f"Writing Shard {shard_idx}...")  # For visibility
            write_doc_tokens(shard_path, all_tokens)
            # Prepare for the next shard
            pbar = None
            shard_idx += 1
            all_tokens = np.zeros((shard_size,), dtype=np.uint16)
            all_tokens[:len(encoded_tokens[num_tokens_remaining_in_current_shard:])] = encoded_tokens[num_tokens_remaining_in_current_shard:]
            start_delimiter = len(encoded_tokens) - num_tokens_remaining_in_current_shard
        
    # Write out the last shard, which will be partially filled -- this is after the above for loop terminates
    if start_delimiter > 0:
        shard_path = os.path.join(DATA_CACHE_DIR, f"val_shard_{shard_idx}.npy")
        print(f"Writing final Shard {shard_idx}...")  # For visibility
        write_doc_tokens(shard_path, all_tokens[:start_delimiter])