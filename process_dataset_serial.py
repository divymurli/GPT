import numpy as np
import os
from tqdm import tqdm
import tiktoken
from datasets import load_dataset

tokenizer = tiktoken.get_encoding("gpt2")

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

if __name__ == "__main__":
    
    local_dir = "./openwebtext_abridged_sharded"
    dataset = load_dataset("stas/openwebtext-10k", split="train")  # Split can be "train", "validation", etc.
    shard_size = 500000

    # Ensure output directory exists
    if not os.path.exists(local_dir):
        os.makedirs(local_dir)

    # extract the documents themselves
    documents = dataset['text']

    all_tokens = np.zeros((shard_size,), dtype=np.uint16)
    start_delimiter = 0
    shard_idx = 0

    # Loop over documents and process them
    for encoded_tokens in map(process_document, documents):

        # case 1: there is enough space left in the current shard
        if start_delimiter + len(encoded_tokens) <= shard_size:
            all_tokens[start_delimiter:start_delimiter + len(encoded_tokens)] = encoded_tokens
            start_delimiter += len(encoded_tokens)
        
        # case 2: if there isn't enough space left in the current shard, write out however much is left, 
        # and put the remainder in the next shard
        elif start_delimiter + len(encoded_tokens) > shard_size:
            tokens_remaining_in_current_shard = shard_size - start_delimiter
            all_tokens[start_delimiter:] = encoded_tokens[:tokens_remaining_in_current_shard]

            # Write the completed shard to disk (keep the first shard to being the 'validation' shard)
            shard_path = os.path.join(local_dir, f"shard_{shard_idx}.npy")
            # if shard_idx == 0:
            #     shard_path = os.path.join(local_dir, f"shard_val_{shard_idx}.npy")
            # else:
            #     shard_path = os.path.join(local_dir, f"shard_{shard_idx}.npy")
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

    # Write out the last shard, which will be partially filled -- this is after the above for loop terminates
    if start_delimiter > 0:
        shard_path = os.path.join(local_dir, f"shard_val_{shard_idx}.npy")
        print(f"Writing final Shard {shard_idx}...")  # For visibility
        write_doc_tokens(shard_path, all_tokens[:start_delimiter])

        # Progress bar for the last partially filled shard
        with tqdm(total=shard_size, unit="tokens", desc=f"Shard {shard_idx}") as pbar:
            pbar.update(start_delimiter)
