import torch
import requests
import ipdb

from torch.utils.data import Dataset, DataLoader
import torch.nn as nn

def get_data():

    return requests.get("https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt").text

class GPTDataset(Dataset):

    def __init__(self, text, block_size):

        self.text = text
        self.block_size = block_size

        # character level tokenizer
        self.chars = sorted(list(set(self.text)))
        self.stoi = {char: i for i, char in enumerate(self.chars)}
        self.itos = {i: char for i, char in enumerate(self.chars)}

    def __getitem__(self, start_point):

        selected_input = self.text[start_point:(start_point + self.block_size)]
        selected_target = self.text[(start_point + 1):(start_point + 1 + self.block_size)]

        encoded_input = [self.stoi[char] for char in selected_input]
        encoded_target = [self.stoi[char] for char in selected_target]
        
        return {
            "encoded_input": torch.tensor(encoded_input, dtype=torch.long),
            "encoded_target": torch.tensor(encoded_target, dtype=torch.long)
        }
    
    def __len__(self):
        return len(self.text) - self.block_size

class Text_Processor:

    def __init__(self, batch_size, block_size):

        self.batch_size = batch_size
        self.block_size = block_size
        self.chars = sorted(list(set(self.text)))
        self.stoi = {ch: i for i, ch in enumerate(self.chars)}
        self.itos = {i: ch for i, ch in enumerate(self.chars)}
        
    def get_data(self):
        return requests.get("https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt").text

if __name__ == "__main__":

    text = get_data()
    dataset = GPTDataset(text=text, block_size=8)
    dataloader = DataLoader(dataset, batch_size=4, shuffle=False)
    vocab_size = len(dataset.chars)

    for i, batch in enumerate(dataloader):
        print(batch)
        break

    embedding_layer = nn.Embedding(vocab_size, 20)

    embeddings = embedding_layer(batch['encoded_input'])
    ipdb.set_trace()




