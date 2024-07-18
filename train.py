import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tokenizer import get_data, GPTDataset
from model import GPT
import collections
from tqdm import tqdm

import ipdb

### basic hparams ###
train_batch_size = 16
eval_batch_size = 32
block_size = 32
num_epochs = 3
lr = 1e-3
criterion = nn.CrossEntropyLoss()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

text = get_data()

train_text = text[:int(0.9*len(text))]
val_text = text[int(0.9*len(text)):]

train_dataset = GPTDataset(train_text, block_size=block_size)
val_dataset = GPTDataset(val_text, block_size=block_size)

train_dataloader = DataLoader(train_dataset, batch_size=train_batch_size, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=eval_batch_size, shuffle=False)


lengths = []
for i in tqdm(range(len(train_dataset))):
    lengths.append(len(train_dataset[i]['encoded_input']))

# train_seq_lengths = collections.Counter([len(el['encoded_input']) for el in train_dataset])
# val_seq_lengths = collections.Counter([len(el['encoded_target']) for el in train_dataset])
# main training loop
ipdb.set_trace()


