import ipdb
import regex as re
import os
import json
import unicodedata

GPT4_SPLIT_PATTERN = r"""'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]++[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+"""


# based on karpathy's minbpe repo (https://github.com/karpathy/minbpe/blob/master/exercise.md), with a slight mod in the encode() function for ease of understanding.

def get_stats(ids):
    counts = {}
    for pair in zip(ids, ids[1:]): # Pythonic way to iterate consecutive elements
        counts[pair] = counts.get(pair, 0) + 1
    return counts

def get_stats_2(ids, counts):
    for pair in zip(ids, ids[1:]): 
        counts[pair] = counts.get(pair, 0) + 1
    return counts

# taken from karpathy's repo
def replace_control_characters(s: str) -> str:
    # we don't want to print control characters
    # which distort the output (e.g. \n or much worse)
    # https://stackoverflow.com/questions/4324790/removing-control-characters-from-a-string-in-python/19016117#19016117
    # http://www.unicode.org/reports/tr44/#GC_Values_Table
    chars = []
    for ch in s:
        if unicodedata.category(ch)[0] != "C":
            chars.append(ch) # this character is ok
        else:
            chars.append(f"\\u{ord(ch):04x}") # escape
    return "".join(chars)

def render_token(t: bytes) -> str:
    # pretty print a token, escaping control characters
    s = t.decode('utf-8', errors='replace')
    s = replace_control_characters(s)
    return s

def merge(ids, pair, idx):
  # in the list of ints (ids), replace all consecutive occurences of pair with the new token idx
  newids = []
  i = 0
  while i < len(ids):
    # if we are not at the very last position AND the pair matches, replace it
    if i < len(ids) - 1 and ids[i] == pair[0] and ids[i+1] == pair[1]:
      newids.append(idx)
      i += 2
    else:
      newids.append(ids[i])
      i += 1
  return newids

class BasicTokenizer:

    def __init__(self, load = False):
        self.merges = {}
        self.vocab = self._build_vocab()
        self.load = load
        if self.load:

            with open("./merges_basic.json", "r") as infile:
                inverted_merges = json.load(infile)
                self.merges = {tuple(val): int(key) for key, val in inverted_merges.items()}

    def train(self, text, vocab_size, vocab_filename="./basic.vocab", merges_dict_name="./merges_basic.json"):
        num_merges = vocab_size - len(self.vocab)

        # get utf encoding of the text
        text = text.encode("utf-8")
        ids = list(map(int, text)) # 0 to 255, byte stream

        for i in range(num_merges):
           stats = get_stats(ids)

           # find the maximally occurring pair
           pair = max(stats, key=stats.get) 

           idx = 256 + i 
           ids = merge(ids, pair, idx)

           # add merge to the merges dict
           self.merges[pair] = idx
           self.vocab[idx] = self.vocab[pair[0]] + self.vocab[pair[1]]

        inverted_merges = {idx: pair for pair, idx in self.merges.items()}        
        with open(merges_dict_name, "w") as outfile_2:
            json.dump(inverted_merges, outfile_2)

        with open(vocab_filename, "w", encoding="utf-8") as f:
            for idx, token in self.vocab.items():
                # note: many tokens may be partial utf-8 sequences
                # and cannot be decoded into valid strings. Here we're using
                # errors='replace' to replace them with the replacement char �.
                # this also means that we couldn't possibly use .vocab in load()
                # because decoding in this way is a lossy operation!
                s = render_token(token)
                # find the children of this token, if any
                if idx in inverted_merges:
                    # if this token has children, render it nicely as a merge
                    idx0, idx1 = inverted_merges[idx]
                    s0 = render_token(self.vocab[idx0])
                    s1 = render_token(self.vocab[idx1])
                    f.write(f"[{s0}][{s1}] -> [{s}] {idx}\n")
                else:
                    # otherwise this is leaf token, just print it
                    # (this should just be the first 256 tokens, the bytes)
                    f.write(f"[{s}] {idx}\n")

    def encode(self, text):
        
        # get utf token stream
        tokens = list(text.encode("utf-8"))

        # initial set of statistics for the tokens, mapping ({id_0, id_1}: frequency)
        # merges dictionary only maps pairs
        # slightly less efficient implementation compared to Karpathy's so that I could break it down
        
        while len(tokens) >= 2:
            stats = get_stats(tokens)
            filter_dict = {}
            for key, _ in stats.items():
               if key in self.merges:
                  filter_dict[key] = self.merges[key]
            
            try:
                pair_to_merge = min(filter_dict, key=filter_dict.get)
                tokens = merge(tokens, pair_to_merge, filter_dict[pair_to_merge])
            except ValueError:
               break

        return tokens
           
    def decode(self, ids):

        return b"".join([self.vocab[id] for id in ids]).decode("utf-8", errors="replace")
    
    def _build_vocab(self):
        
        vocab = {idx: bytes([idx]) for idx in range(256)}
        for (p0, p1), idx in self.merges.items():
            vocab[idx] = vocab[p0] + vocab[p1]

        return vocab
    
class RegexTokenizer:
   
    def __init__(self, load = False):
        self.merges = {}
        self.idx_to_merges = {}
        self.vocab = self._build_vocab()
        self.pattern = re.compile(GPT4_SPLIT_PATTERN)
        self.load = load
        if self.load: 

            with open("./merges_regex.json", "r") as infile:
                inverted_merges = json.load(infile)
                self.merges = {tuple(val): int(key) for key, val in inverted_merges.items()}

    def train(self, text, vocab_size, vocab_filename="./regex.vocab", merges_dict_name="./merges_regex.json"):
        num_merges = vocab_size - len(self.vocab)

        # the purpose of this is to prevent tokenizing across regex boundaries, like e.g.
        # apostrophes, periods, /'s and so on.
        text_chunks = re.findall(self.pattern, text)
        encoded_text_chunks = [list(map(int, text_chunk.encode("utf-8"))) for text_chunk in text_chunks]

        for i in range(num_merges):
            stats = {}
            # collate the stats across all the text chunks
            for encoded_text_chunk in encoded_text_chunks:
                stats = get_stats_2(encoded_text_chunk, stats)
        
            # find the maximally occurring pair
            pair = max(stats, key=stats.get) 

            idx = 256 + i

            # replace all instances of the maximally occurring pair 
            encoded_text_chunks = [merge(ids, pair, idx) for ids in encoded_text_chunks]

            # add merge to the merges dict
            self.merges[pair] = idx
            self.vocab[idx] = self.vocab[pair[0]] + self.vocab[pair[1]]

        # with open(vocab_dict_name, "w") as outfile:
        #     json.dump(self.vocab, outfile)

        inverted_merges = {idx: pair for pair, idx in self.merges.items()}
        with open(merges_dict_name, "w") as outfile_2:
            json.dump(inverted_merges, outfile_2)

        # save file taken from karpathy's nanogpt repo: 
        with open(vocab_filename, "w", encoding="utf-8") as f:
            for idx, token in self.vocab.items():
                # note: many tokens may be partial utf-8 sequences
                # and cannot be decoded into valid strings. Here we're using
                # errors='replace' to replace them with the replacement char �.
                # this also means that we couldn't possibly use .vocab in load()
                # because decoding in this way is a lossy operation!
                s = render_token(token)
                # find the children of this token, if any
                if idx in inverted_merges:
                    # if this token has children, render it nicely as a merge
                    idx0, idx1 = inverted_merges[idx]
                    s0 = render_token(self.vocab[idx0])
                    s1 = render_token(self.vocab[idx1])
                    f.write(f"[{s0}][{s1}] -> [{s}] {idx}\n")
                else:
                    # otherwise this is leaf token, just print it
                    # (this should just be the first 256 tokens, the bytes)
                    f.write(f"[{s}] {idx}\n")

    def decode(self, ids):
        return b"".join([self.vocab[id] for id in ids]).decode("utf-8", errors="replace")
    
    def encode(self, text):
        text_chunks = re.findall(self.pattern, text)
        encoded_text = []
        for chunk in text_chunks:
            encoded_chunk = self.encode_chunk(chunk)
            encoded_text.extend(encoded_chunk)

        return encoded_text
    
    def encode_chunk(self, text):
        
        # get utf token stream
        tokens = list(text.encode("utf-8"))

        # initial set of statistics for the tokens, mapping ({id_0, id_1}: frequency)
        # merges dictionary only maps pairs
        # slightly less efficient implementation compared to Karpathy's so that I could break it down
        
        while len(tokens) >= 2:
            stats = get_stats(tokens)
            filter_dict = {}
            for key, _ in stats.items():
               if key in self.merges:
                  filter_dict[key] = self.merges[key]
            
            try:
                pair_to_merge = min(filter_dict, key=filter_dict.get)
                tokens = merge(tokens, pair_to_merge, filter_dict[pair_to_merge])
            except ValueError:
               break

        return tokens

    def _build_vocab(self):
        
        vocab = {idx: bytes([idx]) for idx in range(256)}
        for (p0, p1), idx in self.merges.items():
            vocab[idx] = vocab[p0] + vocab[p1]

        return vocab


if __name__ == "__main__":
   
    
    file_path = os.path.join(os.getcwd(), 'taylorswift.txt')
    
    with open(file_path, 'r') as file:
        lines = file.readlines()
        text = ''.join(lines)
    
    # first, train the tokenizers
    # tokenizer = BasicTokenizer()
    # regex_tokenizer = RegexTokenizer()
    # tokenizer.train(text=text, vocab_size=2000)
    # regex_tokenizer.train(text=text, vocab_size=2000)
    # basic_encoded_text = tokenizer.encode("hello world, my name is taylor swift")
    # regex_encoded_text = regex_tokenizer.encode("hello world, my name is taylor swift")
    # decoded_text = regex_tokenizer.decode(regex_encoded_text)

    # # next, load the merges
    tokenizer = BasicTokenizer(load=True)
    regex_tokenizer = RegexTokenizer(load=True)

    basic_encoded_text = tokenizer.encode("hello world, my name is taylor swift")
    regex_encoded_text = regex_tokenizer.encode("hello world, my name is taylor swift")

    ipdb.set_trace()



