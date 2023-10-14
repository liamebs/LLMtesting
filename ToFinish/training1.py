import torch
import torch.nn as nn
from torch.nn import functional as F
import mmap
import random
import pickle
# import argparse

# parser = argparse.ArgumentParser(description='This is a demonstration program')
# parser.add_argument('-batch_size', type=str, required=True, help='Please provide a batch size')

# args = parser.parse_args()
# 'python your_script.py -batch_size 32'
# print(f'batch size: {args.batch_size}')

# use gpu for training, if there
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(device)

# hyperparameters, important for training
# batch_size = int(args.batch_size)
batch_size = 32
block_size = 128
max_iters = 200
learning_rate = 3e-4
eval_iters = 50
n_embd = 384
n_head = 1
n_layer = 1

dropout = 0.2


# get vocab.txt into program

from google.colab import drive
drive.mount('/content/drive/', force_remount=True)

file_path = '/content/drive/My Drive/Colab_Notebooks/GPTtesting/openwebtext/vocab.txt'

chars = ""
with open(file_path, 'r', encoding='utf-8') as f:
    text = f.read()
    chars = sorted(list(set(text)))
vocab_size = len(chars)


# initialize encoder and decoder

string_to_int = { ch:i for i,ch in enumerate(chars) }
int_to_string = { i:ch for i,ch in enumerate(chars) }
encode = lambda s: [string_to_int[c] for c in s]
decode = lambda l: ''.join([int_to_string[i] for i in l])

# data now created in the get_random_chunk function
# data = torch.tensor(encode(text), dtype=torch.long)


# memory map for small snippets of text from a single file of any size

train_split_path = '/content/drive/My Drive/Colab_Notebooks/GPTtesting/openwebtext/train_split.txt'
val_split_path = '/content/drive/My Drive/Colab_Notebooks/GPTtesting/openwebtext/val_split.txt'

def get_random_chunk(split):
    filename = train_split_path if split == 'train' else val_split_path
    with open(filename, 'rb') as f:
        with mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ) as mm:
          # determine the file size and a random position to start reading
          file_size = len(mm)
          start_pos = random.randint(0, (file_size) - block_size * batch_size)

          # seek to the random position and read the block of text
          mm.seek(start_pos)
          block = mm.read(block_size*batch_size-1)

          # decode the block to a string, ignoring any invalid byte sequences
          decoded_block = block.decode('utf-8', errors='ignore').replace('\r', '')

          # train and test splits
          data = torch.tensor(encode(decoded_block), dtype=torch.long)
    return data


def get_batch(split):
    data = get_random_chunk(split)
    ix = torch.randint(len(data) - block_size, (batch_size,))
    # print(ix)
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    # activate gpu, if available:
    x, y = x.to(device),y.to(device)
    return x, y


# estimate loss function with decorator to prevent grad. desc.
# from running

@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out


# create 'Head' class; each head running in parallel

class Head(nn.Module):
    """ one head of self-attention """

    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # input of size (batch, time-step, channels)
        # output of size (batch, time-step, head size)
        B,T,C = x.shape
        k = self.key(x)   # (B,T,hs)
        q = self.query(x) # (B,T,hs)

        # compute attention scores ("affinities")
        # transpose T, hs in k, matmul, then scale to reduce any dominance
        wei = q @ k.transpose(-2,-1) * k.shape[-1]**-0.5
        # ^ (B,T,hs) @ (B,hs,T) -> (B,T,T)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        # ^ (B,T,T)
        wei = F.softmax(wei, dim=-1) # (B,T,T)
        wei = self.dropout(wei)

        # perform the weighted aggregation of the values
        v = self.value(x) # (B,T,hs)
        out = wei @ v # (B,T,T) @ (B,T,hs) -> (B,T,hs)
        return out


# create 'Multi Head Attention' class

class MultiHeadAttention(nn.Module):
    """ multiple heads of self-attention in parallel """

    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])

        # add in another learnable parameter here
        self.proj = nn.Linear(head_size * num_heads, n_embd) # 'bias=False'
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1) # dim = (B, T, F)
        out = self.dropout(self.proj(out))
        return out
    
    
# create 'feed forward' class

class FeedForward(nn.Module):
    """ a simple linnear layer followed by non-linearity """

    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


# create a block class; define the decoder layers

class Block(nn.Module):
    """ Transformer block: communication followed by computation """

    def __init__(self, n_embd, n_head):
        # n_embd: embedding dimension, n_head: the number of heads we'd like
        super().__init__()
        head_size = n_embd // n_head
        self.sa = MultiHeadAttention(n_head, head_size)
        self.ffwd = FeedForward(n_embd)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        y = self.sa(x)
        x = self.ln1(x + y)
        y = self.ffwd(x)
        x = self.ln2(x + y)
        return x


# create GPT language model class as subclass of nn.Module
# renamed 'Bigram' model from previously, prior to extending functionality here
class GPTLanguageModel(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        # positional encoding:
        self.position_embedding_table = nn.Embedding(block_size, n_embd)

        # Create our 'decoder layers'; a sequential nn
        self.blocks = nn.Sequential(*[Block(n_embd, n_head=n_head) for _ in range(n_layer)])

        self.ln_f = nn.LayerNorm(n_embd) # final layer norm, prio r to nn.Linear below

        # nn.Linear transformation to softmax-compatible
        self.lm_head = nn.Linear(n_embd, vocab_size)

        # initialize weights
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)


    def forward(self, index, targets=None):
        B, T = index.shape

        # idx and targets are both (B, T) tensor of integers
        tok_emb = self.token_embedding_table(index) # (B,T,C)
        pos_emb = self.position_embedding_table(torch.arange(T, device=device)) # (T,C)
        x = tok_emb + pos_emb # (B,T,C)
        x = self.blocks(x) # (B,T,C)
        x = self.ln_f(x) # (B,T,C)
        logits = self.lm_head(x) # (B,T,vocab_size)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    def generate(self, index, max_new_tokens):
        for _ in range(max_new_tokens):
            logits, loss = self.forward(index)
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=-1)
            index_next = torch.multinomial(probs, num_samples=1)
            index = torch.cat((index, index_next), dim=1)
        return index

# model = GPTLanguageModel(vocab_size)
print("loading model parameters...")

pickle_file = '/content/drive/My Drive/Colab_Notebooks/GPTtesting/openwebtext/model-01.pkl'

with open(pickle_file, 'rb') as f:
    model = pickle.load(f)

print("loaded successfully")

m = model.to(device)


# create a PyTorch optimizer

optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

# training loop
for iter in range(max_iters):
    print(iter)
    if iter % eval_iters == 0:
        losses = estimate_loss()
        print(f"step: {iter}, train loss: {losses['train']:.3f} val loss: {losses['val']:.3f}")


    xb, yb  = get_batch('train')

    logits, loss = model.forward(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()
print(loss.item())

with open(pickle_file, 'rb') as f:
    pickle.dump(model, f)
print('model saved')

