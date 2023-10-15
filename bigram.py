#!/usr/bin/env python
# coding: utf-8

# importing libraries

import torch
import torch.nn as nn
from torch.nn import functional as F
# use gpu for training, if there
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(device)


# define hyperparameters

# important for training

block_size = 8
batch_size = 4
max_iters = 2500

learning_rate = 3e-3

eval_iters = 250

dropout = 0.2


# read text data

# get text

with open('wizard_of_oz.txt', 'r', encoding='utf-8') as f:
    text = f.read()
      
print(len(text)) # prints the no. of characters in the whole text  
print(text[:200]) # prints the first few characters of the text


# create vocabulary

# make vocabulary from text

chars = sorted(set(text))

print(chars) # prints all the unique characters in the text
print(len(chars)) # prints the no. of unique characters
vocab_size = len(chars)


# encode text

# initialize encoder and decoder

string_to_int = { ch:i for i,ch in enumerate(chars) }
print(string_to_int)
int_to_string = { i:ch for i,ch in enumerate(chars) }
print(int_to_string)
encode = lambda s: [string_to_int[c] for c in s]
decode = lambda l: ''.join([int_to_string[i] for i in l])

# encode, decode example

print(encode('hello')) # prints the encoded characters
encoded_hello = encode('hello')
decoded_hello = decode(encoded_hello)
print(decoded_hello) # prints the decoded characters

# encode corpus

data = torch.tensor(encode(text), dtype=torch.long)
print(data[:100]) # prints the first few encoded characters
# of the entire corpus


# train-validation split

n = int(0.8*len(data))
train_data = data[:n]
val_data = data[n:]


# create training batches/labels

# based on the hyperparameters, get random blocks of data and batch them
# into input and target tensors
def get_batch(split):
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    # print(ix)
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    # activate gpu, if available:
    x, y = x.to(device),y.to(device)
    return x, y

# execute the above function for training data to produce batches

x, y = get_batch('train')

print('inputs:')
print(x.shape) # prints the input tensor shape
print(x) # prints the input tensor contents
print('targets:')
print(y.shape) # prints thhe output (target) tensor shape
print(y) # prints the output (target) contents


# loss estimation

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

# example of input


x = train_data[:block_size]
y = train_data[1:block_size+1]
print(f"block size = {block_size}")
print(x)
print(y)
for t in range(block_size):
    context = x[:t+1]
    target = y[t]
    print('when input is', context, 'target is', target)


# bigram language model class

# create nn class as subclass of nn.Module

class BigramLanguageModel(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, vocab_size)

    def forward(self, index, targets=None):
        logits = self.token_embedding_table(index)

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


# model initialization

model = BigramLanguageModel(vocab_size)
m = model.to(device)


# sample output prior to training

context = torch.zeros((1,1), dtype=torch.long, device=device)
generated_chars = decode(m.generate(context, max_new_tokens=500)[0].tolist())
print(generated_chars)
    
    


# training loop

# create a PyTorch optimizer

optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

for iter in range(max_iters):
    if iter % eval_iters == 0:
        losses = estimate_loss()
        print(f"step: {iter}, train loss: {losses['train']:.3f} val loss: {losses['val']:.3f}")
     
        
    xb, yb  = get_batch('train')

    logits, loss = model.forward(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()
print(loss.item())


# text generation

# compare to output generated prior to training

context = torch.zeros((1,1), dtype=torch.long, device=device)
generated_chars = decode(m.generate(context, max_new_tokens=500)[0].tolist())
print(generated_chars)


# In[ ]:




