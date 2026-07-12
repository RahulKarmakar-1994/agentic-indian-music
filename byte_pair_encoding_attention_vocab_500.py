#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 26 14:06:20 2024

@author: tarak
"""

import pandas as pd
import numpy as np
import sklearn as sk
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import decomposition

import torch
import time
# In[2]:


from rdkit import Chem
from rdkit.Chem import AllChem
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

#%%


SMILES_CHARS=['.','c', 'n', 'o', 'C', 'N', 'F', 'K','=','O', 'H', '%','0',']', '[', 'Na','Li', 'Ca','Cd','Te','(', ')', '1','\ ','\\','2','#','Cl','/','B','s','S','Se','Ge','Br','Sn','Zn','Si','se','I', 'Pb','3', '4', '5', '6', '7', '8', '+','-', '9', 'P','*']

len(SMILES_CHARS)
smi2index = dict((c, i) for i, c in enumerate(SMILES_CHARS))
index2smi = dict((i, c) for i, c in enumerate(SMILES_CHARS))
# In[3]:


data=pd.read_csv("/Users/rahulkarmakar/Documents/PostDoc/IIT_Madras/Machine_Learning/glass_transition_data/attachments/tg_raw.csv")
data

text = data['SMILES']
#%%
print(text[:1000])

#%%
chars = list(set(text))

print(''.join(chars))

#%%
stoi = {s:i for i,s in enumerate(SMILES_CHARS)}

#stoi['.'] = 0

itos = {i:s for s,i in stoi.items()}
vocab_size = len(itos)
print(vocab_size)   

#%%
X_tot = np.zeros((7174,314,52))
extracted_characters = [[] for _ in range(7174)]
extracted_token = [[] for _ in range(7174)]
j = 0
for sm in range(7174):
    # Initialize an empty list to hold the extracted characters
    #print('hi')
    
    
    # Use a variable to track the index while looping through the string
    i = 0
    
    smiles_string = list(text[sm])
    str_length = len(text[sm])
    # extracted_characters[j].append('.')
    # extracted_token[j].append(0)
    while i < str_length:
        # Check for multi-character elements
        check = "".join(list(smiles_string[i:i+2]))
        if i < str_length-1 and  check in SMILES_CHARS:
            ix = stoi[check]
            extracted_characters[j].append(check)
            extracted_token[j].append(ix)
            #if check in smi2index:
                #X_tot[sm,j,smi2index[check]] = 1
            i += 2  # Skip the next character since we've just added two    
        else:
            if smiles_string[i] =='%':
                print('yes',sm)
            ix =   stoi[smiles_string[i]]  
            extracted_characters[j].append(smiles_string[i])  # Add the single character
            extracted_token[j].append(ix)
             # Move to the next character
            #print(sm,i,smi2index[smiles_string[i]],smiles_string[i])
            #X_tot[sm,j,smi2index[smiles_string[i]]] = 1
            i += 1 
      
    # extracted_characters[j].append('.')   
    # extracted_token[j].append(0)
    j = j +1 
    
    
    
#%%
stoi = {s:i for i,s in enumerate(SMILES_CHARS)}

#stoi['.'] = 0

itos = {i:s for s,i in stoi.items()}
vocab_size = len(itos)
print(vocab_size)   


#%%

#%%  Generate most occuring pair and merge

def get_stats(ids):
    counts = {}
    for w in ids:
        # for ch1,ch2 in zip(w,w[1:]):
            for bgpair in zip(w, w[1:]):
                counts[bgpair] = counts.get(bgpair, 0) + 1
    return counts

def merge_cons(ids, pair, idx):
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


# ---
vocab_size = 200 # the desired final vocabulary size
num_merges = vocab_size - len(SMILES_CHARS)
ids = list(extracted_token) # copy so we don't destroy the original list
new_smiles = list(SMILES_CHARS)
#%%
merges = {} # (int, int) -> int
for i in range(num_merges):
  stats = get_stats(ids)
  pair = max(stats, key=stats.get)
  idx = len(SMILES_CHARS) + i
  print(f"merging {pair} into a new token {idx}")
  for j in range(len(ids)):
      ids[j] = merge_cons(ids[j], pair, idx)
  merges[pair] = idx
  
#%%

for key,value in merges.items():
    out = []
    print(key[0],key[1],value)
    ix = itos[key[0]]
    iy = itos[key[1]]
    out.append(ix)
    out.append(iy)
    print(ix,iy , ''.join(out))
    new_smiles.append(''.join(out))
    itos[value] = ''.join(out)
    ns = ''.join(out)
    stoi[ns] = value
    
#%%    
length = []  
for i in range(len(ids)):
    length.append(len(ids[i]))  
#%%

ids_mod = [[] for _ in range(7174)]
for sm in range(7174):
    j = 0
    str_length = len(ids[sm])
    ids_mod[sm].append(0)
    while j < str_length:
        ix = ids[sm][j]
        ids_mod[sm].append(ix)
        j = j +1 
    # extracted_characters[j].append('.')   
    ids_mod[sm].append(0)
    
 #%%   
encode = lambda s: [stoi[c] for c in s] # encoder: take a string, output a list of integers
decode = lambda l: ''.join([itos[i] for i in l]) # decoder: take a list of integers, output a string
#%%
  
#%%
# build the dataset
block_size = 8 # context length: how many characters do we take to predict the next one?

def build_dataset(words):  
  X, Y , Z = [], [] , []
  #print(words)
  for w in words:
    context = [0] * block_size
    for ch in w :
      print(context,ch)
      ix = ch
      X.append(context)
      Y.append(ix)
      context = context[1:] + [ix] # crop and append
      Z.append(context)
  X = torch.tensor(X)
  Y = torch.tensor(Y)
  Z = torch.tensor(Z)
  print(X.shape, Y.shape)
  return X, Y, Z

import random
# random.seed(42)
# random.shuffle(ids_mod)



# Generate the same random order
random.seed(42)
indices = list(range(len(ids_mod)))
random.shuffle(indices)

# Shuffle both lists using the same indices
ids_mod = [ids_mod[i] for i in indices]
text_shuffle = [text[i] for i in indices]

n1 = int(0.8*len(ids_mod))
n2 = int(0.9*len(ids_mod))
Xtr, Ytr , Ztr = build_dataset(ids_mod[:n1])
Xdev, Ydev , Zdev = build_dataset(ids_mod[n1:n2])
Xte, Yte, Zte = build_dataset(ids_mod[n2:])       


#%%


import torch.nn as nn
from torch.nn import functional as F



# hyperparameters
batch_size = 32 # how many independent sequences will we process in parallel?
block_size = 8 # what is the maximum context length for predictions?

eval_interval = 10
learning_rate = 1e-3
device = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_iters = 200
n_embd = 64
n_head = 4
n_layer = 4
dropout = 0.1

# ------------

torch.manual_seed(1337)


# create a mapping from characters to integers
# stoi = { ch:i for i,ch in enumerate(chars) }
# itos = { i:ch for i,ch in enumerate(chars) }
encode = lambda s: [stoi[c] for c in s] # encoder: take a string, output a list of integers
decode = lambda l: ''.join([itos[i] for i in l]) # decoder: take a list of integers, output a string



# data loading
# def get_batch(split):
#     # generate a small batch of data of inputs x and targets y
#     data = train_data if split == 'train' else val_data
#     ix = torch.randint(len(data) - block_size, (batch_size,))
#     x = torch.stack([data[i:i+block_size] for i in ix])
#     y = torch.stack([data[i+1:i+block_size+1] for i in ix])
#     x, y = x.to(device), y.to(device)
#     return x, y
#%%
@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            itest = torch.randint(0, Xte.shape[0], (batch_size,))
            X, Y = Xte[itest], Zte[itest] # batch X,Y
            #X, Y = Xte,Zte
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out
#%%
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
        B,T,C = x.shape
        #print(B,T,C)
        k = self.key(x)   # (B,T,C)
        q = self.query(x) # (B,T,C)
        #print(k.shape)
        # compute attention scores ("affinities")
        wei = q @ k.transpose(-2,-1) * k.shape[-1]**-0.5 # (B, T, C) @ (B, C, T) -> (B, T, T)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf')) # (B, T, T)
        wei = F.softmax(wei, dim=-1) # (B, T, T)
        wei = self.dropout(wei)
        # perform the weighted aggregation of the values
        v = self.value(x) # (B,T,C)
        out = wei @ v # (B, T, T) @ (B, T, C) -> (B, T, C)
        return out

class MultiHeadAttention(nn.Module):
    """ multiple heads of self-attention in parallel """

    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(n_embd, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.dropout(self.proj(out))
        return out

class FeedFoward(nn.Module):
    """ a simple linear layer followed by a non-linearity """

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

class Block(nn.Module):
    """ Transformer block: communication followed by computation """

    def __init__(self, n_embd, n_head):
        # n_embd: embedding dimension, n_head: the number of heads we'd like
        super().__init__()
        head_size = n_embd // n_head
        self.sa = MultiHeadAttention(n_head, head_size)
        self.ffwd = FeedFoward(n_embd)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x

# super simple bigram model
class BigramLanguageModel(nn.Module):

    def __init__(self):
        super().__init__()
        # each token directly reads off the logits for the next token from a lookup table
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        self.blocks = nn.Sequential(*[Block(n_embd, n_head=n_head) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd) # final layer norm
        self.lm_head = nn.Linear(n_embd, vocab_size)

    def forward(self, idx, targets=None):
        #print(idx)
        B, T = idx.shape

        # idx and targets are both (B,T) tensor of integers
        tok_emb = self.token_embedding_table(idx) # (B,T,C)
        pos_emb = self.position_embedding_table(torch.arange(T, device=device)) # (T,C)
        x = tok_emb + pos_emb # (B,T,C)
        x = self.blocks(x) # (B,T,C)
        x = self.ln_f(x) # (B,T,C)
        logits = self.lm_head(x) # (B,T,vocab_size)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            #print(B,T,C)
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    def generate(self, idx, max_new_tokens,Temperature):
        # idx is (B, T) array of indices in the current context
        string = []
        idx_prev = idx[:,-1:]
        for g in range(max_new_tokens):
            # crop idx to the last block_size tokens
            idx_cond = idx[:, -block_size:]
            # get the predictions
            logits, loss = self(idx_cond)
            # focus only on the last time step
            logits = logits[:, -1, :]/Temperature # becomes (B, C)
            # apply softmax to get probabilities
            probs = F.softmax(logits, dim=-1) # (B, C)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1) # (B, 1)
            # append sampled index to the running sequence
            
            if idx_prev !=0 and idx_next == 0:
                break
            idx_prev = idx_next
            idx = torch.cat((idx, idx_next), dim=1) # (B, T+1)
            if idx_next!=0:
                string.append(idx_next)
        return idx
#%%

Xtr,  Ztr = Xtr.to(device), Ztr.to(device)
Xdev,  Zdev = Xdev.to(device), Zdev.to(device)
Xte, Zte =  Xte.to(device), Zte.to(device)   
#%%
from torch.utils.data import TensorDataset, DataLoader
# Create TensorDataset and DataLoader
dataset = TensorDataset(Xtr, Ztr)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
#%%
model = BigramLanguageModel()
m = model.to(device)
# print the number of parameters in the model
print(sum(p.numel() for p in m.parameters())/1e6, 'M parameters')
#%%%
# create a PyTorch optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
#%%
eval_interval = 10
learning_rate = 0.0005
max_iters = 70
start = 0

start_time = time.time()
for iter in range(start,max_iters):

    for xb, yb in dataloader:  # This fetches a batch of data
        # xb is input batch, yb is target batch
        logits, loss = model(xb, yb)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    #every once in a while evaluate the loss on train and val sets
    # if iter % eval_interval == 0 or iter == max_iters - 1:
    #     losses = estimate_loss()
    #     print(f"step {iter}: train loss {loss.item():.4f}, val loss {losses['val']:.4f}")
    
#%%

end_time = time.time()
print(f"Execution time: {end_time - start_time} seconds")


losses = estimate_loss()
print(f"val loss {losses['val']:.4f}")
#%%
# torch.save(model, "/Users/rahulkarmakar/Documents/PostDoc/IIT_Madras/Machine_Learning/glass_transition_data/attachments/New_run_after_Random_shuffling/Using_dataloader/Reduce_train_size/byte_pair_attention_model_vocab_500.pth")
#%%
# Load the saved model
# model_path = "byte_pair_attention_model_vocab_200.pth"
# m = torch.load(model_path)

# # Put the model in evaluation mode
# model.eval()
#%%
# =========================
# SAVE MODEL CHECKPOINT
# =========================

save_path = (
    "/Users/rahulkarmakar/Documents/PostDoc/IIT_Madras/"
    "Machine_Learning/glass_transition_data/attachments/"
    "New_run_after_Random_shuffling/Using_dataloader/"
    "Reduce_train_size/bpe_transformer_vocab200.ckpt"
)

checkpoint = {
    # model
    "model_state_dict": model.state_dict(),

    # vocabulary
    "stoi": stoi,
    "itos": itos,
    "vocab_size": vocab_size,

    # BPE info (important for reproducibility)
    "merges": merges,
    "base_vocab": SMILES_CHARS,

    # model hyperparameters
    "block_size": block_size,
    "n_embd": n_embd,
    "n_head": n_head,
    "n_layer": n_layer,
    "dropout": dropout,
}

torch.save(checkpoint, save_path)
print(f"✅ Model checkpoint saved at:\n{save_path}")

#%%
valid_smile = []
#%%
Temperature = 1.0
generation_cycle = 0
for gc in range(generation_cycle):
    # # generate from the model
    context = torch.zeros((1, 1), dtype=torch.long, device=device)
    a = m.generate(context, max_new_tokens=max(length), Temperature=Temperature)[0].tolist()
    #print(decode(a))
    b = []
    for sm in range(len(a)):
        if a[sm] !=0:
            b.append(a[sm])
    assert(b[-1] != 0)        
    sms = decode(b)     
    print(gc)
    if sms:
        molecule = Chem.MolFromSmiles(sms)
        if molecule:
            valid_smile.append(sms)  
#print(decode(m.generate(context, max_new_tokens=50)[0].tolist()))    

#%%

sample = valid_smile[200]#'*c1ccc2c(c1)C(=O)N(c1ccc(C(C)(C)c3ccc(N4C(=O)c5ccc(C(*)(C(F)(F)F)C(F)(F)F)cc5C4=O)cc3)cc1)C2=O'# valid_smile[84]

from rdkit import Chem

from rdkit.Chem.Draw import IPythonConsole
molecule = Chem.MolFromSmiles(sample)
molecule  
#%%
# with open("/Users/rahulkarmakar/Documents/PostDoc/IIT_Madras/Machine_Learning/glass_transition_data/attachments/New_run_after_Random_shuffling/Using_dataloader/generate_smile_vocab_500_attention_T_0.5.txt", "w") as file:
#     for item in valid_smile:
#         file.write(f"{item}\n")

#%% check batch
#generation_cycle = 10000
print('Valid Smiles %',len(valid_smile)/generation_cycle)

#%%

def canonicalize_smiles(smiles_list, batch_size=100):
    """Canonicalize SMILES in batches for large datasets."""
    canonical_smiles = set()
    for i in range(0, len(smiles_list), batch_size):
        batch = smiles_list[i:i + batch_size]
        for smi in batch:
            try:
                mol = Chem.MolFromSmiles(smi)
                if mol:
                    canonical_smiles.add(Chem.MolToSmiles(mol))
            except:
                print(f"Invalid SMILES: {smi}")
    return canonical_smiles

def compare_datasets(training_smiles, generated_smiles):
    """Compare training and generated SMILES datasets."""
    common_smiles = training_smiles.intersection(generated_smiles)
    unique_generated_smiles = generated_smiles - training_smiles
    return common_smiles, unique_generated_smiles


train_set = text_shuffle[:n1]

# Canonicalize SMILES
training_smiles = canonicalize_smiles(train_set)
generated_smiles = canonicalize_smiles(valid_smile)

# Compare
common_smiles, unique_generated_smiles = compare_datasets(training_smiles, generated_smiles)

# Results
print(f"Number of overlapping SMILES: {len(common_smiles)}")
print(f"Number of unique SMILES: {len(unique_generated_smiles)}")
print(f"Total SMILES: {len(unique_generated_smiles)+len(common_smiles)}")
print(f"Unique SMILES %: {len(unique_generated_smiles)/(len(unique_generated_smiles)+len(common_smiles))}")
#%%%


# Save to file
# with open("/Users/rahulkarmakar/Documents/PostDoc/IIT_Madras/Machine_Learning/glass_transition_data/attachments/New_run_after_Random_shuffling/Using_dataloader/train_smiles.txt", "w") as file:
#     for item in train_set:
#         file.write(f"{item}\n")