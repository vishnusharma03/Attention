import torch
import torch.nn as nn
from torch.nn import functional as F
torch.manual_seed(1337)


batch_size = 32
block_size = 8
max_iter = 10000
eval_interval = 500
learning_rate = 1e-3
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
eval_iter = 200
n_embed = 32
head_size = 8

# wget https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt


with open('input.txt', 'r', encoding='utf-8') as f:
    text = f.read()


chars = sorted(list(set(text)))
vocab_size = len(chars)


# Here we are tokenizing at character level
stoi = {ch:i for i,ch in enumerate(chars)} # string to integer mapping
itos = {i:ch for i,ch in enumerate(chars)} # integer to string mapping
encode = lambda s : [stoi[c] for c in s] # encode a string to a list of integers
decode = lambda l : ''.join([itos[i] for i in l]) # decode a list of integer to a sting


data = torch.tensor(encode(text), dtype=torch.long)


n = int(0.9*len(data))
train_data = data[:n]
valid_data = data[n:]


def get_batch(split):
    data = train_data if split=='train' else valid_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    x, y = x.to(device), y. to(device)
    return x, y

@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'valid']:
        losses = torch.zeros(eval_iter)
        for k in range(eval_iter):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

class Head(nn.Module):
    """One of the self attention heads"""
    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embed, head_size, bias=False)
        self.query = nn.Linear(n_embed, head_size, bias=False)
        self.value = nn.Linear(n_embed, head_size, bias=False)
        self.register_buffer("tril", torch.tril(torch.ones(block_size, block_size))) 

    def forward(self, x):
        B,T,C = x.shape
        k = self.key(x) # (B, T, C)
        q = self.query(x) # (B, T, C)
        # computing attention scores ('affinities')
        wei = q @ k.transpose(-2, -1) * C**-0.5 # (B, T, C) @ (B, C, T) -> (B, T, T)
        wei = wei.masked_fill(self.tril[:T, :T]==0, float('-inf')) # (B, T, T)
        wei = F.softmax(wei, dim=-1) # (B, T, T)

        # performing the weighted sum
        v = self.value(x) # (B, T, C)
        out = wei @ v # (B, T, T) @ (B, T, C) -> (B, T, C)
        return out

class BigramLanguageModel(nn.Module):

    def __init__(self):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, n_embed)
        self.position_embedding_table = nn.Embedding(block_size, n_embed)
        self.sa_head = Head(n_embed) 
        self.lm_head = nn.Linear(n_embed, vocab_size)

    def forward(self, idx, targets=None):
        B,T = idx.shape # Batch size and Time

        tok_emb = self.token_embedding_table(idx) # (This will be in the dimension of Batch x Time x Channels)
        pos_emb = self.position_embedding_table(torch.arange(T, device=device)) # (T, C) -> (1, T, C)
        x = tok_emb + pos_emb # (B, T, C); x holds the positional embeddings & token embeddings
        x = self.sa_head(x) # (B, T, C)
        logits = self.lm_head(x) # (This will be in the dimension of Batch x Time x Vocab_size)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)   
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    def generate(self, idx, max_new_tokens): # idx is (B,T) array of indices in the current context
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -block_size:]  # Crop the context so that tokens does not exceed block_size otherwise 
            logits, loss = self(idx_cond) # getting the predictions
            logits = logits[:, -1, :] # plucking out the last step result
            probs = F.softmax(logits, dim=-1) # getting the probabilities
            idx_next = torch.multinomial(probs, num_samples=1) # sample from distribution
            idx = torch.cat((idx, idx_next), dim=1) # append sampled index to running sequence
        return idx
            

model = BigramLanguageModel()
m = model.to(device)


optimizer = torch.optim.AdamW(m.parameters(), lr=1e-3)

for iter in range(max_iter):

    if iter % eval_interval == 0:
        losses = estimate_loss()
        print(f"Step: {iter}, train loss: {losses['train']:.4f}, valid loss: {losses['valid']:.4f}")

    xb, yb = get_batch('train')

    logits, loss = m(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()


context = torch.zeros((1, 1), dtype=torch.long, device=device)

print(decode(m.generate(context, max_new_tokens=500)[0].tolist()))




