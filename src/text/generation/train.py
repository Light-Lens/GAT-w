from src.text.generation.model import GPT, set_params
from src.text.generation.utils import encode
import torch, time

# hyperparameters
batch_size = 16 # how many independent sequences will we process in parallel?
block_size = 64 # what is the maximum context length for predictions?
max_iters = 10000
eval_interval = 200
learning_rate = 1e-3
device = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_iters = 200
n_embd = 64
n_head = 5
n_layer = 5
dropout = 0
savepath = "models\\GAT-w2.pth"
# ------------

with open('data\\data.txt', 'r', encoding='utf-8') as f:
    text = f.read()

# here are all the unique characters that occur in this text
chars = sorted(list(set(text)))
vocab_size = len(chars)
# create a mapping from characters to integers
stoi = { ch:i for i,ch in enumerate(chars) }
itos = { i:ch for i,ch in enumerate(chars) }

# Train and test splits
data = torch.tensor(encode(text, stoi=stoi), dtype=torch.long)
n = int(0.8 * len(data)) # first 80% will be train, rest val
train_data = data[:n]
val_data = data[n:]

# data loading
def get_batch(split):
    # generate a small batch of data of inputs x and targets y
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y

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

# Set parameters and Create an instance of GPT
set_params(_n_embd=n_embd, _n_head=n_head, _n_layer=n_layer, _block_size=block_size, _dropout=dropout, _vocab_size=vocab_size, _device=device)
model = GPT()
m = model.to(device)
# print the number of parameters in the model
print(sum(p.numel() for p in m.parameters())/1e6, 'M parameters')

# create a PyTorch optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

# start timer
start_time = time.perf_counter()

for iter in range(max_iters):
    try:
        if (iter + 1) % eval_interval == 0 or iter == max_iters - 1:
            losses = estimate_loss()
            print(f"step [{iter + 1}/{max_iters}]: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

        # sample a batch of data
        xb, yb = get_batch('train')

        # evaluate the loss
        logits, loss = model(xb, yb)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

    except KeyboardInterrupt:
        break

print(f"Time taken: {(time.perf_counter() - start_time):.0f} sec")
torch.save(
    {
        "state_dict": model.state_dict(),
        "stoi": stoi,
        "itos": itos,
        "device": device,
        "config": {
            "n_embd": n_embd,
            "n_head": n_head,
            "n_layer": n_layer,
            "block_size": block_size,
            "dropout": dropout,
            "vocab_size": vocab_size
        }
    },
    savepath
)
