from src.text.generation.model import GPT, set_params
from src.text.generation.utils import encode
import torch, time

class train:
    def __init__(self, n_layer, n_embd, n_head, lr, dropout, block_size, batch_size):
        # hyperparameters
        self.n_layer = n_layer
        self.n_embd = n_embd
        self.n_head = n_head
        self.learning_rate = lr
        self.dropout = dropout
        self.block_size = block_size # what is the maximum context length for predictions?
        self.batch_size = batch_size # how many independent sequences will we process in parallel?
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

    def preprocess_data(self, filepath):
        with open(filepath, 'r', encoding='utf-8') as f:
            text = f.read()

        # here are all the unique characters that occur in this text
        chars = sorted(list(set(text)))
        self.vocab_size = len(chars)
        # create a mapping from characters to integers
        self.stoi = { ch:i for i,ch in enumerate(chars) }
        self.itos = { i:ch for i,ch in enumerate(chars) }

        # Train and test splits
        data = torch.tensor(encode(text, stoi=self.stoi), dtype=torch.long)
        n = int(0.8 * len(data)) # first 80% will be train, rest val
        self.train_data = data[:n]
        self.val_data = data[n:]

    # data loading
    def get_batch(self, split):
        # generate a small batch of data of inputs x and targets y
        data = self.train_data if split == 'train' else self.val_data
        ix = torch.randint(len(data) - self.block_size, (self.batch_size,))
        x = torch.stack([data[i:i+self.block_size] for i in ix])
        y = torch.stack([data[i+1:i+self.block_size+1] for i in ix])
        x, y = x.to(self.device), y.to(self.device)
        return x, y

    @torch.no_grad()
    def estimate_loss(self, eval_iters):
        out = {}
        self.model.eval()
        for split in ['train', 'val']:
            losses = torch.zeros(eval_iters)
            for k in range(eval_iters):
                X, Y = self.get_batch(split)
                logits, loss = self.model(X, Y)
                losses[k] = loss.item()
            out[split] = losses.mean()
        self.model.train()
        return out

    def train(self, n_steps, eval_interval, eval_iters):
        # Set hyperparameters
        set_params(
            _n_embd = self.n_embd,
            _n_head = self.n_head,
            _n_layer = self.n_layer,
            _block_size = self.block_size,
            _dropout = self.dropout,
            _vocab_size = self.vocab_size,
            _device = self.device
        )

        # Create an instance of GPT
        self.model = GPT()
        m = self.model.to(self.device)
        # print the number of parameters in the model
        print(sum(p.numel() for p in m.parameters())/1e6, 'M parameters')

        # create a PyTorch optimizer
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.learning_rate)

        # start timer
        start_time = time.perf_counter()

        for iter in range(n_steps):
            try:
                if (iter + 1) % eval_interval == 0 or iter == n_steps - 1:
                    losses = self.estimate_loss(eval_iters)
                    print(f"step [{iter + 1}/{n_steps}]: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

                # sample a batch of data
                xb, yb = self.get_batch('train')

                # evaluate the loss
                logits, loss = self.model(xb, yb)
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()

            except KeyboardInterrupt:
                break

        print(f"Time taken: {(time.perf_counter() - start_time):.0f} sec")

    def save(self, savepath):
        torch.save(
            {
                "state_dict": self.model.state_dict(),
                "stoi": self.stoi,
                "itos": self.itos,
                "device": self.device,
                "config": {
                    "n_embd": self.n_embd,
                    "n_head": self.n_head,
                    "n_layer": self.n_layer,
                    "block_size": self.block_size,
                    "dropout": self.dropout,
                    "vocab_size": self.vocab_size
                }
            },
            savepath
        )
