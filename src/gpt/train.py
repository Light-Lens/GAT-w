from colorama import init, Fore, Style
from .model import GPTConfig, GPT
import matplotlib.pyplot as plt
import torch, time, os

init(autoreset=True)

class GPTTrainConfig:
    """
    1. `n_steps`: Number of epochs to train the model for
    2. `eval_interval`: The interval between each loss evaluation
    3. `eval_iters`: The iterations for each loss evaluation
    4. `checkpoint_interval`: The interval between each checkpoint save
    5. `checkpoint_path`: The save path for the checkpoint.
    
    For eg,
    ```python
    GPTTrainConfig.checkpoint_path = "checkpoints\\checkpoint.pth"
    ```

    6. `n_loss_digits`: Number of digits of train and val loss printed `(default: 4)`
    """

    n_steps = 1000
    eval_interval = 100
    eval_iters = 100
    checkpoint_interval = 0
    checkpoint_path = ""
    n_loss_digits = 4

class train:
    def __init__(self, batch_size):
        GPTConfig.device = ("cuda" if torch.cuda.is_available() else "cpu") if GPTConfig.device == "auto" else GPTConfig.device
        # how many independent sequences will we process in parallel?
        self.batch_size = batch_size

        # a dict for keep track of all the losses to be plotted.
        self.losses = {
            "train": [],
            "val": []
        }

        # print the device
        print("Training on", f"{Fore.YELLOW}{Style.BRIGHT}{GPTConfig.device}")

    def prepare(self, data, data_division):
        """
        1. `data`: The encoded training text data.

        For eg,
        ```python
        torch.tensor(encode(text, stoi=self.stoi), dtype=torch.long)
        ```

        2. `data_division`: The first `(data_division * 100)%` will be train, rest val
        """

        # train and test splits
        n = int(data_division * len(data)) # the first (data_division * 100)% will be train, rest val
        self.train_data = data[:n]
        self.val_data = data[n:]

        # print the number of tokens
        print(f"{Fore.WHITE}{Style.BRIGHT}{(len(data)/1e6)}", "M total tokens")
        print(f"{Fore.WHITE}{Style.BRIGHT}{(len(self.train_data)/1e6)}", "M train tokens,", f"{Fore.WHITE}{Style.BRIGHT}{(len(self.val_data)/1e6)}", "M test tokens")

    # data loading
    def get_batch(self, split):
        # generate a small batch of data of inputs x and targets y
        data = self.train_data if split == 'train' else self.val_data
        ix = torch.randint(len(data) - GPTConfig.block_size, (self.batch_size,))
        x = torch.stack([data[i:i+GPTConfig.block_size] for i in ix])
        y = torch.stack([data[i+1:i+GPTConfig.block_size+1] for i in ix])
        x, y = x.to(GPTConfig.device), y.to(GPTConfig.device)
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

    def train(self, lr):
        """
        ```python
        return {
            "state_dict": self.model.state_dict(),
            "device": GPTConfig.device,
            "config": {
                "n_embd": GPTConfig.n_embd,
                "n_head": GPTConfig.n_head,
                "n_layer": GPTConfig.n_layer,
                "block_size": GPTConfig.block_size,
                "dropout": GPTConfig.dropout,
                "vocab_size": GPTConfig.vocab_size
            }
        ```
        """

        # Create an instance of GPT
        self.model = GPT()
        m = self.model.to(GPTConfig.device)
        # print the number of parameters in the model
        print(f"{Fore.WHITE}{Style.BRIGHT}{(sum(p.numel() for p in m.parameters())/1e6)}", 'M parameters')

        # create a PyTorch optimizer
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr)

        # start timer
        start_time = time.perf_counter()

        # train the model for n_steps
        for iter in range(GPTTrainConfig.n_steps):
            try:
                if (iter + 1) % GPTTrainConfig.eval_interval == 0 or iter == GPTTrainConfig.n_steps - 1:
                    losses = self.estimate_loss(GPTTrainConfig.eval_iters)
                    print(f"step [{iter + 1}/{GPTTrainConfig.n_steps}]: train loss {losses['train']:.{GPTTrainConfig.n_loss_digits}f}, val loss {losses['val']:.{GPTTrainConfig.n_loss_digits}f}")
                    self.losses["train"].append(losses['train'])
                    self.losses["val"].append(losses['val'])

                # sample a batch of data
                xb, yb = self.get_batch('train')

                # evaluate the loss
                logits, loss = self.model(xb, yb)
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()

                self.__save_checkpoints__()

            except KeyboardInterrupt:
                break

        print(f"Time taken: {Fore.BLUE}{Style.BRIGHT}{(time.perf_counter() - start_time):.0f} sec")

        return {
            "state_dict": self.model.state_dict(),
            "device": GPTConfig.device,
            "config": {
                "n_embd": GPTConfig.n_embd,
                "n_head": GPTConfig.n_head,
                "n_layer": GPTConfig.n_layer,
                "block_size": GPTConfig.block_size,
                "dropout": GPTConfig.dropout,
                "vocab_size": GPTConfig.vocab_size
            }
        }

    def __save_checkpoints__(self):
        if GPTTrainConfig.checkpoint_interval != 0 and GPTTrainConfig.checkpoint_path != "" and (iter + 1) % GPTTrainConfig.checkpoint_interval == 0:
            # split the filepath into path, filename, and extension
            path, filename_with_extension = os.path.split(GPTTrainConfig.checkpoint_path)
            filename, extension = os.path.splitext(filename_with_extension)

            # save the model checkpoint
            self.save(os.path.join(path, f"{filename}_{(iter + 1)}{extension}"))

    def plot(self, path):
        plt.style.use("seaborn-v0_8-dark")

        for param in ['figure.facecolor', 'axes.facecolor', 'savefig.facecolor']:
            plt.rcParams[param] = '#212946'  # bluish dark grey

        for param in ['text.color', 'axes.labelcolor', 'xtick.color', 'ytick.color']:
            plt.rcParams[param] = '0.9'  # very light grey

        plt.figure(figsize=(18, 8))
        plt.plot(self.losses["train"], label="train loss")
        plt.plot(self.losses["val"], label="val loss")

        plt.xlabel("iteration", fontsize=12)
        plt.ylabel("value", fontsize=12)
        plt.legend(fontsize=12)
        plt.title("train-val loss", fontsize=14)
        plt.savefig(path, bbox_inches="tight")
        plt.close()
