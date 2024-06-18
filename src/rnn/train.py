from colorama import init, Fore, Style
from .model import RNNConfig, RNN
import matplotlib.pyplot as plt
import torch, time, os

init(autoreset=True)

class RNNTrainConfig:
    """
    1. `n_steps`: Number of epochs to train the model for
    2. `eval_interval`: The interval between each loss evaluation
    3. `eval_iters`: The iterations for each loss evaluation
    4. `checkpoint_interval`: The interval between each checkpoint save
    5. `checkpoint_path`: The save path for the checkpoint.
    
    For eg,
    ```python
    RNNTrainConfig.checkpoint_path = "checkpoints\\checkpoint.pth"
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
        RNNConfig.device = ("cuda" if torch.cuda.is_available() else "cpu") if RNNConfig.device == "auto" else RNNConfig.device
        # how many independent sequences will we process in parallel?
        self.batch_size = batch_size

        #NOTE: SUPPORTS DATA DIVISION ONLY UNCOMMENT WHEN DATA DIVISION IS NEEDED.
        # a dict for keep track of all the losses to be plotted.
        # self.losses = {
        #     "train": [],
        #     "val": []
        # }

        # print the device
        print("Training on", f"{Fore.YELLOW}{Style.BRIGHT}{RNNConfig.device}")

    def prepare(self, data):
        """
        1. `data`: The encoded training text data.

        For eg,
        ```python
        torch.tensor(encode(text, stoi=self.stoi), dtype=torch.long)
        ```
        """

        self.data = data

        #NOTE: SUPPORTS DATA DIVISION ONLY UNCOMMENT WHEN DATA DIVISION IS NEEDED.
        # train and test splits
        # n = int(data_division * len(data)) # the first (data_division * 100)% will be train, rest val
        # self.train_data = data[:n]
        # self.val_data = data[n:]

        # print the number of tokens
        print(f"{Fore.WHITE}{Style.BRIGHT}{(len(data)/1e6)}", "M total tokens")
        print(RNNConfig.input_size, "vocab size,", RNNConfig.output_size, "output size")

    # data loading
    def get_batch(self):
        # generate a small batch of data of inputs x and targets y
        ix = torch.randint(len(self.data) - 1, (self.batch_size,))
        x = torch.stack([torch.tensor(self.data[i][0]) for i in ix])
        y = torch.stack([torch.tensor(self.data[i][1]) for i in ix])
        x, y = x.to(RNNConfig.device), y.to(RNNConfig.device)
        return x, y

    #NOTE: SUPPORTS DATA DIVISION ONLY UNCOMMENT WHEN DATA DIVISION IS NEEDED.
    # data loading
    # def get_batch(self, split):
    #     # generate a small batch of data of inputs x and targets y
    #     data = self.train_data if split == 'train' else self.val_data
    #     ix = torch.randint(len(data) - 1, (self.batch_size,))
    #     x = torch.stack([torch.tensor(data[i][0]) for i in ix])
    #     y = torch.stack([torch.tensor(data[i][1]) for i in ix])
    #     x, y = x.to(RNNConfig.device), y.to(RNNConfig.device)
    #     return x, y

    @torch.no_grad()
    def estimate_loss(self, eval_iters):
        out = []
        self.model.eval()
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = self.get_batch()
            _, loss = self.model(X, Y)
            losses[k] = loss.item()
        out = losses.mean()
        self.model.train()
        return out

    #NOTE: SUPPORTS DATA DIVISION ONLY UNCOMMENT WHEN DATA DIVISION IS NEEDED.
    # @torch.no_grad()
    # def estimate_loss(self, eval_iters):
    #     out = {}
    #     self.model.eval()
    #     for split in ['train', 'val']:
    #         losses = torch.zeros(eval_iters)
    #         for k in range(eval_iters):
    #             X, Y = self.get_batch(split)
    #             logits, loss = self.model(X, Y)
    #             losses[k] = loss.item()
    #         out[split] = losses.mean()
    #     self.model.train()
    #     return out

    def train(self, lr):
        # Create an instance of RNN
        self.model = RNN()
        m = self.model.to(RNNConfig.device)
        # print the number of parameters in the model
        print(f"{Fore.WHITE}{Style.BRIGHT}{(sum(p.numel() for p in m.parameters())/1e6)}", 'M parameters')

        # create a PyTorch optimizer
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr)

        # start timer
        start_time = time.perf_counter()

        # train the model for n_steps
        self.losses = []
        for iter in range(RNNTrainConfig.n_steps):
            try:
                if (iter + 1) % RNNTrainConfig.eval_interval == 0 or iter == RNNTrainConfig.n_steps - 1:
                    losses = self.estimate_loss(RNNTrainConfig.eval_iters)
                    print(f"step [{iter + 1}/{RNNTrainConfig.n_steps}]: train loss {losses:.{RNNTrainConfig.n_loss_digits}f}")
                    self.losses.append(losses)
                    #NOTE: SUPPORTS DATA DIVISION ONLY UNCOMMENT WHEN DATA DIVISION IS NEEDED.
                    # print(f"step [{iter + 1}/{RNNTrainConfig.n_steps}]: train loss {losses['train']:.{RNNTrainConfig.n_loss_digits}f}, val loss {losses['val']:.{RNNTrainConfig.n_loss_digits}f}")
                    # self.losses["train"].append(losses['train'])
                    # self.losses["val"].append(losses['val'])

                # sample a batch of data
                xb, yb = self.get_batch()
                # xb, yb = self.get_batch("train") #NOTE: SUPPORTS DATA DIVISION ONLY UNCOMMENT WHEN DATA DIVISION IS NEEDED.

                # evaluate the loss
                _, loss = self.model(xb, yb)
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()

                self.__save_checkpoints__()

            except KeyboardInterrupt:
                break

        print(f"Time taken: {Fore.BLUE}{Style.BRIGHT}{(time.perf_counter() - start_time):.0f} sec")

        return {
            "state_dict": self.model.state_dict(),
            "device": RNNConfig.device,
            "config": {
                "n_hidden": RNNConfig.n_hidden,
                "n_layer": RNNConfig.n_layer,
                "input_size": RNNConfig.input_size,
                "output_size": RNNConfig.output_size
            }
        }

    def __save_checkpoints__(self):
        if RNNTrainConfig.checkpoint_interval != 0 and RNNTrainConfig.checkpoint_path != "" and (iter + 1) % RNNTrainConfig.checkpoint_interval == 0:
            # split the filepath into path, filename, and extension
            path, filename_with_extension = os.path.split(RNNTrainConfig.checkpoint_path)
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
        plt.plot(self.losses, label="train loss")
        #NOTE: SUPPORTS DATA DIVISION ONLY UNCOMMENT WHEN DATA DIVISION IS NEEDED.
        # plt.plot(self.losses["train"], label="train loss")
        # plt.plot(self.losses["val"], label="val loss")

        plt.xlabel("iteration", fontsize=12)
        plt.ylabel("value", fontsize=12)
        plt.legend(fontsize=12)
        plt.title("train-val loss", fontsize=14)
        plt.savefig(path, bbox_inches="tight")
        plt.close()
