from ...utils import tokenize, encode
from ...models.FeedForward import FeedForwardConfig, FeedForward
from ...models.RNN import RNNConfig, RNN
import torch, json, time, os

def one_hot_encode(sequence, dict_size, seq_len, batch_size, device):
    # Creating a multi-dimensional array of zeros with the desired output shape
    features = torch.zeros((batch_size, seq_len, dict_size), dtype=torch.float32, device=device)
    
    # Replacing the 0 at the relevant character index with a 1 to represent that character
    for i in range(batch_size):
        for u in range(seq_len):
            features[i, u, sequence[i][u]] = 1

    return features

class Train:
    def __init__(self, n_layer, n_hidden, lr, batch_size, model="FeedForward", device="auto"):
        """
        @param n_layer: Number of layers
        @param n_hidden: Hidden size
        @param lr: Learning rate
        @param model: Model architecture to train on. [FeedForward, RNN] (default: FeedForward)
        @param device: Training device. [auto, cpu, cuda] (default: auto)
        """

        # hyperparameters
        self.n_hidden = n_hidden
        self.n_layer = n_layer
        self.device = device
        self.learning_rate = lr
        self.batch_size = batch_size # how many independent sequences will we process in parallel?
        self.device = ("cuda" if torch.cuda.is_available() else "cpu") if device == "auto" else device
        self.model_architecture = model

        # print the device
        print("Training on", self.device)

    def preprocess(self, filepath, metadata):
        """
        @param filepath: the location of the json file.
        @param metadata: (classname, pattern_name, output_name)
        """

        with open(filepath, "r", encoding="utf-8") as f:
            jsondata = json.load(f)

        classname, pattern_name, output_name = metadata
        vocab = ["<sep>", "<end>"]
        xy = [] # x: pattern, y: tag

        for intent in jsondata[classname]:
            tokenize_pattern = tokenize(intent[pattern_name])
            vocab.extend(tokenize_pattern)

            data = []
            data.extend(tokenize_pattern)
            data.append("<sep>")
            data.extend(tokenize(intent[output_name]))
            data.append("<end>")
            xy.append(data)

        # here are all the unique characters that occur in this text
        vocab = sorted(list(set(vocab)))
        # create a mapping from characters to integers
        self.stoi = { ch:i for i,ch in enumerate(vocab) }
        self.itos = { i:ch for i,ch in enumerate(vocab) }

        # Padding
        # A simple loop that loops through the list of sentences and,
        # adds '<end>' token until the length of the sentence
        # matches the length of the longest sentence
        maxlen = len(max(xy, key=len))
        self.seq_len = maxlen - 1
        for i in range(len(xy)):
            while len(xy[i]) < maxlen:
                xy[i].append("<end>")

        # Prepare training data
        self.train_data = []
        for i in xy:
            self.train_data.append(torch.tensor(encode(i, stoi=self.stoi), dtype=torch.long))

        # print the number of tokens
        print(len(vocab), "input-output size")
        print(len(self.train_data)/1e6, "M train data")

    # data loading
    def get_batch(self):
        # generate a small batch of data of inputs x and targets y
        ix = torch.randint(len(self.train_data) - 1, (self.batch_size,))
        x = one_hot_encode(
            [
                self.train_data[i][:-1]
                for i in ix
            ],
            len(self.stoi), self.seq_len, self.batch_size, self.device
        )
        y = one_hot_encode(
            [
                self.train_data[i][1:]
                for i in ix
            ],
            len(self.stoi), self.seq_len, self.batch_size, self.device
        )
        return x, y

    @torch.no_grad()
    def estimate_loss(self, eval_iters):
        out = None
        self.model.eval()
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = self.get_batch()
            _, loss = self.model(X, Y)
            losses[k] = loss.item()
        out = losses.mean()
        self.model.train()
        return out

    def train(self, n_steps, eval_interval, eval_iters, checkpoint_interval=0, checkpoint_path=""):
        """
        @param n_steps: number of Epochs to train the model for
        @param eval_interval: the interval between each loss evaluation
        @param eval_iters: the iterations for each loss evaluation
        @param checkpoint_interval: the interval between each checkpoint save
        @param checkpoint_path: the save path for the checkpoint
        """

        # set hyperparameters
        if self.model_architecture == "FeedForward":
            FeedForwardConfig.n_layer = self.n_layer
            FeedForwardConfig.n_hidden = self.n_hidden
            FeedForwardConfig.input_size = len(self.stoi)
            FeedForwardConfig.output_size = len(self.stoi)

            # create an instance of FeedForward network
            self.model = FeedForward()

        elif self.model_architecture == "RNN":
            RNNConfig.n_layer = self.n_layer
            RNNConfig.n_hidden = self.n_hidden
            RNNConfig.input_size = len(self.stoi)
            RNNConfig.output_size = len(self.stoi)

            # create an instance of RNN network
            self.model = RNN()

        else:
            raise Exception(f"{self.model_architecture}: Invalid model architecture.\nAvailable architectures are FeedForward, RNN")

        m = self.model.to(self.device)
        # print the number of parameters in the model
        print(sum(p.numel() for p in m.parameters())/1e6, 'M parameters')

        # create a PyTorch optimizer
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.learning_rate)

        # start timer
        start_time = time.perf_counter()

        # train the model for n_steps
        for iter in range(n_steps):
            try:
                if (iter + 1) % eval_interval == 0 or iter == n_steps - 1:
                    losses = self.estimate_loss(eval_iters)
                    print(f"step [{iter + 1}/{n_steps}]: train loss {losses:.4f}")

                # sample a batch of data
                xb, yb = self.get_batch()

                # evaluate the loss
                _, loss = self.model(xb, yb)
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()

                if checkpoint_interval != 0 and checkpoint_path != "" and (iter + 1) % checkpoint_interval == 0:
                    # split the filepath into path, filename, and extension
                    path, filename_with_extension = os.path.split(checkpoint_path)
                    filename, extension = os.path.splitext(filename_with_extension)

                    # save the model checkpoint
                    self.save(os.path.join(path, f"{filename}_{(iter + 1)}{extension}"))

            except KeyboardInterrupt:
                break

        print(f"Time taken: {(time.perf_counter() - start_time):.0f} sec")

    def save(self, savepath):
        torch.save(
            {
                "state_dict": self.model.state_dict(),
                "model": self.model_architecture,
                "stoi": self.stoi,
                "itos": self.itos,
                "device": self.device,
                "config": {
                    "n_hidden": self.n_hidden,
                    "n_layer": self.n_layer,
                    "seq_len": self.seq_len,
                    "batch_size": self.batch_size
                }
            },
            savepath
        )
