from ..utils import one_hot_encoding, remove_special_chars, tokenize
from ..models.FeedForward import FeedForwardConfig, FeedForward
from ..models.RNN import RNNConfig, RNN
import torch, json, time, os

class Train:
    def __init__(self, n_layer, n_hidden, lr, batch_size, model="FeedForward", device="auto"):
        """
        @param n_layer: Number of layers
        @param n_hidden: Hidden size
        @param lr: Learning rate
        @param model: Model architecture to train on. [FeedForward, RNN] (default: FeedForward)
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
        @param metadata: (classname, tagname, pattern_name)
        """

        with open(filepath, 'r', encoding='utf-8') as f:
            jsondata = json.load(f)

        classname, tagname, pattern_name = metadata
        self.classes = []
        self.vocab = []
        xy = [] # x: pattern, y: tag

        for intent in jsondata[classname]:
            y_encode = f"{classname};{intent[tagname]}"
            self.classes.append(y_encode)

            for pattern in intent[pattern_name]:
                tokenized_words = tokenize(pattern)
                self.vocab.extend(tokenized_words)
                xy.append((tokenized_words, y_encode))

        # Lemmatize, lower each word and remove unnecessary chars.
        self.vocab = remove_special_chars(self.vocab)

        # Remove duplicates and sort
        self.vocab = sorted(set(self.vocab))
        self.classes = sorted(set(self.classes))

        # Train and test splits
        self.train_data = []
        for x, y in xy:
            self.train_data.append((one_hot_encoding(x, self.vocab), self.classes.index(y)))

        # print the number of tokens
        print(len(xy)/1e6, "M total tokens")
        print(len(self.vocab), "vocab size,", len(self.classes), "output size,")
        print(len(self.train_data)/1e6, "M train data")

    # data loading
    def get_batch(self):
        # generate a small batch of data of inputs x and targets y
        ix = torch.randint(len(self.train_data) - 1, (self.batch_size,))
        x = torch.stack([torch.tensor(self.train_data[i][0]) for i in ix])
        y = torch.stack([torch.tensor(self.train_data[i][1]) for i in ix])
        x, y = x.to(self.device), y.to(self.device)
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
            FeedForwardConfig.input_size = len(self.vocab)
            FeedForwardConfig.output_size = len(self.classes)

            # create an instance of FeedForward network
            self.model = FeedForward()

        elif self.model_architecture == "RNN":
            RNNConfig.n_layer = self.n_layer
            RNNConfig.n_hidden = self.n_hidden
            RNNConfig.input_size = len(self.vocab)
            RNNConfig.output_size = len(self.classes)

            # create an instance of RNN network
            self.model = RNN()

        else:
            raise f"{self.model_architecture}: Invalid model architecture.\nAvailable architectures are FeedForward, RNN"

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
                "vocab": self.vocab,
                "classes": self.classes,
                "device": self.device,
                "config": {
                    "n_hidden": self.n_hidden,
                    "n_layer": self.n_layer
                }
            },
            savepath
        )
