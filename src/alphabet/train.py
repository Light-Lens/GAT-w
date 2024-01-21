from src.alphabet.utils import one_hot_encoding, stop_words, tokenize
from src.alphabet.model import FeedForwardConfig, FeedForward
import torch, numpy, json, time, os

class train:
    def __init__(self, n_layer, n_hidden, lr, batch_size, device="auto"):
        # hyperparameters
        self.n_hidden = n_hidden
        self.n_layer = n_layer
        self.device = device
        self.learning_rate = lr
        self.batch_size = batch_size # how many independent sequences will we process in parallel?
        if device == "auto":
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        else:
            self.device = device

        # print the device
        print("Training on", self.device)

    def preprocess(self, filepath, metadata, data_division=0.8):
        """
        @param filepath: the location of the json file.
        @param metadata: (classname, tagname, pattern_name)
        @param data_division: split the dataset into train and val data
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

        # Stem, lower each word and remove stop words
        self.vocab = stop_words(self.vocab)

        # Remove duplicates and sort
        self.vocab = sorted(set(self.vocab))
        self.classes = sorted(set(self.classes))

        # Train and test splits
        data = []
        for x, y in xy:
            data.append((one_hot_encoding(x, self.vocab), self.classes.index(y)))

        n = int(data_division * len(data)) # the first (data_division * 100)% will be train, rest val
        self.train_data = data[:n]
        self.val_data = data[n:]

        # print the number of tokens
        print(len(xy)/1e6, "M total tokens")
        print(len(self.vocab), "vocab size,", len(self.classes), "output size,")
        print(len(self.train_data)/1e6, "M train data,", len(self.val_data)/1e6, "M test data")

    # data loading
    def get_batch(self, split):
        # generate a small batch of data of inputs x and targets y
        data = self.train_data if split == 'train' else self.val_data
        ix = torch.randint(len(data) - 1, (self.batch_size,))
        x = torch.stack([torch.tensor(data[i][0]) for i in ix])
        y = torch.stack([torch.tensor(data[i][1]) for i in ix])
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
                output, loss = self.model(X, Y)
                losses[k] = loss.item()
            out[split] = losses.mean()
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
        FeedForwardConfig.n_layer = self.n_layer
        FeedForwardConfig.n_hidden = self.n_hidden
        FeedForwardConfig.input_size = len(self.vocab)
        FeedForwardConfig.output_size = len(self.classes)

        # create an instance of FeedForward network
        self.model = FeedForward()
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
                    print(f"step [{iter + 1}/{n_steps}]: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

                # sample a batch of data
                xb, yb = self.get_batch('train')

                # evaluate the loss
                out, loss = self.model(xb, yb)
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
