from ...utils import remove_special_chars, tokenize, encode
from ...models.RNN import RNNConfig, RNN
import torch, json, time, os

class RNNForTextExtraction(RNN):
    def predict(self, x):
        out, _ = self(x)
        prob = torch.softmax(out[-1], dim=0).data
        # Taking the class with the highest probability score from the output
        return torch.max(prob, dim=0)[1].item()

class Train:
    def __init__(self, n_layer, n_hidden, lr, batch_size, device="auto"):
        # hyperparameters
        self.n_hidden = n_hidden
        self.n_layer = n_layer
        self.device = device
        self.learning_rate = lr
        self.batch_size = batch_size # how many independent sequences will we process in parallel?
        self.device = ("cuda" if torch.cuda.is_available() else "cpu") if device == "auto" else device

        # print the device
        print("Training on", self.device)

    def preprocess(self, filepath, metadata):
        """
        @param filepath: the location of the json file.
        @param metadata: (classname, pattern_name, desired_output_name)
        """

        with open(filepath, 'r', encoding='utf-8') as f:
            jsondata = json.load(f)

        classname, pattern_name, output_name = metadata

        # Prepare vocab
        vocab = ["<sep>", "<end>", " "]
        xy = []
        for intent in jsondata[classname]:
            pattern = intent[pattern_name]
            vocab.extend(pattern)

            data = []
            data.extend(remove_special_chars(pattern))
            data.append("<sep>")
            data.extend(remove_special_chars(intent[output_name]))
            data.append("<end>")
            xy.append(data)

        # here are all the unique characters that occur in this text
        vocab = sorted(list(set(vocab)))

        # create a mapping from characters to integers
        self.stoi = { ch:i for i,ch in enumerate(vocab) }
        self.itos = { i:ch for i,ch in enumerate(vocab) }

        # Padding
        # A simple loop that loops through the list of sentences and adds a ' ' whitespace until the length of the sentence matches
        # the length of the longest sentence
        maxlen = len(max(xy, key=len))
        self.input_size = maxlen - 1
        for i in range(len(xy)):
            while len(xy[i]) < maxlen:
                xy[i].append("<end>")

        # Prepare training data
        self.train_data = []
        for i in xy:
            self.train_data.append(torch.tensor(encode(i, stoi=self.stoi), dtype=torch.float32))

        # print the number of tokens
        print(self.input_size, "input-output size")
        print(len(self.train_data)/1e6, "M train data")

    # data loading
    def get_batch(self):
        # generate a small batch of data of inputs x and targets y
        ix = torch.randint(len(self.train_data) - 1, (self.batch_size,))
        x = torch.stack([self.train_data[i][:-1] for i in ix])
        y = torch.stack([self.train_data[i][1:] for i in ix])
        x, y = x.to(self.device), y.to(self.device)
        return x, y

    @torch.no_grad()
    def estimate_loss(self, eval_iters):
        loss_mean = None
        self.model.eval()
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = self.get_batch()
            out, loss = self.model(X, Y)
            losses[k] = loss.item()
        loss_mean = losses.mean()
        self.model.train()
        return loss_mean

    def train(self, n_steps, eval_interval, eval_iters, checkpoint_interval=0, checkpoint_path=""):
        """
        @param n_steps: number of Epochs to train the model for
        @param eval_interval: the interval between each loss evaluation
        @param eval_iters: the iterations for each loss evaluation
        @param checkpoint_interval: the interval between each checkpoint save
        @param checkpoint_path: the save path for the checkpoint
        """

        # set hyperparameters
        RNNConfig.n_layer = self.n_layer
        RNNConfig.n_hidden = self.n_hidden
        RNNConfig.input_size = self.input_size
        RNNConfig.output_size = self.input_size

        # create an instance of FeedForward network
        self.model = RNNForTextExtraction()
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
                "stoi": self.stoi,
                "itos": self.itos,
                "device": self.device,
                "config": {
                    "n_hidden": self.n_hidden,
                    "n_layer": self.n_layer,
                    "input_size": self.input_size
                }
            },
            savepath
        )
