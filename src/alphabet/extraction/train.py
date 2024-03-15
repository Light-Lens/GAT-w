from ...utils import remove_special_chars, tokenize
from ...models.RNN import RNNConfig, RNN
import torch, json, time, os

class RNNForTextExtraction(RNN):
    def predict(self, x, vocab):
        out, _ = self(x)
        prob = torch.softmax(out[-1], dim=0).data
        # Taking the class with the highest probability score from the output
        char_ind = torch.max(prob, dim=0)[1].item()

        return vocab[char_ind]

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
        xy = []
        for intent in jsondata[classname]:
            tokenized_pattern = tokenize(intent[pattern_name])
            data = []
            data.extend(tokenize(intent[pattern_name]))
            data.append("<sep>")
            data.extend(tokenize(intent[output_name]))
            xy.append()
            # self.vocab.extend(tokenized_pattern)
            # xy.append((tokenized_pattern, tokenize(intent[output_name])))

        # Remove unnecessary chars, duplicates and sort
        self.vocab = sorted(remove_special_chars(list(set(xy))))

        # Creating a dictionary that maps integers to the words
        int2char = dict(enumerate(xy))

        # Creating another dictionary that maps words to integers
        char2int = {char: ind for ind, char in int2char.items()}

        # Prepare input and target data
        # self.train_data = []
        # for x, y in xy:
        #     xy_to_num = []
        #     xy_to_num.extend([self.vocab.index(i) for i in x if i in self.vocab])
        #     xy_to_num.append(self.vocab.index("<sep>"))
        #     xy_to_num.extend([self.vocab.index(i) for i in y if i in self.vocab])
        #     self.train_data.append(xy_to_num)

        # for i in range(len(self.train_data)):
        #     while len(self.train_data[i]) < len(self.vocab) + 1:
        #         self.train_data[i].append(self.vocab.index("<end>"))

        # print the number of tokens
        print(len(self.vocab), "input-output size")
        print(len(self.train_data)/1e6, "M train data")

    # # data loading
    # def get_batch(self):
    #     # generate a small batch of data of inputs x and targets y
    #     ix = torch.randint(len(self.train_data) - 1, (self.batch_size,))

    #     # input seq
    #     x = torch.stack(
    #         [
    #             torch.tensor(self.train_data[i][:-1], dtype=torch.float32)
    #             for i in ix
    #         ]
    #     )

    #     # target seq
    #     y = torch.stack(
    #         [
    #             torch.tensor(self.train_data[i][1:], dtype=torch.float32)
    #             for i in ix
    #         ]
    #     )

    #     x, y = x.to(self.device), y.to(self.device)
    #     return x, y

    # @torch.no_grad()
    # def estimate_loss(self, eval_iters):
    #     loss_mean = None
    #     self.model.eval()
    #     losses = torch.zeros(eval_iters)
    #     for k in range(eval_iters):
    #         X, Y = self.get_batch()
    #         out, loss = self.model(X, Y)
    #         losses[k] = loss.item()
    #     loss_mean = losses.mean()
    #     self.model.train()
    #     return loss_mean

    # def train(self, n_steps, eval_interval, eval_iters, checkpoint_interval=0, checkpoint_path=""):
    #     """
    #     @param n_steps: number of Epochs to train the model for
    #     @param eval_interval: the interval between each loss evaluation
    #     @param eval_iters: the iterations for each loss evaluation
    #     @param checkpoint_interval: the interval between each checkpoint save
    #     @param checkpoint_path: the save path for the checkpoint
    #     """

    #     # set hyperparameters
    #     RNNConfig.n_layer = self.n_layer
    #     RNNConfig.n_hidden = self.n_hidden
    #     RNNConfig.input_size = len(self.vocab)
    #     RNNConfig.output_size = len(self.vocab)

    #     # create an instance of FeedForward network
    #     self.model = RNNForTextExtraction()
    #     m = self.model.to(self.device)
    #     # print the number of parameters in the model
    #     print(sum(p.numel() for p in m.parameters())/1e6, 'M parameters')

    #     # create a PyTorch optimizer
    #     optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.learning_rate)

    #     # start timer
    #     start_time = time.perf_counter()

    #     # train the model for n_steps
    #     for iter in range(n_steps):
    #         try:
    #             if (iter + 1) % eval_interval == 0 or iter == n_steps - 1:
    #                 losses = self.estimate_loss(eval_iters)
    #                 print(f"step [{iter + 1}/{n_steps}]: train loss {losses:.4f}")

    #             # sample a batch of data
    #             xb, yb = self.get_batch()

    #             # evaluate the loss
    #             out, loss = self.model(xb, yb)
    #             optimizer.zero_grad(set_to_none=True)
    #             loss.backward()
    #             optimizer.step()

    #             if checkpoint_interval != 0 and checkpoint_path != "" and (iter + 1) % checkpoint_interval == 0:
    #                 # split the filepath into path, filename, and extension
    #                 path, filename_with_extension = os.path.split(checkpoint_path)
    #                 filename, extension = os.path.splitext(filename_with_extension)

    #                 # save the model checkpoint
    #                 self.save(os.path.join(path, f"{filename}_{(iter + 1)}{extension}"))

    #         except KeyboardInterrupt:
    #             break

    #     print(f"Time taken: {(time.perf_counter() - start_time):.0f} sec")

    # def save(self, savepath):
    #     torch.save(
    #         {
    #             "state_dict": self.model.state_dict(),
    #             "vocab": self.vocab,
    #             "device": self.device,
    #             "config": {
    #                 "n_hidden": self.n_hidden,
    #                 "n_layer": self.n_layer
    #             }
    #         },
    #         savepath
    #     )
