from ...models.FeedForward import FeedForwardConfig, FeedForward
from ...models.RNN import RNNConfig, RNN
from ...utils import tokenize, encode, decode
from .train import one_hot_encode
import torch

class Sample:
    def __init__(self, model_path):
        # Load the saved model
        model_data = torch.load(model_path)

        self.state_dict = model_data["state_dict"]
        self.model_architecture = model_data["model"]
        self.stoi = model_data["stoi"]
        self.itos = model_data["itos"]
        self.device = model_data["device"]
        self.n_hidden = model_data["config"]["n_hidden"]
        self.n_layer = model_data["config"]["n_layer"]
        self.seq_len = model_data["config"]["seq_len"]
        self.batch_size = model_data["config"]["batch_size"]

    def load(self):
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

        # load the saved model state_dict
        self.model.load_state_dict(self.state_dict)
        self.model.to(self.device)
        self.model.eval()  # set the model to evaluation mode

    # use the model for classification or other tasks
    def predict(self, text):
        sentence = one_hot_encode(
            [torch.tensor(encode(tokenize(text), stoi=self.stoi), dtype=torch.long)],
            len(self.stoi), self.seq_len, self.batch_size, self.device
        )

        return decode(self.model.predict(sentence, self.stoi), itos=self.itos)
