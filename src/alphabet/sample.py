from src.alphabet.utils import one_hot_encoding, stop_words, tokenize
from src.alphabet.model import FeedForwardConfig, FeedForward
import torch

class asample:
    def __init__(self, model_path):
        # Load the saved model
        model_data = torch.load(model_path)

        self.state_dict = model_data["state_dict"]
        self.vocab = model_data["vocab"]
        self.classes = model_data["classes"]
        self.device = model_data["device"]
        self.n_hidden = model_data["config"]["n_hidden"]
        self.n_layer = model_data["config"]["n_layer"]

    def load(self):
        # set hyperparameters
        FeedForwardConfig.n_layer = self.n_layer
        FeedForwardConfig.n_hidden = self.n_hidden
        FeedForwardConfig.input_size = len(self.vocab)
        FeedForwardConfig.output_size = len(self.classes)

        # create an instance of FeedForward network
        self.model = FeedForward()

        # load the saved model state_dict
        self.model.load_state_dict(self.state_dict)
        self.model.to(self.device)
        self.model.eval()  # set the model to evaluation mode

    # use the model for classification or other tasks
    def predict(self, text):
        sentence = stop_words(tokenize(text))

        X = one_hot_encoding(sentence, self.vocab)
        X = X.reshape(1, X.shape[0])
        X = torch.tensor(X).to(self.device)

        return self.model.predict(X, self.classes)
