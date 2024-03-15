from ...utils import remove_special_chars, tokenize, encode, decode
from ...models.RNN import RNNConfig
from .train import RNNForTextExtraction
import torch

class Sample:
    def __init__(self, model_path):
        # Load the saved model
        model_data = torch.load(model_path)

        self.state_dict = model_data["state_dict"]
        self.stoi = model_data["stoi"]
        self.itos = model_data["itos"]
        self.device = model_data["device"]
        self.n_hidden = model_data["config"]["n_hidden"]
        self.n_layer = model_data["config"]["n_layer"]
        self.input_size = model_data["config"]["input_size"]

    def load(self):
        # set hyperparameters
        RNNConfig.n_layer = self.n_layer
        RNNConfig.n_hidden = self.n_hidden
        RNNConfig.input_size = self.input_size
        RNNConfig.output_size = self.input_size

        # create an instance of FeedForward network
        self.model = RNNForTextExtraction()

        # load the saved model state_dict
        self.model.load_state_dict(self.state_dict)
        self.model.to(self.device)
        self.model.eval()  # set the model to evaluation mode

    # use the model for classification or other tasks
    def predict(self, text):
        sentence = remove_special_chars(tokenize(text))
        size = 50 - len(sentence)

        # Now pass in the previous characters and get a new one
        for _ in range(size):
            X = torch.tensor(encode(sentence, stoi=self.stoi), dtype=torch.float32, device=self.device).unsqueeze(0)
            sentence.append(decode(self.model.predict(X), itos=self.itos))
        
        return sentence
