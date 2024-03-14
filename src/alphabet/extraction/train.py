from ...utils import one_hot_encoding, remove_special_chars, tokenize
from ...models.RNN import RNNConfig, RNN
import torch, json, time, os

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
        self.vocab = []
