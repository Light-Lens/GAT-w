from .model import RNNConfig, RNN
import torch

class sample:
    def __init__(self, model_path):
        # Load the saved model
        model_data = torch.load(model_path)

        self.state_dict = model_data["state_dict"]
        RNNConfig.device = model_data["device"]
        RNNConfig.input_size = model_data["config"]["input_size"]
        RNNConfig.output_size = model_data["config"]["output_size"]
        RNNConfig.n_hidden = model_data["config"]["n_hidden"]
        RNNConfig.n_embd = model_data["config"]["n_embd"]
        RNNConfig.n_layer = model_data["config"]["n_layer"]

    def load(self):
        # Create an instance of RNN
        self.model = RNN()

        # Load the saved model state_dict
        self.model.load_state_dict(self.state_dict)
        self.model.to(RNNConfig.device)
        self.model.eval()  # Set the model to evaluation mode

    # use the model for classification or other tasks
    def predict(self, X, classes):
        X = X.reshape(1, X.shape[0])
        X = torch.tensor(X).to(RNNConfig.device)

        i, confidence = self.model.predict(X, classes)
        tag = classes[i]

        return tag, confidence
