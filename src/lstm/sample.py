from .model import LSTMConfig, LSTM
import torch

class sample:
    def __init__(self, model_path):
        # Load the saved model
        model_data = torch.load(model_path)

        self.state_dict = model_data["state_dict"]
        LSTMConfig.device = model_data["device"]
        LSTMConfig.n_embd = model_data["config"]["n_embd"]
        LSTMConfig.n_layer = model_data["config"]["n_layer"]
        LSTMConfig.n_hidden = model_data["config"]["n_hidden"]
        LSTMConfig.block_size = model_data["config"]["block_size"]
        LSTMConfig.input_size = model_data["config"]["input_size"]

    def load(self):
        # Create an instance of LSTM
        self.model = LSTM()

        # Load the saved model state_dict
        self.model.load_state_dict(self.state_dict)
        self.model.to(LSTMConfig.device)
        self.model.eval() # Set the model to evaluation mode

    # Use the model for generation or other tasks
    def generate(self, encoded_text=None, length=100, temperature=1.0):
        # Now pass in the previous characters and get a new one

        if encoded_text == None:
            context = torch.zeros((1, 1), dtype=torch.long, device=LSTMConfig.device)

        else:
            context = torch.tensor(encoded_text, dtype=torch.long, device=LSTMConfig.device).unsqueeze(0)

        chars = []
        for _ in range(length):
            char = self.model.predict(context, temperature)
            chars.append(char)

        return chars
