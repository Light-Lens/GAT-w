import torch

class sample:
    def __init__(self, model_path):
        # Load the saved model
        model_data = torch.load(model_path)

        self.state_dict = model_data["state_dict"]
        self.stoi = model_data["stoi"]
        self.itos = model_data["itos"]
        self.device = model_data["device"]
        self.n_embd = model_data["config"]["n_embd"]
        self.n_head = model_data["config"]["n_head"]
        self.n_layer = model_data["config"]["n_layer"]
        self.block_size = model_data["config"]["block_size"]
        self.dropout = model_data["config"]["dropout"]
        self.vocab_size = model_data["config"]["vocab_size"]

    def load(self):
        pass

    # Use the model for generation or other tasks
    def classify(self):
        pass
