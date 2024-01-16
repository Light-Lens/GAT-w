from src.w2.utils import encode, decode
from src.w2.model import GPT, set_params
import torch

class sample:
    def __init__(self, model_path):
        # Load the saved model
        model_data = torch.load("models\\GAT-w2.pth")

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
        # Set hyperparameters
        set_params(
            _n_embd = self.n_embd,
            _n_head = self.n_head,
            _n_layer = self.n_layer,
            _block_size = self.block_size,
            _dropout = self.dropout,
            _vocab_size = self.vocab_size,
            _device = self.device
        )

        # Create an instance of GPT
        self.model = GPT()

        # Load the saved model state_dict
        self.model.load_state_dict(self.state_dict)
        self.model.to(self.device)
        self.model.eval()  # Set the model to evaluation mode

    # Use the model for generation or other tasks
    def generate(self, text):
        context = torch.tensor(encode(text, stoi=self.stoi), dtype=torch.long, device=self.device).unsqueeze(0)
        return decode(self.model.generate(context, max_new_tokens=100)[0].tolist(), itos=self.itos)
