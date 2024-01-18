from src.alphabet.model import GPTConfig, GPT
from src.write.utils import encode, decode
import torch

class sample:
    def __init__(self, model_path):
        # Load the saved model
        model_data = torch.load(model_path)

        self.state_dict = model_data["state_dict"]
        self.stoi = model_data["stoi"]
        self.itos = model_data["itos"]
        self.intents = model_data["intents"]
        self.intents_inv = model_data["intents_inv"]
        self.device = model_data["device"]
        self.num_classes = model_data["config"]["num_classes"]
        self.n_embd = model_data["config"]["n_embd"]
        self.n_head = model_data["config"]["n_head"]
        self.n_layer = model_data["config"]["n_layer"]
        self.block_size = model_data["config"]["block_size"]
        self.dropout = model_data["config"]["dropout"]

    def load(self):
        # Set hyperparameters
        GPTConfig.n_embd = self.n_embd
        GPTConfig.n_head = self.n_head
        GPTConfig.n_layer = self.n_layer
        GPTConfig.block_size = self.block_size
        GPTConfig.dropout = self.dropout
        GPTConfig.vocab_size = len(self.stoi)
        GPTConfig.output_size = self.num_classes
        GPTConfig.device = self.device

        # Create an instance of GPT
        self.model = GPT()

        # Load the saved model state_dict
        self.model.load_state_dict(self.state_dict)
        self.model.to(self.device)
        self.model.eval()  # Set the model to evaluation mode

    # Use the model for generation or other tasks
    def classify(self, text):
        context = torch.tensor(encode(text, stoi=self.stoi), dtype=torch.long, device=self.device).unsqueeze(0)
        logits, _ = self.model(context)
        predicted_label = torch.argmax(logits, dim=-1).item()
        return self.intents_inv[predicted_label]
