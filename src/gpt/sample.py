from .model import GPTConfig, GPT
import torch

class sample:
    def __init__(self, model_path):
        # Load the saved model
        model_data = torch.load(model_path)

        self.state_dict = model_data["state_dict"]
        GPTConfig.device = model_data["device"]
        GPTConfig.n_embd = model_data["config"]["n_embd"]
        GPTConfig.n_head = model_data["config"]["n_head"]
        GPTConfig.n_layer = model_data["config"]["n_layer"]
        GPTConfig.block_size = model_data["config"]["block_size"]
        GPTConfig.dropout = model_data["config"]["dropout"]
        GPTConfig.vocab_size = model_data["config"]["vocab_size"]

    def load(self):
        # Create an instance of GPT
        self.model = GPT()

        # Load the saved model state_dict
        self.model.load_state_dict(self.state_dict)
        self.model.to(GPTConfig.device)
        self.model.eval() # Set the model to evaluation mode

    # Use the model for generation or other tasks
    def generate(self, encoded_text=None, length=100, temperature=1.0, top_k=None):
        """
        `max_new_tokens`: number of tokens generated in each sample
        `temperature`: 1.0 = no change, < 1.0 = less random, > 1.0 = more random, in predictions
        `tok_k`: retain only the top_k most likely tokens, clamp others to have 0 probability
        """

        if encoded_text == None:
            context = torch.zeros((1, 1), dtype=torch.long, device=GPTConfig.device)

        else:
            context = torch.tensor(encoded_text, dtype=torch.long, device=GPTConfig.device).unsqueeze(0)

        return self.model.generate(context, max_new_tokens=length, temperature=temperature, top_k=top_k)[0].tolist()
