from colorama import Fore, Style, init
from train import Model, generate
import torch

# Initialize colorama
init(autoreset = True)

data = torch.load("models\\model.pth")

model_state = data["model_state"]
dict_size = data["input_size"]
hidden_dim = data["hidden_dim"]
n_layers = data["n_layers"]
device = data["device"]

# Load the model
loaded_model = Model(input_size=dict_size, output_size=dict_size, hidden_dim=hidden_dim, n_layers=hidden_dim)
loaded_model.load_state_dict(model_state)
loaded_model = loaded_model.to(device)
loaded_model.eval()

# Example of using the loaded model for prediction
generated_text = generate(loaded_model, 40, "search what is a nuclear fusion")
print(f"{Fore.CYAN}{Style.BRIGHT}Generated text using loaded model:", generated_text)
