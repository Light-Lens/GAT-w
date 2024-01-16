from src.text.generation.utils import dprint, encode, decode
from src.text.generation.model import GPT, set_params
import torch

# Load teh saved model
model_data = torch.load("models\\GAT-w2.pth")

state_dict = model_data["state_dict"]
stoi = model_data["stoi"]
itos = model_data["itos"]
device = model_data["device"]
n_embd = model_data["config"]["n_embd"]
n_head = model_data["config"]["n_head"]
n_layer = model_data["config"]["n_layer"]
block_size = model_data["config"]["block_size"]
dropout = model_data["config"]["dropout"]
vocab_size = model_data["config"]["vocab_size"]

# Set parameters and Create an instance of GPT
set_params(_n_embd=n_embd, _n_head=n_head, _n_layer=n_layer, _block_size=block_size, _dropout=dropout, _vocab_size=vocab_size, _device=device)
model = GPT()

# Load the saved model state_dict
model.load_state_dict(state_dict)
model.to(device)
model.eval()  # Set the model to evaluation mode

# Use the model for generation or other tasks
# For example:
context = torch.tensor(encode("Hi!", stoi=stoi), dtype=torch.long, device=device).unsqueeze(0)
output = decode(model.generate(context, max_new_tokens=100)[0].tolist(), itos=itos)

print("-" * 100)
print("input:", "Hi!")
print("output:\n")
dprint(output)
