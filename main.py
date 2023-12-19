from colorama import Fore, Style, init
import numpy as np
import nltk

import torch
from torch import nn

# Initialize colorama
init(autoreset = True)

# Load dataset
def sent_tokenize(sentence):
    return nltk.sent_tokenize(sentence.strip())

with open("data\\data.txt", "r", encoding="utf-8") as f:
    lines = [i.strip() for i in f.readlines()]
    text = []

    for line in lines:
        text.extend(sent_tokenize(line))

# Join all the sentences together and extract the unique characters from the combined sentences
chars = set(''.join(text))

# Creating a dictionary that maps integers to the characters
int2char = dict(enumerate(chars))

# Creating another dictionary that maps characters to integers
char2int = {char: ind for ind, char in int2char.items()}

# The length of the longest string
maxlen = len(max(text, key=len))

# A simple loop that loops through the list of sentences and adds a ' ' whitespace until the length of the sentence matches the length of the longest sentence
for i in range(len(text)):
    while len(text[i])<maxlen:
        text[i] += ' '

# Creating lists that will hold our input and target sequences
input_seq = []
target_seq = []

for i in range(len(text)):
    # Remove last character for input sequence
    input_seq.append(text[i][:-1])

    # Remove first character for target sequence
    target_seq.append(text[i][1:])

for i in range(len(text)):
    input_seq[i] = [char2int[character] for character in input_seq[i]]
    target_seq[i] = [char2int[character] for character in target_seq[i]]

dict_size = len(char2int)
seq_len = maxlen - 1
batch_size = len(text)

# torch.cuda.is_available() checks and returns True if a GPU is available, else return False
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def one_hot_encode(sequence, dict_size, seq_len, batch_size):
    return torch.LongTensor(sequence).to(device)

input_seq = one_hot_encode(input_seq, dict_size, seq_len, batch_size)
print(f"{Fore.YELLOW}{Style.BRIGHT}Input shape: {input_seq.shape} --> (Batch Size, Sequence Length, One-Hot Encoding Size)")

# input_seq = torch.from_numpy(input_seq)
target_seq = torch.Tensor(target_seq)

class Model(nn.Module):
    def __init__(self, input_size, output_size, hidden_dim, n_layers):
        super(Model, self).__init__()

        # Defining some parameters
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers

        # Defining the layers
        self.embedding = nn.Embedding(input_size, hidden_dim)
        self.lstm = nn.LSTM(hidden_dim, hidden_dim, n_layers, batch_first=True)
        # self.rnn = nn.RNN(input_size, hidden_dim, n_layers, batch_first=True) # RNN Layer
        self.fc = nn.Linear(hidden_dim, output_size) # Fully connected layer
    
    def forward(self, x):
        batch_size = x.size(0)

        # Pass input through the embedding layer
        x = self.embedding(x)

        # Initialize hidden state for the first input using method defined below
        hidden = self.init_hidden(batch_size)

        # Passing in the input and hidden state into the model and obtaining outputs
        out, hidden = self.lstm(x, hidden)

        # Reshaping the outputs such that it can be fit into the fully connected layer
        out = out.contiguous().view(-1, self.hidden_dim)
        out = self.fc(out)

        return out, hidden

    def init_hidden(self, batch_size):
        # Initialize both hidden state and cell state
        hidden = (torch.zeros(self.n_layers, batch_size, self.hidden_dim).to(device), torch.zeros(self.n_layers, batch_size, self.hidden_dim).to(device))
        return hidden


def predict(model, character, temperature=1.0):
    # One-hot encoding our input to fit into the model
    character = np.array([[char2int[c] for c in character]])
    # character = one_hot_encode(character, dict_size, character.shape[1], 1)
    character = torch.from_numpy(character)
    character = character.to(device)
    
    out, hidden = model(character)

    # prob = nn.functional.softmax(out[-1], dim=0).data
    # # Taking the class with the highest probability score from the output
    # char_ind = torch.max(prob, dim=0)[1].item()

    # Adjust the output probabilities with temperature
    prob = nn.functional.softmax(out[-1] / temperature, dim=0).data

    # Sample from the modified distribution
    char_ind = torch.multinomial(prob, 1).item()


    return int2char[char_ind], hidden

def generate(model, out_len, start='hey', temperature=1.0):
    model.eval() # eval mode
    start = start.lower()
    # First off, run through the starting characters
    chars = list(start)
    size = out_len - len(chars)
    # Now pass in the previous characters and get a new one
    for _ in range(size):
        char, h = predict(model, chars, temperature)
        chars.append(char)

    return ''.join(chars)

# Define hyperparameters
n_epochs = 1000
hidden_dim = 8
n_layers = 1
lr = 0.01

# Instantiate the model with hyperparameters
model = Model(input_size=dict_size, output_size=dict_size, hidden_dim=hidden_dim, n_layers=n_layers)
model = model.to(device) # Set the model to the device that we defined earlier (default is CPU)

# Define Loss, Optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=lr)

# Save data details for the trained model
data = {
    "model_state": None,
    "input_size": dict_size,
    "hidden_dim": hidden_dim,
    "n_layers": n_layers,
    "device": device
}

# Training Run
for epoch in range(1, n_epochs + 1):
    try:
        optimizer.zero_grad() # Clears existing gradients from previous epoch
        input_seq = input_seq.to(device)
        output, hidden = model(input_seq)
        output = output.to(device)
        target_seq = target_seq.to(device)
        loss = criterion(output, target_seq.view(-1).long())
        loss.backward() # Does backpropagation and calculates gradients
        optimizer.step() # Updates the weights accordingly

        print(f'{Fore.WHITE}{Style.BRIGHT}Epoch [{epoch}/{n_epochs}], Loss: {loss.item():.4f}', end="\r")
        if epoch % (n_epochs/10) == 0:
            # print(f'{Fore.WHITE}{Style.BRIGHT}Epoch [{epoch}/{n_epochs}], Loss: {loss.item():.4f}')

            # Save the model checkpoint
            # data["model_state"] = model.state_dict()
            # torch.save(data, f"models\\mid_epoch\\model_{epoch}.pth")
            # print(f"{Fore.YELLOW}{Style.BRIGHT}Model checkpoint saved: models\\mid_epoch\\model_{epoch}.pth")
            print()

    except KeyboardInterrupt:
        print()
        break

with torch.no_grad():
    val_output, _ = model(input_seq)
    val_loss = criterion(val_output, target_seq.view(-1).long())
    val_perplexity = torch.exp(val_loss)
    print(f'Validation Perplexity: {val_perplexity.item():.4f}')

    # data["model_state"] = model.state_dict()
    # torch.save(data, "models\\model.pth")
    # print(f"{Fore.GREEN}{Style.BRIGHT}Final trained model saved!")

# data = torch.load("models\\model.pth")

# model_state = data["model_state"]
# dict_size = data["input_size"]
# hidden_dim = data["hidden_dim"]
# n_layers = data["n_layers"]
# device = data["device"]

# # Load the model
# loaded_model = Model(input_size=dict_size, output_size=dict_size, hidden_dim=hidden_dim, n_layers=n_layers)
# loaded_model.load_state_dict(model_state)
# loaded_model = loaded_model.to(device)
# loaded_model.eval()

text = "Me: Hi there! How are you? GAT-w"
print(f"{Fore.GREEN}{Style.BRIGHT}Input text:", text)
print(f"{Fore.CYAN}{Style.BRIGHT}Generated text:", generate(model, 40, text))

# print(f"{Fore.GREEN}{Style.BRIGHT}Input text:", "search what is a nuclear fusion")
# print(f"{Fore.CYAN}{Style.BRIGHT}Generated text:", generate(model, 200, "search what is a nuclear fusion"))
