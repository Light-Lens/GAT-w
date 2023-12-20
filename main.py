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

# Join all the sentences together and extract the unique words from the combined sentences
words = set(' '.join(text).split())

# Creating a dictionary that maps integers to the words
word2int = {'<UNK>': 0}  # Start with an unknown token
word2int.update({word: idx + 1 for idx, word in enumerate(words)})

# Creating another dictionary that maps words to integers
int2word = {i: w for w, i in word2int.items()}

# The length of the longest sentence
maxlen = len(max(text, key=lambda x: len(x.split())))

# A simple loop that loops through the list of sentences and pads with '<PAD>' until the length of the sentence matches the length of the longest sentence
for i in range(len(text)):
    while len(text[i].split()) < maxlen:
        text[i] += ' <PAD>'

# Creating lists that will hold our input and target sequences
input_seq = []
target_seq = []

for i in range(len(text)):
    # Remove last word for input sequence
    input_seq.append(text[i].split()[:-1])

    # Remove first word for target sequence
    target_seq.append(text[i].split()[1:])

# Convert words to indices
for i in range(len(text)):
    input_seq[i] = [word2int.get(word, word2int['<UNK>']) for word in input_seq[i]]
    target_seq[i] = [word2int.get(word, word2int['<UNK>']) for word in target_seq[i]]

dict_size = len(word2int)
seq_len = maxlen - 1
batch_size = len(text)

# torch.cuda.is_available() checks and returns True if a GPU is available, else return False
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

input_seq = torch.LongTensor(input_seq).to(device)
target_seq = torch.LongTensor(target_seq).to(device)

class Model(nn.Module):
    def __init__(self, input_size, output_size, hidden_dim, n_layers, embedding_dim):
        super(Model, self).__init__()

        # Defining some parameters
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers

        # Defining the layers
        self.embedding = nn.Embedding(input_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, n_layers, batch_first=True) # LSTM layer
        self.fc = nn.Linear(hidden_dim, output_size) # Fully connected layer

    def forward(self, x):
        embedded = self.embedding(x)
        batch_size = x.size(0)

        # Initialize hidden state for the first input using method defined below
        hidden = self.init_hidden(batch_size)

        # Passing in the input and hidden state into the model and obtaining outputs
        out, hidden = self.lstm(embedded, hidden)

        # Reshaping the outputs such that it can be fit into the fully connected layer
        out = out.contiguous().view(-1, self.hidden_dim)
        out = self.fc(out)

        return out, hidden

    def init_hidden(self, batch_size):
        # Initialize both hidden state and cell state
        hidden = (torch.zeros(self.n_layers, batch_size, self.hidden_dim).to(device), torch.zeros(self.n_layers, batch_size, self.hidden_dim).to(device))
        return hidden

def predict(model, words, temperature=1.0):
    indices = [word2int.get(word, word2int['<UNK>']) for word in words]
    indices = torch.tensor(indices).unsqueeze(0).to(device)

    out, hidden = model(indices)

    # Adjust the output probabilities with temperature
    prob = nn.functional.softmax(out[-1] / temperature, dim=0).data
    # Sample from the modified distribution
    word_ind = torch.multinomial(prob, 1).item()

    return int2word[word_ind], hidden

def generate(model, out_len, start, temperature=1.0):
    model.eval()  # eval mode
    start = start.lower()
    # First off, run through the starting words
    words = start.split()
    size = out_len - len(words)
    # Now pass in the previous words and get a new one
    for _ in range(size):
        word, h = predict(model, words, temperature)
        words.append(word)

    return " ".join(words)

# Define hyperparameters
n_epochs = 5000
hidden_dim = 16
embedding_dim = 32
n_layers = 2
lr = 0.01
patience = 3000 # Adjust patience as needed

# Instantiate the model with hyperparameters
model = Model(input_size=dict_size, output_size=dict_size, hidden_dim=hidden_dim, n_layers=n_layers, embedding_dim=embedding_dim)
model = model.to(device) # Set the model to the device that we defined earlier (default is CPU)

# Modify the one_hot_encode function to work with integer sequences
def integer_encode(sequence, seq_len, batch_size):
    features = np.zeros((batch_size, seq_len), dtype=np.int64)
    for i in range(batch_size):
        for u in range(seq_len):
            features[i, u] = sequence[i][u]

    return features

# Convert input_seq to integer-encoded sequences
input_seq_int = integer_encode(input_seq, seq_len, batch_size)
input_seq_int = torch.from_numpy(input_seq_int)

# Define Loss, Optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=lr)

# Save data details for the trained model
data = {
    "model_state": None,
    "input_size": dict_size,
    "hidden_dim": hidden_dim,
    "embedding_dim": embedding_dim,
    "n_layers": n_layers,
    "device": device
}

# Training Run
# Add early stopping
best_loss = float('inf')

for epoch in range(1, n_epochs + 1):
    try:
        optimizer.zero_grad() # Clears existing gradients from previous epoch
        input_seq_int = input_seq_int.to(device)
        output, hidden = model(input_seq_int)
        output = output.to(device)
        target_seq = target_seq.to(device)
        loss = criterion(output, target_seq.view(-1).long())
        loss.backward() # Does backpropagation and calculates gradients
        optimizer.step() # Updates the weights accordingly

        print(f'{Fore.WHITE}{Style.BRIGHT}Epoch [{epoch}/{n_epochs}], Loss: {loss.item():.4f}', end="\r")
        if epoch % (n_epochs/10) == 0:
            # Save the model checkpoint
            # data["model_state"] = model.state_dict()
            # torch.save(data, f"models\\mid_epoch\\model_{epoch}.pth")
            # print(f"\n{Fore.YELLOW}{Style.BRIGHT}Model checkpoint saved: models\\mid_epoch\\model_{epoch}.pth")
            print()

        # Check for early stopping
        if loss < best_loss:
            best_loss = loss
            # data["model_state"] = model.state_dict()
            # torch.save(data, f"models\\mid_epoch\\model_{epoch}.pth")

        else:
            patience -= 1
            if patience == 0:
                print(f"\n{Fore.RED}{Style.BRIGHT}Early stopping:", "No improvement in validation loss.\n")
                break

    except KeyboardInterrupt:
        print()
        break

    # data["model_state"] = model.state_dict()
    # torch.save(data, "models\\model.pth")
    # print(f"{Fore.GREEN}{Style.BRIGHT}Final trained model saved!")

# data = torch.load("models\\model.pth")

# model_state = data["model_state"]
# dict_size = data["input_size"]
# hidden_dim = data["hidden_dim"]
# embedding_dim = data["embedding_dim"]
# n_layers = data["n_layers"]
# device = data["device"]

# Load the model
# loaded_model = Model(input_size=dict_size, output_size=dict_size, hidden_dim=hidden_dim, n_layers=n_layers, embedding_dim=embedding_dim)
# loaded_model.load_state_dict(model_state)
# loaded_model = loaded_model.to(device)
# loaded_model.eval()

text = [
    "search about what is a nuclear fusion",
    "search about how a search engine works",
    "search about what is a search engine",
    "open spotify for me please",
    "open chrome for me please",
    "please open microsoft edge"
]

for i in text:
    print(f"{Fore.GREEN}{Style.BRIGHT}Input text:", i)
    print(f"{Fore.CYAN}{Style.BRIGHT}Generated text:", generate(model, 200, i))
    print()
