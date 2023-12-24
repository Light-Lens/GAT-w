from torch import nn, torch
import numpy

# https://github.com/LaurentVeyssier/Text-generation-using-LSTM/blob/master/Chararacter-Level%20RNN%2C%20Exercise.ipynb
class LSTM(nn.Module):
    def __init__(self, tokens, output_size, hidden_dim, n_layers, embedding_dim, dropout=0):
        super(LSTM, self).__init__()
        # Defining device
        self.device = None

        # Defining some parameters
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers

        # Creating character dictionaries
        # Join all the sentences together and extract the unique characters from the combined sentences
        self.tokens = tokens

        # Creating a dictionary that maps integers to the characters
        self.int2char = dict(enumerate(self.tokens))

        # Creating another dictionary that maps characters to integers
        self.char2int = {char: ind for ind, char in self.int2char.items()}

        # Defining the layers
        self.embedding = nn.Embedding(len(self.tokens), embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, n_layers, batch_first=True, dropout=dropout) # LSTM layer
        self.dropout = nn.Dropout(dropout) # Dropout layer
        self.fc = nn.Linear(hidden_dim, output_size) # Fully connected layer

        # initialize the weights
        self.init_weights()

    # Forward pass through the network.
    # These inputs are x, and the hidden/cell state `hc`.
    def forward(self, x):
        embedded = self.embedding(x)
        batch_size = x.size(0)

        # Initialize hidden state for the first input using method defined below
        hidden = self.init_hidden(batch_size)

        # Passing in the input and hidden state into the model and obtaining outputs
        out, hidden = self.lstm(embedded, hidden)

        # Reshaping the outputs such that it can be fit into the fully connected layer
        out = out.contiguous().view(-1, self.hidden_dim)

        x = self.dropout(x)
        x = x.view(-1, self.hidden_dim)

        out = self.fc(out)

        return out, hidden

    def predict(self, character, temperature=1.0):
        character = numpy.array([[self.char2int[c] for c in character]])
        character = torch.from_numpy(character)
        character = character.to(self.device)

        out, hidden = self.forward(character)

        # Adjust the output probabilities with temperature
        prob = nn.functional.softmax(out[-1] / temperature, dim=0).data
        # Sample from the modified distribution
        char_ind = torch.multinomial(prob, 1).item()

        return self.int2char[char_ind], hidden

    # Initialize both hidden state and cell state
    # Create two new tensors with sizes n_layers x batch_size x hidden_dim,
    # initialized to zero, for hidden state and cell state of LSTM
    def init_hidden(self, batch_size):
        return (
            torch.zeros(self.n_layers, batch_size, self.hidden_dim).to(self.device),
            torch.zeros(self.n_layers, batch_size, self.hidden_dim).to(self.device)
        )

    # Initialize weights for fully connected layer.
    def init_weights(self):
        # Set bias tensor to all zeros
        self.fc.bias.data.fill_(0)
        # FC weights as random uniform
        self.fc.weight.data.uniform_(-1, 1)
