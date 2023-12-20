from torch import nn, torch

class Model(nn.Module):
    def __init__(self, input_size, output_size, hidden_dim, n_layers, embedding_dim):
        super(Model, self).__init__()

        self.device = None

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
        hidden = (torch.zeros(self.n_layers, batch_size, self.hidden_dim).to(self.device), torch.zeros(self.n_layers, batch_size, self.hidden_dim).to(self.device))
        return hidden
