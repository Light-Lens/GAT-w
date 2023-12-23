from torch import nn, torch

class LSTM(nn.Module):
    def __init__(self, input_size, output_size, hidden_dim, n_layers, embedding_dim, dropout=0):
        super(LSTM, self).__init__()

        # Defining device
        self.device = None

        # Defining some parameters
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers

        # Defining the layers
        self.embedding = nn.Embedding(input_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, n_layers, batch_first=True, dropout=dropout) # LSTM layer
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

    # Initialize both hidden state and cell state
    def init_hidden(self, batch_size):
        hidden = (torch.zeros(self.n_layers, batch_size, self.hidden_dim).to(self.device), torch.zeros(self.n_layers, batch_size, self.hidden_dim).to(self.device))
        return hidden

class Transformer(nn.Module):
    def __init__(self, input_size, output_size, hidden_dim, n_head, n_layers, dropout):
        super(Transformer, self).__init__()

        # Defining the layers
        self.embedding = nn.Embedding(input_size, hidden_dim)
        self.transformer = nn.Transformer( d_model=hidden_dim, nhead=n_head, num_encoder_layers=n_layers, num_decoder_layers=n_layers, dropout=dropout) # Transformer layer
        self.fc = nn.Linear(hidden_dim, output_size) # Fully connected layer

    def forward(self, x):
        embedded = self.embedding(x)
        transformer_out = self.transformer(embedded, embedded)
        output = self.fc(transformer_out[-1, :, :])  # Use the last time step's output
        return output
