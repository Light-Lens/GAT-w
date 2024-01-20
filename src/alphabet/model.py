from torch.nn import functional as F
import torch.nn as nn
import torch

class LSTMConfig:
    n_embd = 8
    n_hidden = 2
    n_layer = 2
    dropout = 0
    vocab_size = None
    output_size = None
    device = None

class LSTM(nn.Module):
    def __init__(self):
        super(LSTM, self).__init__()

        # Defining the layers
        self.embedding = nn.Embedding(LSTMConfig.vocab_size, LSTMConfig.n_embd)
        self.lstm = nn.LSTM(LSTMConfig.n_embd, LSTMConfig.n_hidden, LSTMConfig.n_layer, batch_first=True) # LSTM layer
        self.dropout = nn.Dropout(LSTMConfig.dropout) # Dropout layer
        self.fc = nn.Linear(LSTMConfig.n_hidden, LSTMConfig.output_size) # Fully connected layer

    def forward(self, x, targets=None):
        embedded = self.embedding(x)
        batch_size = x.size(0)

        # Initialize hidden state for the first input using method defined below
        hidden = self._init_hidden(batch_size)

        # Passing in the input and hidden state into the model and obtaining outputs
        out, hidden = self.lstm(embedded, hidden)

        # Reshaping the outputs such that it can be fit into the fully connected layer
        out = out.contiguous().view(-1, LSTMConfig.n_hidden)
        out = self.fc(out)

        if targets is None:
            loss = None

        else:
            targets = targets.unsqueeze(2)
            targets = targets.expand(-1, -1, LSTMConfig.n_hidden)
            targets = targets.view(-1, LSTMConfig.n_hidden).float()
            loss = F.cross_entropy(out, targets)

        return out, hidden, loss

    def _init_hidden(self, batch_size):
        # Initialize both hidden state and cell state
        return (
            torch.zeros(LSTMConfig.n_layer, batch_size, LSTMConfig.n_hidden).to(LSTMConfig.device),
            torch.zeros(LSTMConfig.n_layer, batch_size, LSTMConfig.n_hidden).to(LSTMConfig.device)
        )
