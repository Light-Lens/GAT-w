from torch.nn import functional as F
import torch.nn as nn, torch

class RNNConfig:
    n_hidden = 2
    n_layer = 1
    input_size = None
    output_size = None
    device = None

class RNN(nn.Module):
    def __init__(self):
        super(RNN, self).__init__()

        #Defining the layers
        self.rnn = nn.RNN(RNNConfig.input_size, RNNConfig.n_hidden, RNNConfig.n_layer, batch_first=True) # RNN Layer
        self.fc = nn.Linear(RNNConfig.n_hidden, RNNConfig.output_size) # Fully connected layer
    
    def forward(self, x, targets=None):
        # Apply RNN layer
        out, _ = self.rnn(x)

        # Squeeze the dimensions to remove the batch_first dimension
        out = out.squeeze(1)
        out = self.fc(out)

        # Calculate loss
        loss = None if targets is None else F.cross_entropy(out, targets)
        return out, loss

    def predict(self, x, classes):
        out, _ = self(x)
        _, predicted = torch.max(out, dim=1)

        tag = classes[predicted.item()]

        probs = torch.softmax(out, dim=1)
        prob = probs[0][predicted.item()]
        confidence = prob.item()

        return tag, confidence

# class FeedForwardConfig:
#     n_hidden = 2
#     n_layer = 1
#     input_size = None
#     output_size = None

# class FeedForward(nn.Module):
#     def __init__(self):
#         super(FeedForward, self).__init__()

#         # Create a list to hold the layers
#         self.layers = nn.ModuleList()

#         # Add input layer
#         self.layers.append(nn.Linear(FeedForwardConfig.input_size, FeedForwardConfig.n_hidden))
#         self.layers.append(nn.ReLU())

#         # Add hidden layers
#         for _ in range(FeedForwardConfig.n_layer - 1):
#             self.layers.append(nn.Linear(FeedForwardConfig.n_hidden, FeedForwardConfig.n_hidden))
#             self.layers.append(nn.ReLU())

#         # Add output layer
#         self.layers.append(nn.Linear(FeedForwardConfig.n_hidden, FeedForwardConfig.output_size))

#     def forward(self, x, targets=None):
#         out = x
#         for layer in self.layers:
#             out = layer(out)

#         # Apply softmax to obtain probabilities for each class
#         out = F.softmax(out, dim=-1)

#         if targets is None:
#             loss = None

#         else:
#             loss = F.cross_entropy(out, targets)

#         return out, loss

#     def predict(self, x, classes):
#         out, _ = self(x)
#         _, predicted = torch.max(out, dim=1)

#         tag = classes[predicted.item()]

#         probs = torch.softmax(out, dim=1)
#         prob = probs[0][predicted.item()]
#         confidence = prob.item()

#         return tag, confidence
