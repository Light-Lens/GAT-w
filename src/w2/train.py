from colorama import Fore, Style, init
from torch import nn
from models import LSTM
import random, torch, numpy

# Initialize colorama
init(autoreset = True)

class TextDataLoader:
    def __init__(self, filepath):
        # Define dataset location
        self.filepath = filepath

        # Define input size and input/target sequence
        self.dict_size = None
        self.input_seq = None
        self.target_seq = None

    def preprocess_data(self, text):
        with open(self.filepath, "r", encoding="utf-8") as f:
            text = f.read()

        sentences = sent_tokenize(text)
        random.shuffle(sentences)

        # Join all the sentences together and extract the unique characters from the combined sentences
        chars = set("".join(sentences))

        # Creating a dictionary that maps integers to the characters
        int2char = dict(enumerate(chars))

        # Creating another dictionary that maps characters to integers
        char2int = {char: ind for ind, char in int2char.items()}

        # The length of the longest string
        maxlen = len(max(sentences, key=len))

        # A simple loop that loops through the list of sentences and adds a ' ' whitespace until the length of the sentence matches the length of the longest sentence
        for i in range(len(sentences)):
            while len(sentences[i]) < maxlen:
                sentences[i] += " "

        # Creating lists that will hold our input and target sequences
        input_seq = []
        target_seq = []

        for i in range(len(sentences)):
            # Remove last character for input sequence
            input_seq.append(sentences[i][:-1])

            # Remove first character for target sequence
            target_seq.append(sentences[i][1:])

        for i in range(len(sentences)):
            input_seq[i] = [char2int[character] for character in input_seq[i]]
            target_seq[i] = [char2int[character] for character in target_seq[i]]

        # dict_size = len(char2int)
        # seq_len = maxlen - 1
        # batch_size = len(sentences)

        self.dict_size = len(char2int)
        self.input_seq = input_seq
        self.target_seq = target_seq

class Train:
    def __init__(self, input_size, output_size, n_epochs, seq_len, batch_size, hidden_dim, embedding_dim, n_layers, lr):
        self.input_size = input_size
        self.output_size = output_size

        # Define hyperparameters
        self.n_epochs = n_epochs
        self.seq_len = seq_len
        self.batch_size = batch_size
        self.hidden_dim = hidden_dim
        self.embedding_dim = embedding_dim
        self.n_layers = n_layers
        self.learning_rate = lr

        self.dropout = 0
        self.patience = 100
        self.savepath = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Define save data dict
        self.savedata = {
            "model_state": None,
            "input_size": self.input_size,
            "output_size": self.output_size,
            "hidden_dim": self.hidden_dim,
            "embedding_dim": self.embedding_dim,
            "n_layers": self.n_layers,
            "dropout": self.dropout,
            "device": self.device
        }

    # Modify the one_hot_encode function to work with integer sequences
    def integer_encode(self, sequence, seq_len, batch_size):
        features = numpy.zeros((batch_size, seq_len), dtype=numpy.int64)
        for i in range(batch_size):
            for u in range(seq_len):
                features[i, u] = sequence[i][u]

        return features

    def train(self):
        # Instantiate the model with hyperparameters
        model = LSTM(
            input_size=self.input_size,
            output_size=self.output_size,
            hidden_dim=self.hidden_dim,
            n_layers=self.n_layers,
            embedding_dim=self.embedding_dim,
            dropout=self.dropout
        )
        model = model.to(self.device) # Set the model to the device that we defined earlier (default is CPU)

        # Convert input_seq to integer-encoded sequences
        input_seq_int = self.integer_encode(input_seq, self.seq_len, self.batch_size)
        input_seq_int = torch.from_numpy(input_seq_int)

        # Loss and optimizer
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=self.learning_rate)

        # Add early stopping
        best_loss = float('inf')

        # Train the model
        for epoch in range(1, self.n_epochs + 1):
            try:
                optimizer.zero_grad() # Clears existing gradients from previous epoch
                input_seq_int = input_seq_int.to(self.device)
                output, hidden = model(input_seq_int)
                output = output.to(self.device)
                target_seq = target_seq.to(self.device)
                loss = criterion(output, target_seq.view(-1).long())
                loss.backward() # Does backpropagation and calculates gradients
                optimizer.step() # Updates the weights accordingly

                print(f"Epoch [{epoch}/{self.n_epochs}], Loss: {loss.item():.4f}", end="\r")
                if epoch % (self.n_epochs/10) == 0:
                    print()

                # Check for early stopping
                if loss < best_loss:
                    best_loss = loss

                else:
                    self.patience -= 1
                    if self.patience == 0:
                        print(f"\n{Fore.RED}{Style.BRIGHT}Early stopping:", "No improvement in validation loss.\n")
                        break

            except KeyboardInterrupt:
                print()
                break

        if self.savepath != None:
            self.savedata["model_state"] = model.state_dict()
            torch.save(self.savedata, self.savepath)
