from colorama import Fore, Style, init
from src.utils import sent_tokenize
from src.models import LSTM
from torch import nn
import random, numpy, torch

# Initialize colorama
init(autoreset = True)

class train:
    def __init__(self, n_epochs, hidden_dim, embedding_dim, n_layers, lr, seq_len=None, batch_size=None):
        # Define hyperparameters
        self.n_epochs = n_epochs
        self.hidden_dim = hidden_dim
        self.embedding_dim = embedding_dim
        self.n_layers = n_layers
        self.learning_rate = lr
        self.seq_len = seq_len
        self.batch_size = batch_size

        self.dropout = 0
        self.patience = 500
        self.model_architecture = LSTM

        self.savepath = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Preprocess data
    def preprocess(self, path: str):
        print(f"{Fore.YELLOW}{Style.BRIGHT}Preprocessing text..")
        with open(path, "r", encoding="utf-8") as f:
            data = [i.strip() for i in f.readlines()]

        sentences = []
        for text in data:
            sentences.extend(sent_tokenize(text))

        random.shuffle(sentences)

        # Join all the sentences together and extract the unique characters from the combined sentences
        chars = set("".join(sentences))

        # Creating a dictionary that maps integers to the characters
        self.int2char = dict(enumerate(chars))

        # Creating another dictionary that maps characters to integers
        self.char2int = {char: ind for ind, char in self.int2char.items()}

        # The length of the longest string
        maxlen = len(max(sentences, key=len))

        # A simple loop that loops through the list of sentences and adds a ' ' whitespace until the length of the sentence matches the length of the longest sentence
        for i in range(len(sentences)):
            while len(sentences[i]) < maxlen:
                sentences[i] += " "

        # Creating lists that will hold our input and target sequences
        self.input_seq = []
        self.target_seq = []

        for i in range(len(sentences)):
            # Remove last character for input sequence
            self.input_seq.append(sentences[i][:-1])

            # Remove first character for target sequence
            self.target_seq.append(sentences[i][1:])

        for i in range(len(sentences)):
            self.input_seq[i] = [self.char2int[character] for character in self.input_seq[i]]
            self.target_seq[i] = [self.char2int[character] for character in self.target_seq[i]]

        self.input_seq = torch.LongTensor(self.input_seq).to(self.device)
        self.target_seq = torch.LongTensor(self.target_seq).to(self.device)

        self.dict_size = len(self.char2int)
        if self.seq_len == None:
            self.seq_len = maxlen - 1

        if self.batch_size == None:
            self.batch_size = len(sentences)

    # Modify the one_hot_encode function to work with integer sequences
    def integer_encode(self, sequence, seq_len, batch_size):
        features = numpy.zeros((batch_size, seq_len), dtype=numpy.int64)
        for i in range(batch_size):
            for u in range(seq_len):
                features[i, u] = sequence[i][u]

        return features

    def train(self):
        print(f"{Fore.YELLOW}{Style.BRIGHT}Input shape: {self.input_seq.shape} --> (Batch Size, Sequence Length)")

        # Instantiate the model with hyperparameters
        model = self.model_architecture(
            input_size=self.dict_size,
            output_size=self.dict_size,
            hidden_dim=self.hidden_dim,
            n_layers=self.n_layers,
            embedding_dim=self.embedding_dim,
            dropout=self.dropout
        )
        model = model.to(self.device) # Set the model to the device that we defined earlier (default is CPU)

        # Convert input_seq to integer-encoded sequences
        input_seq_int = self.integer_encode(self.input_seq, self.seq_len, self.batch_size)
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
                self.target_seq = self.target_seq.to(self.device)
                loss = criterion(output, self.target_seq.view(-1).long())
                loss.backward() # Does backpropagation and calculates gradients
                optimizer.step() # Updates the weights accordingly

                print(f"{Fore.WHITE}{Style.BRIGHT}Epoch [{epoch}/{self.n_epochs}], Loss: {loss.item():.4f}", end="\r")
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

        # Define save data dict
        model_data = {
            "model_state": model.state_dict(),
            "input_size": self.dict_size,
            "hidden_dim": self.hidden_dim,
            "embedding_dim": self.embedding_dim,
            "n_layers": self.n_layers,
            "dropout": self.dropout,
            "device": self.device,
            "int2char": self.int2char,
            "char2int": self.char2int,
            "model_architecture": self.model_architecture
        }

        if self.savepath != None:
            torch.save(model_data, self.savepath)

        return model_data
