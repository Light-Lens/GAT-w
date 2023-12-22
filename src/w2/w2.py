from colorama import Fore, Style, init
from torch import nn

from src.utils import sent_tokenize
from src.models import LSTM

import random, numpy, torch

# Initialize colorama
init(autoreset = True)

class w2:
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
        self.patience = 100

        self.savepath = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def predict(self, model, character, temperature=1.0):
        character = numpy.array([[self.char2int[c] for c in character]])
        character = torch.from_numpy(character)
        character = character.to(self.device)
        
        out, hidden = model(character)

        # Adjust the output probabilities with temperature
        prob = nn.functional.softmax(out[-1] / temperature, dim=0).data
        # Sample from the modified distribution
        char_ind = torch.multinomial(prob, 1).item()

        return self.int2char[char_ind], hidden

    def generate(self, model, out_len, start, temperature=1.0):
        model.eval() # eval mode
        start = start.lower()
        # First off, run through the starting characters
        chars = list(start)
        size = out_len - len(chars)
        # Now pass in the previous characters and get a new one
        for _ in range(size):
            char, h = self.predict(model, chars, temperature)
            chars.append(char)

        return "".join(chars)

    # Preprocess data
    def preprocess(self, path: str):
        print(f"{Fore.YELLOW}{Style.BRIGHT}1.", "Loading data..")
        with open(path, "r", encoding="utf-8") as f:
            text = f.read()

        print(f"{Fore.YELLOW}{Style.BRIGHT}2.", "Preprocessing..")
        sentences = sent_tokenize(text)
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
        print(f"{Fore.YELLOW}{Style.BRIGHT}3.", "Preparing model to train..")

        # Instantiate the model with hyperparameters
        model = LSTM(
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
        print(f"{Fore.YELLOW}{Style.BRIGHT}4.", "Training..")
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

        print(f"{Fore.YELLOW}{Style.BRIGHT}5.", "Training complete..")

        if self.savepath != None:
            print(f"{Fore.YELLOW}{Style.BRIGHT}6.", "Saving model..")

            # Define save data dict
            savedata = {
                "model_state": model.state_dict(),
                "input_size": self.dict_size,
                "hidden_dim": self.hidden_dim,
                "embedding_dim": self.embedding_dim,
                "n_layers": self.n_layers,
                "dropout": self.dropout,
                "device": self.device
            }

            torch.save(savedata, self.savepath)
