from colorama import Fore, Style, init
from src.utils import sent_tokenize
from src.models import LSTM
from torch import nn
import random, numpy, torch

# Initialize colorama
init(autoreset = True)

class train:
    def __init__(self, n_epochs, hidden_dim, embedding_dim, n_layers, lr, batch_size=None, seq_len=None, dropout=0, patience=100):
        # Define hyperparameters
        self.n_epochs = n_epochs
        self.hidden_dim = hidden_dim
        self.embedding_dim = embedding_dim
        self.n_layers = n_layers
        self.learning_rate = lr
        self.batch_size = batch_size
        self.seq_len = seq_len

        self.dropout = dropout
        self.patience = patience
        self.model_architecture = LSTM

        self.savepath = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Preprocess data
    def preprocess(self, path: str, data_division=0.7):
        print(f"{Fore.YELLOW}{Style.BRIGHT}Preprocessing text..")
        with open(path, "r", encoding="utf-8") as f:
            data = [i.strip() for i in f.readlines()]

        sentences = []
        for text in data:
            sentences.extend(sent_tokenize(text))

        random.shuffle(sentences)

        # Calculate the number of lines for training
        num_train_lines = int(len(sentences) * data_division)

        # Split the lines into training and testing sets
        train_data = sentences[:num_train_lines]
        test_data = sentences[num_train_lines:]

        print(f"{Fore.YELLOW}{Style.BRIGHT}Data division: {len(train_data), len(test_data)} --> (Train Data Length, Test Data Length)")

        self.preprocess_train_data(train_data)
        self.preprocess_test_data(test_data)

        print(f"{Fore.YELLOW}{Style.BRIGHT}Train Input shape: {self.train_input_seq.shape} --> (Batch Size, Sequence Length)")
        print(f"{Fore.YELLOW}{Style.BRIGHT}Test Input shape: {self.test_input_seq.shape} --> (Batch Size, Sequence Length)")

    def preprocess_train_data(self, train_data):
        # Join all the sentences together and extract the unique characters from the combined sentences
        chars = set("".join(train_data))

        # Creating a dictionary that maps integers to the characters
        self.int2char = dict(enumerate(chars))

        # Creating another dictionary that maps characters to integers
        self.char2int = {char: ind for ind, char in self.int2char.items()}

        # If sequence length is None then, set sequence length as the length of the longest string
        longest_str_len = len(max(train_data, key=len))
        if self.seq_len == None:
            maxlen = longest_str_len
            self.seq_len = maxlen - 1

        print(f"{Fore.YELLOW}{Style.BRIGHT}Longest Train String Length: {longest_str_len}")

        # A simple loop that loops through the list of sentences and adds a ' ' whitespace until the length of the sentence matches the length of the longest sentence
        for i in range(len(train_data)):
            while len(train_data[i]) < maxlen:
                train_data[i] += " "

        # Creating lists that will hold our input and target sequences
        self.train_input_seq = []
        self.train_target_seq = []

        # Automatically set batch size.
        if self.batch_size == None:
            self.batch_size = len(train_data)

        for i in range(self.batch_size):
            # Remove last character for input sequence
            self.train_input_seq.append(train_data[i][:maxlen-1])

            # Remove first character for target sequence
            self.train_target_seq.append(train_data[i][1:maxlen])

        for i in range(self.batch_size):
            self.train_input_seq[i] = [self.char2int[character] for character in self.train_input_seq[i]]
            self.train_target_seq[i] = [self.char2int[character] for character in self.train_target_seq[i]]

        self.train_input_seq = torch.LongTensor(self.train_input_seq).to(self.device)
        self.train_target_seq = torch.LongTensor(self.train_target_seq).to(self.device)

        self.dict_size = len(self.char2int)

        print(f"{Fore.YELLOW}{Style.BRIGHT}Vocab size: ({self.dict_size}, ", end="")

    def preprocess_test_data(self, test_data):
        # Join all the sentences together and extract the unique characters from the combined sentences
        chars = set("".join(test_data))

        # Creating a dictionary that maps integers to the characters
        int2char = dict(enumerate(chars))

        # Creating another dictionary that maps characters to integers
        char2int = {char: ind for ind, char in int2char.items()}

        # The length of the longest string
        maxlen = self.seq_len

        # A simple loop that loops through the list of sentences and adds a ' ' whitespace until the length of the sentence matches the length of the longest sentence
        for i in range(len(test_data)):
            while len(test_data[i]) < maxlen:
                test_data[i] += " "

        # Creating lists that will hold our input and target sequences
        self.test_input_seq = []
        self.test_target_seq = []

        for i in range(self.batch_size):
            # Remove last character for input sequence
            self.test_input_seq.append(test_data[i][:maxlen-1])

            # Remove first character for target sequence
            self.test_target_seq.append(test_data[i][1:maxlen])

        for i in range(self.batch_size):
            self.test_input_seq[i] = [char2int[character] for character in self.test_input_seq[i]]
            self.test_target_seq[i] = [char2int[character] for character in self.test_target_seq[i]]

        self.test_input_seq = torch.LongTensor(self.test_input_seq).to(self.device)
        self.test_target_seq = torch.LongTensor(self.test_target_seq).to(self.device)

        print(f"{Fore.YELLOW}{Style.BRIGHT}{len(char2int)}) -> (Train Vocab Size, Test Vocab Size)")

    # Modify the one_hot_encode function to work with integer sequences
    def integer_encode(self, sequence, seq_len, batch_size):
        features = numpy.zeros((batch_size, seq_len), dtype=numpy.int64)
        for i in range(batch_size):
            for u in range(seq_len):
                features[i, u] = sequence[i][u]

        return features

    def train(self):
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
        input_seq_int = self.integer_encode(self.train_input_seq, self.seq_len, self.batch_size)
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
                self.train_target_seq = self.train_target_seq.to(self.device)
                loss = criterion(output, self.train_target_seq.view(-1).long())
                loss.backward() # Does backpropagation and calculates gradients
                optimizer.step() # Updates the weights accordingly

                # Validation loss on the test set
                model.eval()
                with torch.no_grad():
                    test_output, _ = model(self.test_input_seq)
                    test_loss = criterion(test_output, self.test_target_seq.view(-1).long())
                model.train()

                print(f"{Fore.WHITE}{Style.BRIGHT}Epoch [{epoch}/{self.n_epochs}], Loss: {loss.item():.4f}", end="\r")
                if epoch % (self.n_epochs/10) == 0:
                    print()

                # Check for early stopping
                if test_loss < best_loss:
                    best_loss = test_loss

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
