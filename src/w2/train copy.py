from colorama import Fore, Style, init
from src.models import LSTM
from torch import nn
import numpy, torch

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
            sentences = [i.strip() for i in f.readlines()]

        # random.shuffle(sentences)
        # If sequence length is None then set sequence length as the length of the longest string
        if self.seq_len == None:
            self.seq_len = len(max(sentences, key=len)) - 1

        else:
            temp_sentences = []
            for i in sentences:
                for j in range(0, len(i), self.seq_len):
                    temp_sentences.append(i[j:j+self.seq_len])

            sentences = temp_sentences

        # https://statisticsglobe.com/add-string-each-element-list-python#:~:text=In%20this%20example%2C%20we%20will,the%20elements%20in%20a%20list.&text=As%20you%20can%20see%20in,of%20fruit%3A%20with%20the%20element.
        # A simple loop that loops through the list of sentences and adds a ' ' whitespace until the length of the sentence matches the length of the longest sentence
        # The lambda function takes an element as input and returns the concatenation of " " with the element.
        # The map function returns a map object that is converted to a list using the list() function.
        sentences = list(map(lambda x: x + " " * (self.seq_len - len(x)), sentences))

        # Calculate the number of lines for training
        num_train_lines = int(len(sentences) * data_division)

        # Split the lines into training and testing sets
        train_data = sentences[:num_train_lines]
        test_data = sentences[num_train_lines:]

        print(f"{Fore.YELLOW}{Style.BRIGHT}Data division: {len(train_data), len(test_data)} --> (Train Data Length, Test Data Length)")

        self.int2char, self.char2int, self.train_input_seq, self.train_target_seq, self.dict_size = self.preprocess_data(train_data)
        _, _, self.test_input_seq, self.test_target_seq, _ = self.preprocess_data(test_data)

        print(f"{Fore.YELLOW}{Style.BRIGHT}Train Input shape: {self.train_input_seq.shape} --> (Batch Size, Sequence Length)")
        print(f"{Fore.YELLOW}{Style.BRIGHT}Test Input shape: {self.test_input_seq.shape} --> (Batch Size, Sequence Length)")
        print(f"{Fore.YELLOW}{Style.BRIGHT}Vocab size: {self.dict_size, len(self.char2int)} -> (Train Vocab Size, Test Vocab Size)")

    def preprocess_data(self, data):
        # Join all the sentences together and extract the unique characters from the combined sentences
        chars = set("".join(data))

        # Creating a dictionary that maps integers to the characters
        int2char = dict(enumerate(chars))

        # Creating another dictionary that maps characters to integers
        char2int = {char: ind for ind, char in int2char.items()}

        # Creating lists that will hold our input and target sequences
        input_seq = []
        target_seq = []

        # Automatically set batch size.
        if self.batch_size == None:
            self.batch_size = len(data)

        for i in range(self.batch_size):
            # Remove last character for input sequence
            input_seq.append(data[i][:self.seq_len-1])

            # Remove first character for target sequence
            target_seq.append(data[i][1:self.seq_len])

        for i in range(self.batch_size):
            input_seq[i] = [char2int[character] for character in input_seq[i]]
            target_seq[i] = [char2int[character] for character in target_seq[i]]

        input_seq = torch.LongTensor(input_seq).to(self.device)
        target_seq = torch.LongTensor(target_seq).to(self.device)

        dict_size = len(char2int)

        return int2char, char2int, input_seq, target_seq, dict_size

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
        #! BUG BUG BUG FUCK THIS PIECE OF SHIT.
        features = numpy.zeros((self.batch_size, self.seq_len), dtype=numpy.int64)
        for i in range(self.batch_size):
            for u in range(self.seq_len):
                features[i, u] = self.train_input_seq[i][u]

        train_input_seq_int = features
        train_input_seq_int = torch.from_numpy(train_input_seq_int)

        # Loss and optimizer
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=self.learning_rate)

        # Add early stopping
        best_loss = float('inf')

        # Train the model
        for epoch in range(1, self.n_epochs + 1):
            try:
                optimizer.zero_grad() # Clears existing gradients from previous epoch
                train_input_seq_int = train_input_seq_int.to(self.device)
                output, hidden = model(train_input_seq_int)
                output = output.to(self.device)
                self.train_target_seq = self.train_target_seq.to(self.device)
                loss = criterion(output, self.train_target_seq.view(-1).long())
                loss.backward() # Does backpropagation and calculates gradients
                optimizer.step() # Updates the weights accordingly

                # Validation loss on the test set
                if self.test_input_seq.shape != torch.Size([0]):
                    model.eval()
                    with torch.no_grad():
                        test_output, _ = model(self.test_input_seq)
                        test_loss = criterion(test_output, self.test_target_seq.view(-1).long())
                    model.train()

                else:
                    test_loss = loss

                print(f"{Fore.WHITE}{Style.BRIGHT}Epoch [{epoch}/{self.n_epochs}], Train Loss: {loss.item():.4f}, Val loss: {test_loss.item():.4f}", end="\r")
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
