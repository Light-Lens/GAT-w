from colorama import Fore, Style, init
from src.utils import sent_tokenize
from src.models import LSTM
from torch import nn
import random, numpy, torch

# Initialize colorama
init(autoreset = True)

class Train:
    def __init__(self, n_epochs, hidden_dim, embedding_dim, n_layers, lr, seq_len, batch_size=None, clip_value=1, dropout=0, patience=100):
        # Define hyperparameters
        self.n_epochs = n_epochs
        self.hidden_dim = hidden_dim
        self.embedding_dim = embedding_dim
        self.n_layers = n_layers
        self.learning_rate = lr

        self.batch_size = batch_size
        self.seq_len = seq_len

        self.clip = clip_value
        self.dropout = dropout
        self.patience = patience
        self.model_architecture = LSTM

        self.savepath = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def preprocess(self, path, data_division=0.7):
        print(f"{Fore.YELLOW}{Style.BRIGHT}Preprocessing text..")
        with open(path, "r", encoding="utf-8") as f:
            text = f.read()

        # Encode the text and map each character to an integer and vice versa
        # We create two dictonaries:
        # 1. int2char, which maps integers to characters
        # 2. char2int, which maps characters to unique integers

        # Join all the sentences together and extract the unique characters from the combined sentences
        self.chars = set(text)

        # Creating a dictionary that maps integers to the characters
        self.int2char = dict(enumerate(self.chars))

        # Creating another dictionary that maps characters to integers
        self.char2int = {char: ind for ind, char in self.int2char.items()}

        sentences = []
        for i in sentences:
            for j in range(0, len(i), self.seq_len):
                sentences.append(i[j:j+self.seq_len])

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

        # Automatically set batch size.
        if self.batch_size == None:
            self.batch_size = len(train_data)

        self.train_input_seq, self.train_target_seq = self.create_input_target_sequence(train_data)
        self.test_input_seq, self.test_target_seq = self.create_input_target_sequence(test_data)

        print(f"{Fore.YELLOW}{Style.BRIGHT}Train Input shape: {self.train_input_seq.shape} --> (Batch Size, Sequence Length)")
        print(f"{Fore.YELLOW}{Style.BRIGHT}Test Input shape: {self.test_input_seq.shape} --> (Batch Size, Sequence Length)")

    def create_input_target_sequence(self, data):
        # Creating lists that will hold our input and target sequences
        input_seq = []
        target_seq = []

        for i in range(self.batch_size):
            # Remove last character for input sequence
            input_seq.append(data[i][:self.seq_len-1])

            # Remove first character for target sequence
            target_seq.append(data[i][1:self.seq_len])

        for i in range(self.batch_size):
            input_seq[i] = [self.char2int[character] for character in input_seq[i]]
            target_seq[i] = [self.char2int[character] for character in target_seq[i]]

        input_seq = torch.LongTensor(input_seq).to(self.device)
        target_seq = torch.LongTensor(target_seq).to(self.device)

        return input_seq, target_seq

    def one_hot_encode(self, sequence, vocab_size):
        # Creating a multi-dimensional array of zeros with the desired output shape
        features = numpy.zeros((self.batch_size, self.seq_len, vocab_size), dtype=numpy.float32)
        
        # Replacing the 0 at the relevant character index with a 1 to represent that character
        for i in range(self.batch_size):
            for u in range(self.seq_len):
                features[i, u, sequence[i, u]] = 1

        return features
    
    def train(self):
        vocab_size = len(self.char2int)
        print(f"{Fore.YELLOW}{Style.BRIGHT}Train Vocab Size: {vocab_size}")

        # Instantiate the model with hyperparameters
        model = self.model_architecture(
            tokens = self.chars,
            output_size = vocab_size,
            hidden_dim = self.hidden_dim,
            n_layers = self.n_layers,
            embedding_dim = self.embedding_dim,
            dropout = self.dropout
        )
        model = model.to(self.device) # Set the model to the device that we defined earlier (default is CPU)

        self.train_input_seq = self.one_hot_encode(self.train_input_seq, vocab_size)
        self.train_input_seq = torch.from_numpy(self.train_input_seq)
        self.train_input_seq = self.train_input_seq.to(self.device)

        self.train_target_seq = torch.Tensor(self.train_target_seq)
        self.train_target_seq = self.train_target_seq.to(self.device)

        # https://stackoverflow.com/a/49201237/18121288
        total_params = sum(p.numel() for p in model.parameters())
        total_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

        print(f"{Fore.YELLOW}{Style.BRIGHT}Paramaters: {total_params, total_trainable_params} -> (Total Parameters, Total Trainable Parameters)")

        # Loss and optimizer
        model.train()
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=self.learning_rate)

        # Train the model
        for epoch in range(1, self.n_epochs + 1):
            try:
                optimizer.zero_grad() # Clears existing gradients from previous epoch
                output, hidden = model.forward(self.train_input_seq)
                output = output.to(self.device)

                loss = criterion(output, self.train_target_seq.view(-1).long())
                loss.backward() # Does backpropagation and calculates gradients

                optimizer.step() # Updates the weights accordingly

            except KeyboardInterrupt:
                print()
                break
