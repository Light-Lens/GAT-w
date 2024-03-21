import torch

class Train:
    def __init__(self, n_layer, n_hidden, lr, batch_size, model="FeedForward", device="auto"):
        """
        @param n_layer: Number of layers
        @param n_hidden: Hidden size
        @param lr: Learning rate
        @param model: Model architecture to train on. [FeedForward, RNN] (default: FeedForward)
        @param device: Training device. [auto, cpu, cuda] (default: auto)
        """
        # hyperparameters
        self.n_hidden = n_hidden
        self.n_layer = n_layer
        self.device = device
        self.learning_rate = lr
        self.batch_size = batch_size # how many independent sequences will we process in parallel?
        self.device = ("cuda" if torch.cuda.is_available() else "cpu") if device == "auto" else device
        self.model_architecture = model

        # print the device
        print("Training on", self.device)

    def preprocess(self, filepath, metadata, data_division=0.8):
        """
        @param filepath: the location of the json file.
        @param metadata: (classname, tagname, pattern_name)
        @param data_division: if None then only train otherwise train and test (between: 0 and 1) (default: 0.8)
        """
        pass

    def train(self, n_steps, eval_interval, eval_iters, checkpoint_interval=0, checkpoint_path="", n_loss_digits=4):
        """
        @param n_steps: number of Epochs to train the model for
        @param eval_interval: the interval between each loss evaluation
        @param eval_iters: the iterations for each loss evaluation
        @param checkpoint_interval: the interval between each checkpoint save (default: 0)
        @param checkpoint_path: the save path for the checkpoint (default: empty string)
        @param n_loss_digits: Number of digits of train and val loss printed (default: 4)
        """

    def save(self, savepath):
        pass
