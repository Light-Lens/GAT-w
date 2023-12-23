from torch import nn
import torch, numpy

class eval:
    def __init__(self, model_data):
        self.model_state = model_data["model_state"]
        self.dict_size = model_data["input_size"]
        self.hidden_dim = model_data["hidden_dim"]
        self.embedding_dim = model_data["embedding_dim"]
        self.n_layers = model_data["n_layers"]
        self.dropout = model_data["dropout"]
        self.device = model_data["device"]
        self.int2char = model_data["int2char"]
        self.char2int = model_data["char2int"]
        self.model_architecture = model_data["model_architecture"]

        self.model = self.model_architecture(
            input_size = self.dict_size,
            output_size = self.dict_size,
            hidden_dim = self.hidden_dim,
            n_layers = self.n_layers,
            embedding_dim = self.embedding_dim,
            dropout = self.dropout
        )
        self.model.load_state_dict(self.model_state)
        self.model = self.model.to(self.device)

    def predict(self, character, temperature=1.0):
        character = numpy.array([[self.char2int[c] for c in character]])
        character = torch.from_numpy(character)
        character = character.to(self.device)
        
        out, hidden = self.model(character)

        # Adjust the output probabilities with temperature
        prob = nn.functional.softmax(out[-1] / temperature, dim=0).data
        # Sample from the modified distribution
        char_ind = torch.multinomial(prob, 1).item()

        return self.int2char[char_ind], hidden

    def generate(self, seed, outlen, temperature=1.0):
        self.model.eval() # eval mode

        # First off, run through the starting characters
        chars = list(seed)
        size = outlen - len(chars)

        # Now pass in the previous characters and get a new one
        for _ in range(size):
            char, h = self.predict(chars, temperature)
            chars.append(char)

        return "".join(chars)
