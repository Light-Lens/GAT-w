from torch import nn
from src.utils import sent_tokenize
import random, numpy, torch

class Eval:
    def __init__(self, device):
        self.device = device

    def load_dataset(self, path):
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
