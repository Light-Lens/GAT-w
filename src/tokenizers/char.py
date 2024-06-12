class CharTokenizer:
    # encoder: take a string, output a list of integers
    def encode(text, stoi):
        return [stoi[c] for c in text]

    # decoder: take a list of integers, output a string
    def decode(tensor, itos):
        return ''.join([itos[i] for i in tensor])
