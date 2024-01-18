import time

# ChatGPT like print effect.
# dprint -> delay print
def dprint(text, delay=0.001):
    for char in text:
        print(char, end="", flush=True)
        time.sleep(delay)
    print()

# encoder: take a string, output a list of integers
def encode(text, stoi):
    return [stoi[c] for c in text]

# decoder: take a list of integers, output a string
def decode(tensor, itos):
    return ''.join([itos[i] for i in tensor])
