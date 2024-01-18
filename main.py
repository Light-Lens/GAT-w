from src.write.train import train
from src.write.sample import sample
import time

# ChatGPT like print effect.
# dprint -> delay print
def dprint(text, delay=0.001):
    for char in text:
        print(char, end="", flush=True)
        time.sleep(delay)
    print()

with open("data\\extract.txt", 'r', encoding='utf-8') as f:
    text = [i.strip() for i in f.readlines()]

maxlen = len(max(text, key=len))

# A simple loop that loops through the list of sentences and adds a ' ' whitespace until the length of the sentence matches
# the length of the longest sentence
for i in range(len(text)):
    while len(text[i]) < maxlen:
        text[i] += ' '

# Train the model.
T1 = train(
    batch_size = 16,
    block_size = maxlen,
    lr = 1e-2,
    n_embd = 4,
    n_layer = 2,
    n_head = 2,
    dropout = 0
)

T1.preprocess_data("\n".join(text), 0.9)
T1.train(
    n_steps = 5000,
    eval_interval = 500,
    eval_iters = 200
)

T1.save("models\\GAT-w2_extract.pth")

# Use the model
S1 = sample("models\\GAT-w2_extract.pth")
S1.load()

dprint(S1.generate("", length=500))

# while True:
#     inp = input("> ")
#     if inp == "":
#         continue

#     elif inp == "q" or inp == "bye":
#         break

#     dprint(S1.generate(inp, length=500))
