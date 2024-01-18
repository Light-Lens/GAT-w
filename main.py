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

# Train the model.
T1 = train(
    batch_size = 128,
    block_size = 100,
    lr = 1e-3,
    n_embd = 64,
    n_layer = 5,
    n_head = 5,
    dropout = 0
)

with open("data\\extract.txt", 'r', encoding='utf-8') as f:
    text = f.read()

T1.preprocess_data(text, 0.9)
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
