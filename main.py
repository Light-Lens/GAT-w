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
    batch_size = 64,
    block_size = 50,
    lr = 2e-3,
    n_embd = 16,
    n_layer = 4,
    n_head = 4,
    dropout = 0
)

T1.preprocess_data("data\\data_small.txt", 0.9)
T1.train(
    n_steps = 5000,
    eval_interval = 1000,
    eval_iters = 200
)

T1.save("models\\GAT-w2_chat.pth")

# Use the model
S1 = sample("models\\GAT-w2_chat.pth")
S1.load()

while True:
    inp = input("> ")
    if inp == "":
        continue

    elif inp == "q" or inp == "bye":
        break

    dprint(S1.generate(inp, length=50))
