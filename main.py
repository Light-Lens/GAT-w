from src.w2.train import train
from src.w2.sample import sample
from src.w2.utils import dprint

# Train the model.
T1 = train(
    batch_size = 16,
    block_size = 1024,
    lr = 3e-4,
    n_embd = 128,
    n_layer = 8,
    n_head = 8,
    dropout = 0
)

T1.preprocess_data("data\\data.txt")
T1.train(
    n_steps = 300000,
    eval_interval = 2000,
    eval_iters = 200
)

T1.save("models\\GAT-w2.pth")

# Use the model
S1 = sample("models\\GAT-w2.pth")
S1.load()

while True:
    inp = input("> ")
    if inp == "":
        continue

    elif inp == "q" or inp == "bye":
        break

    dprint(S1.generate(inp, length=100))
