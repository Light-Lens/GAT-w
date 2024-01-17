from src.w2.train import train
from src.w2.sample import sample
from src.w2.utils import dprint

# Train the model.
T1 = train(
    batch_size = 64,
    block_size = 128,
    lr = 3e-4,
    n_embd = 256,
    n_layer = 6,
    n_head = 6,
    dropout = 0.2
)

T1.preprocess_data("data\\data_small.txt")
T1.train(
    n_steps = 10000,
    eval_interval = 1000,
    eval_iters = 200
)

T1.save("models\\GAT-w2_small.pth")

# Use the model
S1 = sample("models\\GAT-w2_small.pth")
S1.load()

while True:
    inp = input("> ")
    if inp == "":
        continue

    elif inp == "q" or inp == "bye":
        break

    dprint(S1.generate(inp, length=50))
