from src.w2.train import train
from src.w2.sample import sample
from src.w2.utils import dprint

# Train the model.
T1 = train(
    batch_size = 16,
    block_size = 50,
    lr = 2e-3,
    n_embd = 64,
    n_layer = 6,
    n_head = 6,
    dropout = 0.2
)

T1.preprocess_data("data\\data_small.txt", 0.9)
T1.train(
    n_steps = 100000,
    eval_interval = 2000,
    eval_iters = 200,
    checkpoint_interval = 2000,
    checkpoint_path = "models\\checkpoint\\GAT-w2_small.pth"
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
