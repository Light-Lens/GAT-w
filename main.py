from src.w2.train import train
from src.w2.sample import sample
from src.w2.utils import dprint

# Train the model.
T1 = train(
    batch_size = 16,
    block_size = 64,
    lr = 1e-3,
    n_embd = 64,
    n_layer = 6,
    n_head = 6,
    dropout = 0
)

T1.preprocess_data("data\\data_small.txt")
T1.train(
    n_steps = 10000,
    eval_interval = 1000,
    eval_iters = 200
)

T1.save("models\\GAT-w2.pth")

# Use the model
S1 = sample("models\\GAT-w2.pth")
S1.load()
dprint(S1.generate("Human 1: Hi\nHuman 2: ", length=20, temperature=0.2, top_k=200))
