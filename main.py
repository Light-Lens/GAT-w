from src.text.generation.train import train
from src.text.generation.sample import sample
from src.text.generation.utils import dprint

# Train the model.
T1 = train(
    n_layer = 5,
    n_embd = 64,
    n_head = 5,
    lr = 1e-3,
    dropout = 0,
    block_size = 64,
    batch_size = 16
)

T1.preprocess_data("data\\data.txt")
T1.train(
    n_steps = 5000,
    eval_interval = 200,
    eval_iters = 200
)

T1.save("models\\GAT-w2.pth")

# Use the model
S1 = sample("models\\GAT-w2.pth")
S1.load()
dprint(S1.generate("Hi"))
