from src.write.train import train
from src.write.sample import sample
from src.write.utils import dprint

# Train the model.
T1 = train(
    batch_size = 16,
    block_size = 50,
    lr = 1e-2,
    n_embd = 16,
    n_layer = 3,
    n_head = 3,
    dropout = 0.2
)

T1.preprocess_data("data\\data_chatgpt.txt", 0.9)
T1.train(
    n_steps = 5000,
    eval_interval = 1000,
    eval_iters = 200
)

T1.save("models\\GAT-w2_light.pth")

# Use the model
S1 = sample("models\\GAT-w2_light.pth")
S1.load()

while True:
    inp = input("> ")
    if inp == "":
        continue

    elif inp == "q" or inp == "bye":
        break

    dprint(S1.generate(inp, length=50))
