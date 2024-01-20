from src.alphabet.train import train
from src.alphabet.sample import sample

T1 = train(
    batch_size = 8,
    lr = 1e-2,
    n_embd = 8,
    n_layer = 2,
    n_hidden = 2,
    dropout = 0
)

T1.preprocess("data\\and.json", ("and", "bool", "patterns"))
T1.train(
    n_steps = 5000,
    eval_interval = 500,
    eval_iters = 500
)

T1.save("models\\and.pth")

S1 = sample("models\\and.pth")
S1.load()
S1.classify("please open google chrome")

# from src.write.train import train
# from src.write.sample import sample
# import time

# # ChatGPT like print effect.
# # dprint -> delay print
# def dprint(text, delay=0.001):
#     for char in text:
#         print(char, end="", flush=True)
#         time.sleep(delay)
#     print()

# # Train the model.
# T1 = train(
#     batch_size = 64,
#     block_size = 1024,
#     lr = 2e-3,
#     n_embd = 368,
#     n_layer = 12,
#     n_head = 12,
#     dropout = 0
# )

# T1.preprocess("data\\data.txt", 0.9)
# T1.train(
#     n_steps = 5000,
#     eval_interval = 500,
#     eval_iters = 500
# )

# T1.save("models\\GAT-w2.pth")

# # # Use the model
# S1 = sample("models\\GAT-w2.pth")
# S1.load()

# dprint(S1.generate("", length=50))
