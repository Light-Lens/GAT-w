# from src.write.train import train
# from src.write.sample import sample
from src.alphabet.train import train
from src.alphabet.sample import sample
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
    block_size = 100,
    lr = 2e-3,
    n_embd = 64,
    n_layer = 12,
    n_head = 12,
    dropout = 0
)

T1.preprocess("data\\and.json", "and", 0.9)
# T1.preprocess("data\\data.txt", 0.9)
T1.train(
    n_steps = 5000,
    eval_interval = 500,
    eval_iters = 200
)

T1.save("models\\and.pth")

# # Use the model
S1 = sample("models\\and.pth")
S1.load()

print(S1.classify("search for a neural network and play mere hi liye"))
print(S1.classify("if you don't mind would you please search google about shahrukh khan"))
print(S1.classify("open chrome and search on the internet How do you make a game engine"))
print(S1.classify("please search on google What is a game engine?"))

# dprint(S1.generate("", length=50))
