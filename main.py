from src.alphabet.train import Train
from src.alphabet.sample import Sample
from src.write.train import Train as wtrain
from src.write.sample import Sample as wsample
import time

# ChatGPT like print effect.
# dprint -> delay print
def dprint(text, delay=0.001):
    for char in text:
        print(char, end="", flush=True)
        time.sleep(delay)
    print()

# Train the model.
# gen = wtrain(
#     batch_size = 64,
#     block_size = 1024,
#     lr = 2e-3,
#     n_embd = 368,
#     n_layer = 12,
#     n_head = 12,
#     dropout = 0
# )

# gen.preprocess("data\\data.txt", 0.9)
# gen.train(
#     n_steps = 5000,
#     eval_interval = 500,
#     eval_iters = 500
# )

# gen.save("models\\GAT-w2.pth")

# # Use the model
# S1 = wsample("models\\GAT-w2.pth")
# S1.load()

# dprint(S1.generate("", length=50))



# Train the model
classify = Train(
    n_layer = 3,
    n_hidden = 2,
    lr = 1e-3,
    batch_size = 32,
)

classify.preprocess("data\\and.json", metadata=("and", "bool", "patterns"))
classify.train(
    n_steps = 4000,
    eval_interval = 200,
    eval_iters = 4000
)

classify.save("models\\and.pth")


test = [
    "open Google chrome and Can you please search google for me?",
    "What is a game engine?",
    "please search on google What is a game engine?",
    "Google Chrome",
    "Please start Unity game engine for me please",
    "open youtube.com please",
    "if you don't mind would you please search google about shahrukh khan",
    "how to start a youtube channel",
    "what is the capital of paris",
    "please search on google what is the capital of paris",
    "reload this PC, it needs some",
    "i'm gonna be away for a while, please lock this pc",
    "shutdown my workstation please",
    "shutdown google chrome",
    "kill this app",
    "tell me current date please",
    "tell me today's time",
    "is it monday today?",
    "9 AM?",
    "Please start Unity game engine for me please",
    "start chrome.exe",
    "You know yesterday I was playing a game and I lost it :(",
    "How do you make a game engine and remember to make it in c++",
    "open chrome and search on the internet How do you make a game engine"
]

S1 = Sample("models\\and.pth")
S1.load()
for i in test:
    t, c = S1.predict(i)
    print(i)
    print(t, f"{c:.4}")
    print()
# print(S1.predict("open Google chrome and Can you please search google for me?"))
# print(S1.predict("What is a game engine?"))
# print(S1.predict("please search on google What is a game engine?"))
# print(S1.predict("Google Chrome"))
# print(S1.predict("Please start Unity game engine for me please"))
# print(S1.predict("open youtube.com please"))
# print(S1.predict("if you don't mind would you please search google about shahrukh khan"))
# print(S1.predict("how to start a youtube channel"))
# print(S1.predict("what is the capital of paris"))
# print(S1.predict("please search on google what is the capital of paris"))
# print(S1.predict("reload this PC, it needs some"))
# print(S1.predict("i'm gonna be away for a while, please lock this pc"))
# print(S1.predict("shutdown my workstation please"))
# print(S1.predict("shutdown google chrome"))
# print(S1.predict("kill this app"))
# print(S1.predict("tell me current date please"))
# print(S1.predict("tell me today's time"))
# print(S1.predict("is it monday today?"))
# print(S1.predict("9 AM?"))
# print(S1.predict("Please start Unity game engine for me please"))
# print(S1.predict("start chrome.exe"))
# print(S1.predict("You know yesterday I was playing a game and I lost it :("))
# print(S1.predict("How do you make a game engine and remember to make it in c++"))
# print(S1.predict("open chrome and search on the internet How do you make a game engine"))
