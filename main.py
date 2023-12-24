from src.w2.train import train
import src.w2.eval as eval, torch

T1 = train(n_epochs = 1000, hidden_dim = 128, embedding_dim = 128, n_layers = 3, lr = 0.001, batch_size = 64, patience = 500)
T1.savepath = "models\\model.pth"
T1.preprocess("data\\data.txt")
model_data = T1.train()

# model_data = torch.load("models\\model.pth")

C1 = eval.eval(model_data)
print(C1.generate("Find the product of the numbers: 5 and 8", 100))
# print(C1.generate("please open google chrome", 100))
# print(C1.generate("search on google what works a search engine", 100))
