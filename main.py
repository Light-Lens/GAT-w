from src.w2.train import train
import src.w2.eval as eval, torch

# T1 = train(n_epochs = 2500, hidden_dim = 32, embedding_dim = 8, n_layers = 1, lr = 0.01, patience = 1000)
# T1.savepath = "models\\model.pth"
# T1.preprocess("data\\data.txt")
# model_data = T1.train()

model_data = torch.load("models\\model.pth")

C1 = eval.eval(model_data)
print(C1.generate("please open microsoft edge", 100))
