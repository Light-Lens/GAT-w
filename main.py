from src.w2.train import train
import src.w2.eval as eval, torch

T1 = train(
    n_epochs = 10000,
    hidden_dim = 128,
    embedding_dim = 256,
    n_layers = 3,
    lr = 1e-3,
    batch_size = 8,
    patience = 8000
)

T1.savepath = "models\\model.pth"

T1.preprocess("data\\data.txt")
model_data = T1.train()

# model_data = torch.load("models\\model.pth")

C1 = eval.eval(model_data)
print(C1.generate("Cupid laid by his brand", 1000))
