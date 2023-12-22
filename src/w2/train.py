import torch

class Train:
    def __init__(self, n_epochs=1000, hidden_di=16, embedding_dim=16, n_layers=2, lr=0.01, dropout=0, patience=100):
        # Define hyperparameters
        self.n_epochs = n_epochs
        hidden_dim = 16
        embedding_dim = 16
        n_layers = 4
        dropout = 0
        lr = 0.01
        patience = 10000 # Adjust patience as needed

    def train(self):
        # Add early stopping
        best_loss = float('inf')

        for epoch in range(1, self.n_epochs + 1):
            try:
                optimizer.zero_grad() # Clears existing gradients from previous epoch
                input_seq_int = input_seq_int.to(device)
                output, hidden = model(input_seq_int)
                output = output.to(device)
                target_seq = target_seq.to(device)
                loss = criterion(output, target_seq.view(-1).long())
                loss.backward() # Does backpropagation and calculates gradients
                optimizer.step() # Updates the weights accordingly

                print(f'{Fore.WHITE}{Style.BRIGHT}Epoch [{epoch}/{self.n_epochs}], Loss: {loss.item():.4f}', end="\r")
                if epoch % (self.n_epochs/10) == 0:
                    # Save the model checkpoint
                    # data["model_state"] = model.state_dict()
                    # torch.save(data, f"models\\mid_epoch\\{epoch}.pth")
                    print()

                # Check for early stopping
                if loss < best_loss:
                    best_loss = loss

                else:
                    patience -= 1
                    if patience == 0:
                        print(f"\n{Fore.RED}{Style.BRIGHT}Early stopping:", "No improvement in validation loss.\n")
                        break

            except KeyboardInterrupt:
                print()
                break
