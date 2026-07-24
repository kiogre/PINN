import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from networks import AB2Net
from torch.utils.data import TensorDataset, random_split, DataLoader

'''
def save_checkpoint(model, optimizer, scheduler, epoch, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    torch.save({
        "model":     model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "epoch":     epoch,
    }, tmp)
    #os.replace(tmp, path)
    #tqdm.write(f"  -> checkpoint salvato: {path}  (epoch {epoch})")
'''

def train(epochs: int, 
          BodyNetwork: nn.Module, 
          train_loader: DataLoader, 
          optimizer: torch.optim.Optimizer, 
          scheduler, 
          criterion: nn.Module = nn.MSELoss(), 
          device: torch.device = torch.device('cpu')):
    pass

def make_dataset():
    pass

def test(BodyNetwork: nn.Module, 
         test_loader: DataLoader, 
         device: torch.device = torch.device('cpu')):
    pass

def train_network(DEVICE: torch.device = torch.device('cpu')):

    torch.manual_seed(96)

    print(f"Device {DEVICE}")

    # Parameters
    dim = 2
    seed = 0

    pad_dim = 4
    n_blocks = 4

    epochs = 1500
    batch_size = 256
    test_ratio = 0.2
    lr = 2**-7

    # Network initialization
    BodyNetwork = AB2Net(
        in_features=dim,
        out_features=2,
        pad_dim=pad_dim,
        num_blocks=n_blocks,
        dtype=torch.float
    ).to(DEVICE)

    # Dataset and train test split initialization
    x, y = make_dataset()

    dataset = TensorDataset(x, y)
    test_len = int(len(dataset) * test_ratio)
    train_len = len(dataset) - test_len

    train_set, test_set = random_split(
        dataset,
        [train_len, test_len],
        generator=torch.Generator().manual_seed(seed)
    )

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)

    # Model training
    optimizer = torch.optim.Adam(
        BodyNetwork.parameters(),
        lr=lr
    )

    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer,
        step_size=30,
        gamma=0.96875,
        last_epoch=-1
    )

    losses = train(
        epochs,
        BodyNetwork,
        train_loader,
        optimizer,
        scheduler,
        criterion=nn.CrossEntropyLoss(),
        device=DEVICE
    )

    # Performance evaluation
    test(BodyNetwork, test_loader, device=DEVICE)

    return BodyNetwork, optimizer, scheduler, epochs

if __name__ == "__main__":

    print("="*90)

    PATH = "./PINN_savefile/save.pt"

    torch.seed(42)
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    net, optimizer, scheduler, epoch = train_network(DEVICE)
    torch.save({
            "model":     net.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "epoch":     epoch,
        }, )