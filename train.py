import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from networks import AB2Net
from torch.utils.data import TensorDataset, random_split, DataLoader
import random
from math import sqrt

'''
L'output della rete adesso è m_1, p_1 vettore, q_1 vettore e avanti così
Bisogna modificare tutte le loss
'''

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


def physics_loss_2_body(input, output, masses, G=1.0, eps=1e-3):
    """Enforce Newton's law of gravitation for all 12 first-order ODEs."""

    # tmp, perché abbiamo anche le masse
    x1, y1 = output[:, 0:1], output[:, 1:2]
    x2, y2 = output[:, 2:3], output[:, 3:4]
    vx1, vy1 = output[:, 4:5], output[:, 5:6]
    vx2, vy2 = output[:, 6:7], output[:, 7:8]

    ones = torch.ones_like(x1)

    dx1 = torch.autograd.grad(x1, input, ones, create_graph=True)[0]
    dy1 = torch.autograd.grad(y1, input, ones, create_graph=True)[0]
    dx2 = torch.autograd.grad(x2, input, ones, create_graph=True)[0]
    dy2 = torch.autograd.grad(y2, input, ones, create_graph=True)[0]
    dvx1 = torch.autograd.grad(vx1, input, ones, create_graph=True)[0]
    dvy1 = torch.autograd.grad(vy1, input, ones, create_graph=True)[0]
    dvx2 = torch.autograd.grad(vx2, input, ones, create_graph=True)[0]
    dvy2 = torch.autograd.grad(vy2, input, ones, create_graph=True)[0]

    m1, m2 = masses

    r12 = torch.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2 + eps ** 2)

    ax1 = G * m2 * (x2 - x1) / r12 ** 3
    ay1 = G * m2 * (y2 - y1) / r12 ** 3
    ax2 = G * m1 * (x1 - x2) / r12 ** 3
    ay2 = G * m1 * (y1 - y2) / r12 ** 3

    # Kinematic residuals
    loss = (torch.mean((dx1 - vx1) ** 2) + torch.mean((dy1 - vy1) ** 2) +
            torch.mean((dx2 - vx2) ** 2) + torch.mean((dy2 - vy2) ** 2))
    # Dynamic residuals
    loss = loss + (
        torch.mean((dvx1 - ax1) ** 2) + torch.mean((dvy1 - ay1) ** 2) +
        torch.mean((dvx2 - ax2) ** 2) + torch.mean((dvy2 - ay2) ** 2))
    return loss


def conservation_loss(input, output, masses, G=1.0, eps=1e-3):
    """
    Penalise energy and angular-momentum drift along the trajectory.
    E(t) should equal E(0); L(t) should equal L(0).
    """
    '''
    with torch.no_grad():
        output = model(t_col)
    states_np = output.numpy()'''

    E = compute_energy(output, masses, G, eps)
    E0 = compute_energy(input, masses, G, eps)
    L = compute_angular_momentum(output, masses)
    L0 = compute_angular_momentum(input, masses)

    loss_E = np.mean((E - E0) ** 2)
    loss_L = np.mean((L - L0) ** 2)
    return torch.tensor(loss_E + loss_L, dtype=torch.float32)


def compute_energy(states, masses, G=1.0, eps=1e-3):
    """Total energy E = T + V.
    DA MODIFICARE"""
    x1, y1 = states[:, 0], states[:, 1]
    x2, y2 = states[:, 2], states[:, 3]
    x3, y3 = states[:, 4], states[:, 5]
    vx1, vy1 = states[:, 6], states[:, 7]
    vx2, vy2 = states[:, 8], states[:, 9]
    vx3, vy3 = states[:, 10], states[:, 11]
    m1, m2, m3 = masses

    T = (0.5 * m1 * (vx1 ** 2 + vy1 ** 2) +
         0.5 * m2 * (vx2 ** 2 + vy2 ** 2) +
         0.5 * m3 * (vx3 ** 2 + vy3 ** 2))

    r12 = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2 + eps ** 2)
    r13 = np.sqrt((x3 - x1) ** 2 + (y3 - y1) ** 2 + eps ** 2)
    r23 = np.sqrt((x3 - x2) ** 2 + (y3 - y2) ** 2 + eps ** 2)

    V = -G * (m1 * m2 / r12 + m1 * m3 / r13 + m2 * m3 / r23)
    return T + V

def compute_angular_momentum(states):
    """Total angular momentum L = sum m_i*(x_i*vy_i - y_i*vx_i)."""
    m1 = states[:, 0]
    m2 = states[:, 5]
    L = (m1 * (states[:, 1] * states[:, 7] - states[:, 2] * states[:, 6]) +
         m2 * (states[:, 3] * states[:, 9] - states[:, 4] * states[:, 8]))
    return L


def train(epochs: int, 
          BodyNetwork: nn.Module,
          optimizer: torch.optim.Optimizer, 
          scheduler,
          device: torch.device = torch.device('cpu'),
          batch_size: int = 256,
          c_weight: float = 1.0):

    loss_history = []
    # Ricordarsi questo per l'input, altrimenti niente gradiente all'indietro: input.requires_grad_(True)
    for i in range(epochs):
        total_t = random.randint(1, 30)
        optimizer.zero_grad()

        total = 0
        input = generate_instance(batch_size, device)
        input.requires_grad_(True)

        for _ in total_t:
            output = BodyNetwork(input)
            # ATTENZIONE A INPUT E OUTPUT COSA CONTENGONO
            p_loss = physics_loss_2_body(input, output)
            c_loss = conservation_loss(input, output)

            total += p_loss + c_weight * c_loss
            input = output

        total /= total_t

        total.backward()
        optimizer.step()
        scheduler.step(total.item())
        loss_history.append(total.item())

    return loss_history


def test(BodyNetwork: nn.Module,
         device: torch.device = torch.device('cpu')):
    pass


def generate_instance(batch_size: int = 256,
                      device: torch.device = torch.device('cpu')):
    scale = 1000
    batch = []

    for i in range(batch_size):
        x_0 = [0, 0]
        x_1 = [0, 0]
        x_1[0] = random.random() * scale
        q_0 = np.random.normal(size = 2) * sqrt(scale)
        q_1 = np.random.normal(size = 2) * sqrt(scale)

        m_0 = random.randint(1, 2*scale)
        m_1 = random.randint(1, 2*scale)
        batch.append((m_0, x_0[0], x_0[1], q_0[0], q_1[0], m_1, x_1[0], x_1[1], q_1[0], q_1[1]))

    return torch.tensor(batch, device=device, requires_grad=True)


def train_network(DEVICE: torch.device = torch.device('cpu')):

    torch.manual_seed(42)

    print(f"Device {DEVICE}")

    # Parameters
    dim = 2

    pad_dim = 4
    n_blocks = 4

    epochs = 2000
    batch_size = 256
    lr = 2**-7

    # Network initialization
    BodyNetwork = AB2Net(
        in_features=dim,
        out_features=2,
        pad_dim=pad_dim,
        num_blocks=n_blocks,
        dtype=torch.float
    ).to(DEVICE)

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
        optimizer,
        scheduler,
        criterion=nn.CrossEntropyLoss(),
        device=DEVICE,
        batch_size = batch_size
    )

    # Performance evaluation
    test(BodyNetwork, device=DEVICE)

    return BodyNetwork, optimizer, scheduler, epochs, losses

if __name__ == "__main__":

    print("="*90)

    PATH = "./PINN_savefile/save.pt"

    torch.seed(42)
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    net, optimizer, scheduler, epoch, losses = train_network(DEVICE)
    torch.save({
            "model":     net.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "epoch":     epoch,
            "losses":    losses
        }, )