import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from networks import AB2Net
import random
from tqdm import tqdm
from scipy.integrate import solve_ivp

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


def physics_loss_2_body(input, output, dt, G=1.0, eps=1e-3):
    m1, m2 = output[:, 0], output[:, 5]

    x1_p, y1_p = input[:, 1:2], input[:, 2:3]
    vx1_p, vy1_p = input[:, 3:4], input[:, 4:5]
    x2_p, y2_p = input[:, 6:7], input[:, 7:8]
    vx2_p, vy2_p = input[:, 8:9], input[:, 9:10]

    x1, y1 = output[:, 1:2], output[:, 2:3]
    vx1, vy1 = output[:, 3:4], output[:, 4:5]
    x2, y2 = output[:, 6:7], output[:, 7:8]
    vx2, vy2 = output[:, 8:9], output[:, 9:10]

    def accel(xa, ya, xb, yb, m_other):
        r = torch.sqrt((xb - xa) ** 2 + (yb - ya) ** 2 + eps ** 2)
        return G * m_other * (xb - xa) / r ** 3, G * m_other * (yb - ya) / r ** 3

    ax1_p, ay1_p = accel(x1_p, y1_p, x2_p, y2_p, m2)
    ax2_p, ay2_p = accel(x2_p, y2_p, x1_p, y1_p, m1)

    # v(t+dt/2), usata solo internamente per il residuo di posizione (velocity Verlet)
    vx1_half = vx1_p + 0.5 * ax1_p * dt
    vy1_half = vy1_p + 0.5 * ay1_p * dt
    vx2_half = vx2_p + 0.5 * ax2_p * dt
    vy2_half = vy2_p + 0.5 * ay2_p * dt

    loss = (torch.mean((x1 - x1_p - vx1_half * dt) ** 2) +
            torch.mean((y1 - y1_p - vy1_half * dt) ** 2) +
            torch.mean((x2 - x2_p - vx2_half * dt) ** 2) +
            torch.mean((y2 - y2_p - vy2_half * dt) ** 2))

    ax1, ay1 = accel(x1, y1, x2, y2, m2)   # forza valutata sulla posizione predetta
    ax2, ay2 = accel(x2, y2, x1, y1, m1)

    loss = loss + (
        torch.mean((vx1 - vx1_half - 0.5 * ax1 * dt) ** 2) +
        torch.mean((vy1 - vy1_half - 0.5 * ay1 * dt) ** 2) +
        torch.mean((vx2 - vx2_half - 0.5 * ax2 * dt) ** 2) +
        torch.mean((vy2 - vy2_half - 0.5 * ay2 * dt) ** 2))
    return loss


def compute_energy(states, G=1.0, eps=1e-3):
    """Total energy E = T + V."""
    m1, m2 = states[:, 0], states[:, 5]
    x1, y1 = states[:, 1], states[:, 2]
    x2, y2 = states[:, 6], states[:, 7]
    vx1, vy1 = states[:, 3], states[:, 4]
    vx2, vy2 = states[:, 8], states[:, 9]

    T = 0.5 * m1 * (vx1**2 + vy1**2) + 0.5 * m2 * (vx2**2 + vy2**2)
    r12 = torch.sqrt((x2 - x1)**2 + (y2 - y1)**2 + eps**2)
    V = -G * (m1 * m2 / r12)
    return T + V


def compute_angular_momentum(states):
    """Total angular momentum L = sum m_i*(x_i*vy_i - y_i*vx_i)."""
    m1 = states[:, 0]
    m2 = states[:, 5]
    L = (m1 * (states[:, 1] * states[:, 4] - states[:, 2] * states[:, 3]) +
         m2 * (states[:, 6] * states[:, 9] - states[:, 7] * states[:, 8]))
    return L


def conservation_loss(input, output, G=1.0, eps=1e-3):
    """Penalise RELATIVE energy and angular-momentum drift between input state and output state.

    Usiamo il drift relativo (rispetto a |E0|, |L0|) invece che assoluto: con masse/velocità
    che variano di batch in batch, il drift assoluto puo' avere scale molto diverse da p_loss
    e dominare il gradiente in modo instabile. Il drift relativo resta O(1) indipendentemente
    dalla scala del sistema fisico campionato.
    """
    E, E0 = compute_energy(output, G, eps), compute_energy(input, G, eps)
    L, L0 = compute_angular_momentum(output), compute_angular_momentum(input)

    loss_E = torch.mean(((E - E0) / (E0.abs() + eps)) ** 2)
    loss_L = torch.mean(((L - L0) / (L0.abs() + eps)) ** 2)
    return loss_E + loss_L


def train(epochs: int,
          BodyNetwork: nn.Module,
          optimizer: torch.optim.Optimizer,
          scheduler,
          device: torch.device = torch.device('cpu'),
          batch_size: int = 256,
          dt: float = 0.05):

    loss_history = []
    # Ricordarsi questo per l'input, altrimenti niente gradiente all'indietro: input.requires_grad_(True)
    pbar = tqdm(range(epochs), desc="Training")

    for i in pbar:
        total_t = random.randint(1, min(30, 1 + i // 50))
        optimizer.zero_grad()

        c_weight = min(i / 100, 1)

        total = 0
        input = generate_instance(batch_size, device)
        input.requires_grad_(True)

        for _ in range(total_t):
            output = BodyNetwork(input)
            # ATTENZIONE A INPUT E OUTPUT COSA CONTENGONO
            p_loss = physics_loss_2_body(input, output, dt)
            c_loss = conservation_loss(input, output)

            total += p_loss + c_weight * c_loss
            input = output

        total /= total_t

        # Controllo di sanità: se un batch produce una loss non finita (nan/inf, es. per un
        # incontro ravvicinato tra i due corpi con r12 -> 0), saltiamo lo step invece di
        # propagare gradienti corrotti che possono far collassare la rete per il resto del
        # training.
        if not torch.isfinite(total):
            tqdm.write(f"epoch {i}: loss non finita ({total.item()}), skip step")
            continue

        total.backward()
        torch.nn.utils.clip_grad_norm_(BodyNetwork.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step(total.item())
        loss_history.append(total.item())

        pbar.set_postfix(loss=f"{total.item():.4e}", lr=f"{optimizer.param_groups[0]['lr']:.1e}")

    return loss_history


def generate_instance(batch_size=256, device=torch.device('cpu'), dtype=torch.float64):
    # Unità nondimensionali: masse e distanze O(1) invece di O(1000)
    m0 = torch.rand(batch_size, 1, device=device, dtype=dtype) * 2 + 0.1   # massa in [0.1, 2.1]
    m1 = torch.rand(batch_size, 1, device=device, dtype=dtype) * 2 + 0.1

    x0 = torch.zeros(batch_size, 2, device=device, dtype=dtype)            # corpo 1 nell'origine
    x1 = torch.zeros(batch_size, 2, device=device, dtype=dtype)
    x1[:, 0] = torch.rand(batch_size, device=device, dtype=dtype) * 2 + 0.5  # corpo 2 su asse x, distanza in [0.5, 2.5]

    q0 = torch.randn(batch_size, 2, device=device, dtype=dtype) * 0.3      # velocità O(0.1-1)
    q1 = torch.randn(batch_size, 2, device=device, dtype=dtype) * 0.3

    state = torch.cat([m0, x0, q0, m1, x1, q1], dim=1)
    state.requires_grad_(True)
    return state


def train_network(DEVICE: torch.device = torch.device('cpu')):

    print(f"Device {DEVICE}")

    # Parameters
    dim = 10

    n_blocks = 4

    epochs = 2000
    batch_size = 256
    lr = 1e-3
    dt = 0.05

    # Network initialization
    BodyNetwork = AB2Net(
        in_features=dim,
        out_features=8,
        num_blocks=n_blocks,
        dtype=torch.float64
    ).to(DEVICE)

    # Model training
    optimizer = torch.optim.Adam(
        BodyNetwork.parameters(),
        lr=lr
    )

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.5,
        patience=50,
        min_lr=1e-6  # evita che il lr collassi fino a diventare inutile su un plateau rumoroso
    )

    losses = train(
        epochs,
        BodyNetwork,
        optimizer,
        scheduler,
        device=DEVICE,
        batch_size=batch_size,
        dt=dt
    )

    return BodyNetwork, optimizer, scheduler, epochs, losses


if __name__ == "__main__":

    print("=" * 90)

    PATH = "./PINN_savefile/save.pt"

    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)  # fissa anche l'RNG della GPU, non solo quello CPU

    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    net, optimizer, scheduler, epoch, losses = train_network(DEVICE)
    torch.save({
        "model": net.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "epoch": epoch,
        "losses": losses
    }, PATH)

    plt.plot(losses)
    plt.yscale('log')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.savefig('loss_curve.png')