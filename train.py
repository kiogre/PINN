import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from networks import Acceleration2BodyNet, Acceleration2BodyNetv2
from networks import Acceleration2BodyNetv3, Acceleration2BodyNetv4
import random
from tqdm import tqdm
from scipy.integrate import solve_ivp


def canonicalize_translation(state: torch.Tensor):
    """
    Trasla lo stato [Batch, N * 5] in modo che il Centro di Massa (CoM)
    sia esattamente in (0, 0).
    
    Restituisce:
    - state_canon: lo stato traslato col CoM in (0,0)
    - R_cm: il vettore di traslazione del CoM [Batch, 1, 2] per ripristinare le posizioni originali
    """
    B = state.shape[0]
    x_r = state.view(B, -1, 5) # [Batch, N, 5] -> [m, x, y, vx, vy]

    m = x_r[:, :, 0:1]     # [B, N, 1]
    p = x_r[:, :, 1:3]     # [B, N, 2]
    v = x_r[:, :, 3:5]     # [B, N, 2] (le velocità non cambiano per traslazione spaziale)

    # Calcolo del Centro di Massa: R_cm = sum(m_i * p_i) / sum(m_i)
    M_tot = torch.sum(m, dim=1, keepdim=True) # [B, 1, 1]
    R_cm = torch.sum(m * p, dim=1, keepdim=True) / M_tot # [B, 1, 2]

    # Traslazione delle posizioni: CoM diventa (0, 0)
    p_canon = p - R_cm  

    state_canon = torch.cat([m, p_canon, v], dim=-1).view(B, -1)
    return state_canon, R_cm


def uncanonicalize_translation(state_canon: torch.Tensor, R_cm: torch.Tensor):
    """
    Ripristina le posizioni originali sommando indietro R_cm.
    """
    B = state_canon.shape[0]
    x_r = state_canon.view(B, -1, 5)

    m = x_r[:, :, 0:1]
    p_canon = x_r[:, :, 1:3]
    v = x_r[:, :, 3:5]

    p_orig = p_canon + R_cm

    state_orig = torch.cat([m, p_orig, v], dim=-1).view(B, -1)
    return state_orig


def physics_loss_2_body(input_state, output_state, dt=0.01, G=1.0, eps=1e-3, net=None):
    """
    Valuta se l'accelerazione appresa dalla rete coincide con l'accelerazione
    gravitazionale di Newton a_grav = G * m_other * r / |r|^3
    """
    B = input_state.shape[0]
    x_r = input_state.view(B, 2, 5)

    m1, m2 = x_r[:, 0, 0:1], x_r[:, 1, 0:1] # [B, 1]
    p1, p2 = x_r[:, 0, 1:3], x_r[:, 1, 1:3] # [B, 2]

    # Vettore distanza relativa r_12 e r_21
    r12 = p2 - p1
    dist = torch.sqrt(torch.sum(r12**2, dim=-1, keepdim=True) + eps**2) # [B, 1]

    # Accelerazione gravitazionale ESATTA (Target di Newton)
    a1_target = G * m2 * r12 / (dist ** 3)
    a2_target = - G * m1 * r12 / (dist ** 3)
    a_target = torch.stack([a1_target, a2_target], dim=1) # [B, 2, 2]

    # Accelerazione PREDETTA dalla rete sullo stato corrente
    # Se passi 'net' usiamo la funzione interna predict_acceleration
    if net is not None:
        a_pred = net.predict_acceleration(input_state)
    else:
        # Alternativa derivata dall'output del forward via Verlet: a ~ (v_next - v) / dt
        v_in = x_r[:, :, 3:5]
        v_out = output_state.view(B, 2, 5)[:, :, 3:5]
        a_pred = (v_out - v_in) / dt

    # Huber Loss (Smooth L1) per evitare instabilità ai pericentri
    loss = F.smooth_l1_loss(a_pred, a_target, beta=1e-2)
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


def conservation_loss(initial, output, G=1.0, eps=1e-2):
    """Penalise RELATIVE energy and angular-momentum drift rispetto allo stato INIZIALE
    della traiettoria (non allo step precedente): la legge di conservazione dice
    E(t) = E(0) per ogni t, quindi il riferimento corretto è sempre lo stato a inizio
    rollout, non l'ultimo stato visitato.
    """
    E, E0 = compute_energy(output, G, eps), compute_energy(initial, G, eps)
    L, L0 = compute_angular_momentum(output), compute_angular_momentum(initial)

    loss_E = torch.mean(((E - E0) / (E0.abs() + eps)) ** 2)
    loss_L = torch.mean(((L - L0) / (L0.abs() + eps)) ** 2)
    return loss_E + loss_L


def _run_epoch(i, BodyNetwork, optimizer, scheduler, device, batch_size, dt,
               total_t_max, c_weight, dtype, a_weight):
    """Un'epoca di training: rollout autoregressivo, loss, backward, step.
    Fattorizzata a parte perche' e' condivisa identica tra fase di pretraining
    (c_weight=0) e fase principale (c_weight>0) -- l'unica differenza tra le due
    fasi e' come viene calcolato c_weight e total_t_max da fuori.
    """
    optimizer.zero_grad()

    total_t = random.randint(1, total_t_max)

    total = 0
    p_total = 0
    c_total = 0
    current_input = generate_instance(batch_size, device, dtype=dtype)
    initial = current_input.clone()  # riferimento fisso per la conservation loss

    for _ in range(total_t):
        # Assicurati che il forward della rete e la loss usino lo stesso dt se necessario
        output = BodyNetwork(current_input, dt) 
        
        # Passiamo net=BodyNetwork alla physics loss per usare direttamente predict_acceleration
        p_loss = physics_loss_2_body(current_input, output, dt=dt, net=BodyNetwork)
        c_loss = conservation_loss(initial, output, eps=1e-2)

        total += p_loss + c_weight * c_loss + a_weight
        p_total += p_loss
        c_total += c_loss

        # Canonizziamo l'output per usarlo come input del prossimo step nel rollout
        current_input, _ = canonicalize_translation(output)

    total /= total_t
    p_total /= total_t
    c_total /= total_t

    if not torch.isfinite(total):
        tqdm.write(f"epoch {i}: loss non finita ({total}), skip step")
        return None

    total.backward()
    torch.nn.utils.clip_grad_norm_(BodyNetwork.parameters(), max_norm=1.0)
    optimizer.step()
    scheduler.step(total.item())

    return total.item(), p_total.item(), c_total.item()


def train(epochs_pretrain: int,
          epochs_main: int,
          BodyNetwork: nn.Module,
          optimizer: torch.optim.Optimizer,
          scheduler,
          device: torch.device = torch.device('cpu'),
          batch_size: int = 256,
          dt: float = 0.01,
          dtype: torch.dtype = torch.float64):
    """
    Training in due fasi.

    Fase 1 (pretraining, epochs_pretrain epoche): loss = SOLO p_loss (c_weight=0),
    rollout corto (total_t max = 5, fisso e piccolo). Obiettivo: imparare la
    dinamica di base -- direzione e modulo dello spostamento per singolo step --
    senza interferenza del termine di conservazione, replicando le condizioni del
    test di overfitting su batch fisso che ha convergo in poche centinaia di
    epoche. Qui i dati SONO variabili (generate_instance ad ogni epoca), quindi
    serve piu' training del test isolato, ma senza la distrazione di c_loss il
    segnale utile arriva alla rete molto piu' pulito fin da subito.

    Fase 2 (principale, epochs_main epoche): loss = p_loss + c_weight*c_loss,
    con c_weight che cresce gradualmente da 0 e curriculum su total_t che
    allunga il rollout fino a 80 step. Qui la rete, gia' partita da una dinamica
    di base corretta, viene raffinata per rispettare anche la conservazione e
    per restare stabile su rollout piu' lunghi.
    """

    loss_history, p_loss_history, c_loss_history = [], [], []

    # Pretraining, solo p_loss, rollout corto
    pbar = tqdm(range(epochs_pretrain), desc="Pretraining (solo p_loss)")
    for i in pbar:
        total_t_max = min(5, 1 + i // 200)

        result = _run_epoch(i, BodyNetwork, optimizer, scheduler, device,
                             batch_size, dt, total_t_max, c_weight=0.0, dtype = dtype, a_weight=0.0)
        if result is None:
            continue

        total, p_loss_val, c_loss_val = result
        loss_history.append(total)
        p_loss_history.append(p_loss_val)
        c_loss_history.append(c_loss_val)

        pbar.set_postfix(p_loss=f"{p_loss_val:.4e}", c_loss=f"{c_loss_val:.4e}",
                          lr=f"{optimizer.param_groups[0]['lr']:.1e}")

    max_horizon = 80
    ramp_fraction = 0.5  # il curriculum arriva al massimo entro il 50% di epochs_main
    ramp_denom = max(1, int(epochs_main * ramp_fraction / max_horizon))

    a_weight = 0.5

    pbar = tqdm(range(epochs_main), desc="Training principale")
    for i in pbar:
        total_t_max = min(max_horizon, 1 + i // ramp_denom)

        c_weight = min(i / 100, 1) / 2

        result = _run_epoch(i, BodyNetwork, optimizer, scheduler, device,
                             batch_size, dt, total_t_max, c_weight=c_weight, dtype=dtype, a_weight=a_weight)
        if result is None:
            continue

        total, p_loss_val, c_loss_val = result
        loss_history.append(total)
        p_loss_history.append(p_loss_val)
        c_loss_history.append(c_loss_val)

        pbar.set_postfix(p_loss=f"{p_loss_val:.4e}", c_loss=f"{c_loss_val:.4e}",
                          lr=f"{optimizer.param_groups[0]['lr']:.1e}")

    return loss_history, p_loss_history, c_loss_history


def generate_instance(batch_size=256,
                      device=torch.device('cpu'),
                      dtype=torch.float64,
                      G=1.0):
    # Masse
    m1 = torch.rand(batch_size, 1, device=device, dtype=dtype) * 1.5 + 0.5
    m2 = torch.rand(batch_size, 1, device=device, dtype=dtype) * 1.5 + 0.5
    M_tot = m1 + m2

    # Distanza casuale r e angolo theta casuale (in tutto il piano, [0, 2*pi))
    dist = torch.rand(batch_size, 1, device=device, dtype=dtype) * 2.0 + 0.5
    theta = torch.rand(batch_size, 1, device=device, dtype=dtype) * 2.0 * np.pi

    # Vettore posizione relativa r_12 = p2 - p1
    rx = dist * torch.cos(theta)
    ry = dist * torch.sin(theta)

    # Posizioni nel sistema del Centro di Massa (CoM in 0,0):
    # m1 * p1 + m2 * p2 = 0  e  p2 - p1 = r12
    # => p1 = -(m2 / M_tot) * r12
    # => p2 =  (m1 / M_tot) * r12
    x1 = -(m2 / M_tot) * rx
    y1 = -(m2 / M_tot) * ry
    x2 =  (m1 / M_tot) * rx
    y2 =  (m1 / M_tot) * ry

    # Velocità orbitali
    v_circ = torch.sqrt(G * M_tot / dist)
    v_factor = torch.rand(batch_size, 1, device=device, dtype=dtype) * 1.5 + 0.3
    v_tangential = v_circ * v_factor
    v_radial = (torch.rand(batch_size, 1, device=device, dtype=dtype) - 0.5) * 1.0 * v_circ

    vx_rel = v_radial * torch.cos(theta) - v_tangential * torch.sin(theta)
    vy_rel = v_radial * torch.sin(theta) + v_tangential * torch.cos(theta)

    # Velocità nel centro di massa (V_cm = 0)
    vx1 = -(m2 / M_tot) * vx_rel
    vy1 = -(m2 / M_tot) * vy_rel
    vx2 =  (m1 / M_tot) * vx_rel
    vy2 =  (m1 / M_tot) * vy_rel

    raw_state = torch.cat([m1, x1, y1, vx1, vy1, m2, x2, y2, vx2, vy2], dim=1)
    
    # Ritorna lo stato canonico (CoM in 0,0)
    state_canon, _ = canonicalize_translation(raw_state)
    return state_canon

def train_network(DEVICE: torch.device = torch.device('cpu')):

    print(f"Device {DEVICE}")

    # Parameters
    n_body = 2
    n_blocks = 4

    epochs_pretrain = 1000
    epochs_main = 3000
    batch_size = 256
    lr = 1e-3
    dt = 0.01

    dtype = torch.float64

    # Network initialization
    BodyNetwork = Acceleration2BodyNetv4(
        num_blocks=n_blocks,
        dtype=dtype,
        device=DEVICE
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

    losses, p_losses, c_losses = train(
        epochs_pretrain,
        epochs_main,
        BodyNetwork,
        optimizer,
        scheduler,
        device=DEVICE,
        batch_size=batch_size,
        dt=dt,
        dtype=dtype
    )

    return BodyNetwork, optimizer, scheduler, epochs_pretrain + epochs_main, losses, p_losses, c_losses


if __name__ == "__main__":

    print("=" * 90)

    PATH = "./PINN_savefile/save_equivariance_acc_v4.pt"

    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)  # fissa anche l'RNG della GPU, non solo quello CPU

    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    net, optimizer, scheduler, epoch, losses, p_losses, c_losses = train_network(DEVICE)
    torch.save({
        "model": net.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "epoch": epoch,
        "losses": losses,
        "p_losses": p_losses,
        "c_losses": c_losses,
    }, PATH)

    plt.plot(losses, label="total", alpha=0.6)
    plt.plot(p_losses, label="p_loss", alpha=0.8)
    plt.plot(c_losses, label="c_loss", alpha=0.8)
    plt.yscale('log')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.legend()
    plt.savefig('loss_curve.png')
