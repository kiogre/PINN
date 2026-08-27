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
from networks_2 import Acceleration2BodyNetv5, Acceleration2BodyNetv6

def canonicalize_translation(state: torch.Tensor):
    """
    Trasla lo stato [Batch, N * 5] in modo che il Centro di Massa (CoM)
    sia esattamente in (0, 0).
    """
    B = state.shape[0]
    x_r = state.view(B, -1, 5) # [Batch, N, 5] -> [m, x, y, vx, vy]

    m = x_r[:, :, 0:1]     # [B, N, 1]
    p = x_r[:, :, 1:3]     # [B, N, 2]
    v = x_r[:, :, 3:5]     # [B, N, 2]

    M_tot = torch.sum(m, dim=1, keepdim=True) # [B, 1, 1]
    R_cm = torch.sum(m * p, dim=1, keepdim=True) / M_tot # [B, 1, 2]

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


def compute_lrl_vector(states, G=1.0, eps=1e-2):
    """Calcola il vettore LRL per ciascun campione nel batch."""
    m1, m2 = states[:, 0:1], states[:, 5:6]
    M_tot = m1 + m2
    
    r = states[:, 6:8] - states[:, 1:3]     # r_12 [B, 2]
    v = states[:, 8:10] - states[:, 3:5]   # v_12 [B, 2]
    
    r_norm = torch.sqrt(torch.sum(r**2, dim=1, keepdim=True) + eps**2)
    
    L_z = r[:, 0:1] * v[:, 1:2] - r[:, 1:2] * v[:, 0:1]
    v_cross_L = torch.cat([v[:, 1:2] * L_z, -v[:, 0:1] * L_z], dim=1)
    
    A = v_cross_L - G * M_tot * (r / r_norm)
    return A


def physics_loss_2_body(input_state, output_state, dt=0.01, G=1.0, eps=1e-3, net=None, g_weight=0.1):
    """
    g-PINN Loss:
    1. Valuta la differenza tra accelerazione appresa e accelerazione teorica (residual loss).
    2. Valuta la differenza tra i GRADIENTI dell'accelerazione appresa e i GRADIENTI 
       dell'accelerazione teorica rispetto alle posizioni (gradient residual loss).
    """
    B = input_state.shape[0]

    # Abilita la registrazione dei gradienti sull'input per g-PINN
    if not input_state.requires_grad:
        input_state = input_state.clone().detach().requires_grad_(True)

    x_r = input_state.view(B, 2, 5)

    m1, m2 = x_r[:, 0, 0:1], x_r[:, 1, 0:1] # [B, 1]
    p1, p2 = x_r[:, 0, 1:3], x_r[:, 1, 1:3] # [B, 2]

    # Vettore distanza relativa r_12 = p2 - p1
    r12 = p2 - p1 # [B, 2]
    dist_sq = torch.sum(r12**2, dim=-1, keepdim=True) + eps**2
    dist = torch.sqrt(dist_sq) # [B, 1]

    # Accelerazione gravitazionale ESATTA (Target di Newton)
    a1_target = G * m2 * r12 / (dist ** 3)
    a2_target = - G * m1 * r12 / (dist ** 3)
    a_target = torch.stack([a1_target, a2_target], dim=1) # [B, 2, 2]

    # Predizione della rete
    if net is not None:
        a_pred = net.predict_acceleration(input_state) # [B, 2, 2] o formato compatibile
    else:
        v_in = x_r[:, :, 3:5]
        v_out = output_state.view(B, 2, 5)[:, :, 3:5]
        a_pred = (v_out - v_in) / dt

    # 1. Standard PINN Loss (Residual)
    loss_res = F.smooth_l1_loss(a_pred, a_target, beta=1e-2)

    # 2. Gradient-enhanced (g-PINN) Component
    # Derivata analitica di a_target rispetto a p1 e p2 (via r12)
    # da/dr = G*m * (I / |r|^3 - 3 * r (r^T) / |r|^5)
    I = torch.eye(2, device=input_state.device, dtype=input_state.dtype).unsqueeze(0) # [1, 2, 2]
    r_outer = torch.bmm(r12.unsqueeze(2), r12.unsqueeze(1)) # [B, 2, 2]
    
    grad_a1_target = G * m2.unsqueeze(-1) * (I / (dist.unsqueeze(-1)**3) - 3 * r_outer / (dist.unsqueeze(-1)**5))
    
    # Calcolo dei gradienti di a_pred rispetto all'input tramite autograd.
    # Teniamo il vettore COMPLETO (10 componenti) invece di tagliare subito
    # a p1: le colonne relative a p2 (indici 6:8) sono già calcolate da questa
    # stessa chiamata, a costo zero, e portano un secondo vincolo fisico.
    grad_a1_x_full = torch.autograd.grad(
        a_pred[:, 0, 0], input_state,
        grad_outputs=torch.ones_like(a_pred[:, 0, 0]),
        create_graph=True, retain_graph=True
    )[0] # [B, 10] -- derivate di a1x rispetto a tutto lo stato

    grad_a1_y_full = torch.autograd.grad(
        a_pred[:, 0, 1], input_state,
        grad_outputs=torch.ones_like(a_pred[:, 0, 1]),
        create_graph=True, retain_graph=True
    )[0] # [B, 10] -- derivate di a1y rispetto a tutto lo stato

    # d(a1)/d(p1): r12 = p2 - p1  =>  d(a1)/d(p1) = -grad_a1_target
    grad_a1_pred_p1 = torch.stack([grad_a1_x_full[:, 1:3], grad_a1_y_full[:, 1:3]], dim=1) # [B, 2, 2]
    # d(a1)/d(p2): stesso ragionamento  =>  d(a1)/d(p2) = +grad_a1_target
    grad_a1_pred_p2 = torch.stack([grad_a1_x_full[:, 6:8], grad_a1_y_full[:, 6:8]], dim=1) # [B, 2, 2]

    # Loss sui gradienti della risposta della rete (Grad-Loss), su entrambi i vincoli
    loss_grad_p1 = F.smooth_l1_loss(grad_a1_pred_p1, -grad_a1_target, beta=1e-2)
    loss_grad_p2 = F.smooth_l1_loss(grad_a1_pred_p2, grad_a1_target, beta=1e-2)
    loss_grad = 0.5 * (loss_grad_p1 + loss_grad_p2)

    total_physics_loss = loss_res + g_weight * loss_grad
    return total_physics_loss


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
    """Penalise RELATIVE energy and angular-momentum drift."""
    E, E0 = compute_energy(output, G, eps), compute_energy(initial, G, eps)
    L, L0 = compute_angular_momentum(output), compute_angular_momentum(initial)

    loss_E = torch.mean(((E - E0) / (E0.abs() + eps)) ** 2)
    loss_L = torch.mean(((L - L0) / (L0.abs() + eps)) ** 2)
    return loss_E + loss_L


def _run_epoch(i, BodyNetwork, optimizer, scheduler, device, batch_size, dt,
               total_t_max, c_weight, dtype, a_weight, g_weight=0.1):
    optimizer.zero_grad()

    total_t = random.randint(1, total_t_max)

    total = 0
    p_total = 0
    c_total = 0
    
    current_input = generate_instance(batch_size, device, dtype=dtype)
    initial = current_input.clone()

    for _ in range(total_t):
        output = BodyNetwork(current_input, dt) 
        
        # Inserita la g_weight per abilitare la perdita sui gradienti della g-PINN
        p_loss = physics_loss_2_body(current_input, output, dt=dt, net=BodyNetwork, g_weight=g_weight)
        c_loss = conservation_loss(initial, output, eps=1e-2)

        A, A0 = compute_lrl_vector(output, eps=1e-2), compute_lrl_vector(initial, eps=1e-2)
        loss_A = torch.mean(((A - A0) / (A0.abs() + 1e-2)) ** 2)

        total += p_loss + c_weight * c_loss + a_weight * loss_A
        p_total += p_loss
        c_total += c_loss

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
          dtype: torch.dtype = torch.float64,
          g_weight: float = 0.1):

    loss_history, p_loss_history, c_loss_history = [], [], []

    pbar = tqdm(range(epochs_pretrain), desc="Pretraining (g-PINN solo p_loss)")
    for i in pbar:
        total_t_max = min(5, 1 + i // 200)

        result = _run_epoch(i, BodyNetwork, optimizer, scheduler, device,
                             batch_size, dt, total_t_max, c_weight=0.0, dtype=dtype, a_weight=0.0, g_weight=g_weight)
        if result is None:
            continue

        total, p_loss_val, c_loss_val = result
        loss_history.append(total)
        p_loss_history.append(p_loss_val)
        c_loss_history.append(c_loss_val)

        pbar.set_postfix(p_loss=f"{p_loss_val:.4e}", c_loss=f"{c_loss_val:.4e}",
                          lr=f"{optimizer.param_groups[0]['lr']:.1e}")

    max_horizon = 80
    ramp_fraction = 0.5
    ramp_denom = max(1, int(epochs_main * ramp_fraction / max_horizon))
    a_weight = 0.5

    pbar = tqdm(range(epochs_main), desc="Training principale (g-PINN)")
    for i in pbar:
        total_t_max = min(max_horizon, 1 + i // ramp_denom)
        c_weight = min(i / 100, 1) / 2

        result = _run_epoch(i, BodyNetwork, optimizer, scheduler, device,
                             batch_size, dt, total_t_max, c_weight=c_weight, dtype=dtype, a_weight=a_weight, g_weight=g_weight)
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
    m1 = torch.rand(batch_size, 1, device=device, dtype=dtype) * 1.5 + 0.5
    m2 = torch.rand(batch_size, 1, device=device, dtype=dtype) * 1.5 + 0.5
    M_tot = m1 + m2

    dist = torch.rand(batch_size, 1, device=device, dtype=dtype) * 2.0 + 0.5
    theta = torch.rand(batch_size, 1, device=device, dtype=dtype) * 2.0 * np.pi

    rx = dist * torch.cos(theta)
    ry = dist * torch.sin(theta)

    x1 = -(m2 / M_tot) * rx
    y1 = -(m2 / M_tot) * ry
    x2 =  (m1 / M_tot) * rx
    y2 =  (m1 / M_tot) * ry

    v_circ = torch.sqrt(G * M_tot / dist)
    v_factor = torch.rand(batch_size, 1, device=device, dtype=dtype) * 1.5 + 0.3
    v_tangential = v_circ * v_factor
    v_radial = (torch.rand(batch_size, 1, device=device, dtype=dtype) - 0.5) * 1.0 * v_circ

    vx_rel = v_radial * torch.cos(theta) - v_tangential * torch.sin(theta)
    vy_rel = v_radial * torch.sin(theta) + v_tangential * torch.cos(theta)

    vx1 = -(m2 / M_tot) * vx_rel
    vy1 = -(m2 / M_tot) * vy_rel
    vx2 =  (m1 / M_tot) * vx_rel
    vy2 =  (m1 / M_tot) * vy_rel

    raw_state = torch.cat([m1, x1, y1, vx1, vy1, m2, x2, y2, vx2, vy2], dim=1)
    state_canon, _ = canonicalize_translation(raw_state)
    return state_canon


def train_network(DEVICE: torch.device = torch.device('cpu')):

    print(f"Device {DEVICE}")

    n_blocks = 4
    epochs_pretrain = 1000
    epochs_main = 3000
    batch_size = 256
    lr = 1e-3
    dt = 0.01
    g_weight = 0.1 # Peso per il termine di gradient-enhanced loss

    dtype = torch.float64

    BodyNetwork = Acceleration2BodyNetv6(
        num_blocks=n_blocks,
        dtype=dtype,
        device=DEVICE
    ).to(DEVICE)

    optimizer = torch.optim.Adam(
        BodyNetwork.parameters(),
        lr=lr
    )

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.5,
        patience=50,
        min_lr=1e-6
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
        dtype=dtype,
        g_weight=g_weight
    )

    return BodyNetwork, optimizer, scheduler, epochs_pretrain + epochs_main, losses, p_losses, c_losses


if __name__ == "__main__":

    print("=" * 90)

    PATH = "./PINN_savefile/save_equivariance_acc_v6_gpinn.pt"

    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)

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
    plt.plot(p_losses, label="p_loss (g-PINN)", alpha=0.8)
    plt.plot(c_losses, label="c_loss", alpha=0.8)
    plt.yscale('log')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.legend()
    plt.savefig('loss_curve_gpinn.png')