import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from networks import AccelerationNBodyNetv4
import random
from tqdm import tqdm


def canonicalize_translation(state: torch.Tensor):
    """
    Trasla lo stato [Batch, N * 5] in modo che il Centro di Massa (CoM)
    sia esattamente in (0, 0).
    """
    B = state.shape[0]
    x_r = state.view(B, -1, 5)

    m = x_r[:, :, 0:1]
    p = x_r[:, :, 1:3]
    v = x_r[:, :, 3:5]

    M_tot = torch.sum(m, dim=1, keepdim=True)
    R_cm = torch.sum(m * p, dim=1, keepdim=True) / M_tot

    p_canon = p - R_cm

    state_canon = torch.cat([m, p_canon, v], dim=-1).view(B, -1)
    return state_canon, R_cm


def uncanonicalize_translation(state_canon: torch.Tensor, R_cm: torch.Tensor):
    B = state_canon.shape[0]
    x_r = state_canon.view(B, -1, 5)

    m = x_r[:, :, 0:1]
    p_canon = x_r[:, :, 1:3]
    v = x_r[:, :, 3:5]

    p_orig = p_canon + R_cm

    state_orig = torch.cat([m, p_orig, v], dim=-1).view(B, -1)
    return state_orig


def min_pairwise_distance(state):
    """Distanza minima tra coppie di corpi, per ogni sample nel batch."""
    B = state.shape[0]
    N = state.shape[1] // 5
    p = state.view(B, N, 5)[:, :, 1:3]
    diff = p.unsqueeze(1) - p.unsqueeze(2)          # [B, N, N, 2]
    dist = torch.sqrt(torch.sum(diff ** 2, dim=-1))  # [B, N, N]
    mask = 1.0 - torch.eye(N, device=state.device, dtype=state.dtype)
    dist_masked = dist * mask + (1.0 - mask) * 1e6   
    return dist_masked.view(B, -1).min(dim=1).values  # [B]


def physics_loss_n_body(input_state, output_state, dt=0.01, G=1.0, eps=1e-3, net=None, g_weight=0.1):
    """
    g-PINN per N-Body:
    - Standard Residual Loss sull'accelerazione (a_pred - a_target).
    - Gradient Residual Loss sulla derivata spaziale dell'accelerazione (da/dp).
    """
    B = input_state.shape[0]
    N = input_state.shape[1] // 5

    # Abilitiamo la derivazione rispetto all'input per la g-PINN
    if not input_state.requires_grad:
        input_state = input_state.clone().detach().requires_grad_(True)

    x_r = input_state.view(B, N, 5)
    m = x_r[:, :, 0:1]   # [B, N, 1]
    p = x_r[:, :, 1:3]   # [B, N, 2]

    # diff[b, i, j] = p_j - p_i
    diff = p.unsqueeze(1) - p.unsqueeze(2)               # [B, N, N, 2]
    dist_sq = torch.sum(diff ** 2, dim=-1, keepdim=True) + eps ** 2 # [B, N, N, 1]
    dist = torch.sqrt(dist_sq)                           # [B, N, N, 1]

    # Target Accelerazione di Newton
    m_j = m.view(B, 1, N, 1)  # massa del corpo j
    a_target = G * torch.sum(m_j * diff / (dist ** 3), dim=2)  # [B, N, 2]

    # Predizione dell'accelerazione dalla rete
    if net is not None:
        a_pred = net.predict_acceleration(input_state)   # [B, N, 2]
    else:
        v_in = x_r[:, :, 3:5]
        v_out = output_state.view(B, N, 5)[:, :, 3:5]
        a_pred = (v_out - v_in) / dt

    # Residual loss standard
    loss_res = F.smooth_l1_loss(a_pred, a_target, beta=1e-2)

    # Target del Gradiente Analitico (da_i / dp_i)
    # Matrice identita 2x2
    I = torch.eye(2, device=input_state.device, dtype=input_state.dtype).view(1, 1, 1, 2, 2)
    diff_outer = torch.matmul(diff.unsqueeze(-1), diff.unsqueeze(-2)) # [B, N, N, 2, 2]
    
    # Derivata del contributo di gravità di j su i rispetto a p_i
    grad_term = G * m_j.unsqueeze(-1) * (I / (dist.unsqueeze(-1)**3) - 3 * diff_outer / (dist.unsqueeze(-1)**5))
    mask = (1.0 - torch.eye(N, device=input_state.device, dtype=input_state.dtype)).view(1, N, N, 1, 1)
    
    # Derivata analitica totale di a_i rispetto a p_i
    grad_a_target_diag = -torch.sum(grad_term * mask, dim=2) # [B, N, 2, 2]

    # Gradiente Predetto dalla Rete via Autograd
    # Calcoliamo la derivata di ciascuna componente x, y dell'accelerazione di ogni corpo
    grad_a_pred_list = []
    for body_idx in range(N):
        # Componente x
        grad_x = torch.autograd.grad(
            a_pred[:, body_idx, 0], input_state,
            grad_outputs=torch.ones_like(a_pred[:, body_idx, 0]),
            create_graph=True, retain_graph=True
        )[0].view(B, N, 5)[:, body_idx, 1:3] # derivate rispetto a (x_i, y_i)

        # Componente y
        grad_y = torch.autograd.grad(
            a_pred[:, body_idx, 1], input_state,
            grad_outputs=torch.ones_like(a_pred[:, body_idx, 1]),
            create_graph=True, retain_graph=True
        )[0].view(B, N, 5)[:, body_idx, 1:3] # derivate rispetto a (x_i, y_i)

        grad_a_pred_body = torch.stack([grad_x, grad_y], dim=1) # [B, 2, 2]
        grad_a_pred_list.append(grad_a_pred_body)

    grad_a_pred = torch.stack(grad_a_pred_list, dim=1) # [B, N, 2, 2]

    # Gradient Residual Loss
    loss_grad = F.smooth_l1_loss(grad_a_pred, grad_a_target_diag, beta=1e-2)

    total_loss = loss_res + g_weight * loss_grad
    return total_loss


def compute_energy(states, G=1.0, eps=1e-3):
    """Energia totale E = T + V."""
    B = states.shape[0]
    N = states.shape[1] // 5
    x_r = states.view(B, N, 5)

    m = x_r[:, :, 0]     # [B, N]
    p = x_r[:, :, 1:3]   # [B, N, 2]
    v = x_r[:, :, 3:5]   # [B, N, 2]

    T = 0.5 * torch.sum(m * torch.sum(v ** 2, dim=-1), dim=-1)  # [B]

    diff = p.unsqueeze(1) - p.unsqueeze(2)                       # [B, N, N, 2]
    dist = torch.sqrt(torch.sum(diff ** 2, dim=-1) + eps ** 2)   # [B, N, N]
    m_ij = m.unsqueeze(1) * m.unsqueeze(2)                       # [B, N, N]

    mask = 1.0 - torch.eye(N, device=states.device, dtype=states.dtype)
    V_pairs = -G * m_ij / dist * mask
    V = 0.5 * torch.sum(V_pairs, dim=(1, 2))

    return T + V


def compute_angular_momentum(states):
    """Momento angolare totale L = sum_i m_i*(x_i*vy_i - y_i*vx_i)."""
    B = states.shape[0]
    N = states.shape[1] // 5
    x_r = states.view(B, N, 5)

    m = x_r[:, :, 0]
    p = x_r[:, :, 1:3]
    v = x_r[:, :, 3:5]

    L = torch.sum(m * (p[:, :, 0] * v[:, :, 1] - p[:, :, 1] * v[:, :, 0]), dim=-1)
    return L


def conservation_loss(initial, output, G=1.0, eps=1e-2):
    """Penalizzazione del drift di energia e momento angolare."""
    E, E0 = compute_energy(output, G, eps), compute_energy(initial, G, eps)
    L, L0 = compute_angular_momentum(output), compute_angular_momentum(initial)

    loss_E = torch.mean(((E - E0) / (E0.abs() + eps)) ** 2)
    loss_L = torch.mean(((L - L0) / (L0.abs() + eps)) ** 2)
    return loss_E + loss_L


def _run_epoch(i, BodyNetwork, optimizer, scheduler, device, batch_size, dt,
               total_t_max, c_weight, dtype, n_obj, g_weight=0.1):
    optimizer.zero_grad()

    total_t = random.randint(1, total_t_max)

    total = 0
    p_total = 0
    c_total = 0
    current_input = generate_instance(batch_size, n_obj, device, dtype=dtype)
    initial = current_input.clone()

    for _ in range(total_t):
        output = BodyNetwork(current_input, dt)

        # Integrazione della physics_loss_n_body in configurazione g-PINN
        p_loss = physics_loss_n_body(current_input, output, dt=dt, net=BodyNetwork, g_weight=g_weight)
        c_loss = conservation_loss(initial, output, eps=1e-2)

        total += p_loss + c_weight * c_loss
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
          n_obj: int,
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
                             batch_size, dt, total_t_max, c_weight=0.0, dtype=dtype, n_obj=n_obj, g_weight=g_weight)
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

    pbar = tqdm(range(epochs_main), desc="Training principale (g-PINN)")
    for i in pbar:
        total_t_max = min(max_horizon, 1 + i // ramp_denom)
        c_weight = min(i / 100, 1) / 2

        result = _run_epoch(i, BodyNetwork, optimizer, scheduler, device,
                             batch_size, dt, total_t_max, c_weight=c_weight, dtype=dtype, n_obj=n_obj, g_weight=g_weight)
        if result is None:
            continue

        total, p_loss_val, c_loss_val = result
        loss_history.append(total)
        p_loss_history.append(p_loss_val)
        c_loss_history.append(c_loss_val)

        pbar.set_postfix(p_loss=f"{p_loss_val:.4e}", c_loss=f"{c_loss_val:.4e}",
                          lr=f"{optimizer.param_groups[0]['lr']:.1e}")

    return loss_history, p_loss_history, c_loss_history


def generate_instance(batch_size, n_obj, device=torch.device('cpu'), dtype=torch.float64, G=1.0):
    """Generatore di stati iniziali per N corpi."""
    m = torch.rand(batch_size, n_obj, 1, device=device, dtype=dtype) * 1.5 + 0.5  
    M_tot = torch.sum(m, dim=1, keepdim=True)  

    shell_idx = torch.arange(n_obj, device=device, dtype=dtype).view(1, n_obj, 1) + 1.0  
    r_scale = torch.rand(batch_size, 1, 1, device=device, dtype=dtype) * 0.5 + 0.7        
    radius = shell_idx * r_scale + torch.rand(batch_size, n_obj, 1, device=device, dtype=dtype) * 0.3
    angle = torch.rand(batch_size, n_obj, 1, device=device, dtype=dtype) * 2.0 * np.pi

    p = torch.cat([radius * torch.cos(angle), radius * torch.sin(angle)], dim=-1)  

    v_circ = torch.sqrt(G * M_tot / radius)                                        
    v_factor = torch.rand(batch_size, n_obj, 1, device=device, dtype=dtype) * 1.0 + 0.4
    tangential_dir = torch.cat([-torch.sin(angle), torch.cos(angle)], dim=-1)      
    radial_dir = torch.cat([torch.cos(angle), torch.sin(angle)], dim=-1)
    v_radial_factor = (torch.rand(batch_size, n_obj, 1, device=device, dtype=dtype) - 0.5) * 0.6

    v = v_circ * v_factor * tangential_dir + v_circ * v_radial_factor * radial_dir  

    v_cm = torch.sum(m * v, dim=1, keepdim=True) / M_tot
    v = v - v_cm

    raw_state = torch.cat([m, p, v], dim=-1).view(batch_size, -1)  

    state_canon, _ = canonicalize_translation(raw_state)  
    return state_canon


def train_network(DEVICE: torch.device = torch.device('cpu'), n_body: int = 3):

    print(f"Device {DEVICE}, N corpi = {n_body}")

    n_blocks = 4
    epochs_pretrain = 1000
    epochs_main = 3000
    batch_size = 128
    lr = 1e-3
    dt = 0.01
    g_weight = 0.1 # Peso g-PINN
    dtype = torch.float64

    BodyNetwork = AccelerationNBodyNetv4(
        n_obj=n_body,
        num_blocks=n_blocks,
        dtype=dtype,
        device=DEVICE
    ).to(DEVICE)

    optimizer = torch.optim.Adam(BodyNetwork.parameters(), lr=lr)

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=50, min_lr=1e-6
    )

    losses, p_losses, c_losses = train(
        epochs_pretrain,
        epochs_main,
        BodyNetwork,
        optimizer,
        scheduler,
        n_obj=n_body,
        device=DEVICE,
        batch_size=batch_size,
        dt=dt,
        dtype=dtype,
        g_weight=g_weight
    )

    return BodyNetwork, optimizer, scheduler, epochs_pretrain + epochs_main, losses, p_losses, c_losses


if __name__ == "__main__":

    print("=" * 90)

    N_BODY = 3
    PATH = f"./PINN_savefile/save_nbody_{N_BODY}_gpinn_acc.pt"

    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)

    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    net, optimizer, scheduler, epoch, losses, p_losses, c_losses = train_network(DEVICE, n_body=N_BODY)
    torch.save({
        "model": net.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "epoch": epoch,
        "n_body": N_BODY,
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
    plt.savefig('loss_curve_nbody_gpinn.png')