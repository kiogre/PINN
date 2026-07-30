import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from networks import AB2Net, FullyEquivariant2BodyNet
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

def canonicalize_translation(state: torch.Tensor):
    """
    Trasla lo stato [Batch, N * 5] in modo che il corpo più a sinistra (min x)
    vada in (0, 0). Tutti gli altri corpi avranno x >= 0 (semipiano destro).
    Nessuna rotazione applicata.
    
    Restituisce:
    - state_canon: lo stato traslato
    - p_min: il vettore di traslazione [Batch, 1, 2] per poter tornare indietro
    """
    B = state.shape[0]
    x_r = state.view(B, -1, 5) # [Batch, N, 5] -> [m, x, y, vx, vy]

    m = x_r[:, :, 0:1]     # [B, N, 1]
    p = x_r[:, :, 1:3]     # [B, N, 2]
    v = x_r[:, :, 3:5]     # [B, N, 2] (le velocità NON cambiano per pura traslazione)

    # Trova la posizione del corpo con coordinata x minima
    idx_min = torch.argmin(p[:, :, 0], dim=1)                # [B]
    p_min = p[torch.arange(B), idx_min].unsqueeze(1)         # [B, 1, 2]

    # Traslazione delle posizioni: il nodo min x diventa (0, 0)
    p_canon = p - p_min  

    state_canon = torch.cat([m, p_canon, v], dim=-1).view(B, -1)
    return state_canon, p_min


def uncanonicalize_translation(state_canon: torch.Tensor, p_min: torch.Tensor):
    """
    Ripristina le posizioni originali sommando indietro p_min.
    """
    B = state_canon.shape[0]
    x_r = state_canon.view(B, -1, 5)

    m = x_r[:, :, 0:1]
    p_canon = x_r[:, :, 1:3]
    v = x_r[:, :, 3:5]

    p_orig = p_canon + p_min

    state_orig = torch.cat([m, p_orig, v], dim=-1).view(B, -1)
    return state_orig

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


def conservation_loss(initial, output, G=1.0, eps=1e-3):
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
               total_t_max, c_weight, dtype):
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
        output = BodyNetwork(current_input)
        
        p_loss = physics_loss_2_body(current_input, output, dt)
        c_loss = conservation_loss(initial, output)

        total += p_loss + c_weight * c_loss
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
          dt: float = 0.05,
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
    allunga il rollout fino a 30 step. Qui la rete, gia' partita da una dinamica
    di base corretta, viene raffinata per rispettare anche la conservazione e
    per restare stabile su rollout piu' lunghi.
    """

    loss_history, p_loss_history, c_loss_history = [], [], []

    # ------------------------------------------------------------
    # Fase 1: pretraining, solo p_loss, rollout corto
    # ------------------------------------------------------------
    pbar = tqdm(range(epochs_pretrain), desc="Pretraining (solo p_loss)")
    for i in pbar:
        total_t_max = min(5, 1 + i // 200)

        result = _run_epoch(i, BodyNetwork, optimizer, scheduler, device,
                             batch_size, dt, total_t_max, c_weight=0.0, dtype = dtype)
        if result is None:
            continue

        total, p_loss_val, c_loss_val = result
        loss_history.append(total)
        p_loss_history.append(p_loss_val)
        c_loss_history.append(c_loss_val)

        pbar.set_postfix(p_loss=f"{p_loss_val:.4e}", c_loss=f"{c_loss_val:.4e}",
                          lr=f"{optimizer.param_groups[0]['lr']:.1e}")

    # ------------------------------------------------------------
    # Fase 2: training principale, p_loss + c_weight*c_loss, curriculum esteso
    # ------------------------------------------------------------
    # total_t_max sale fino a 80 (non piu' 30): con la base di direzione gia'
    # solida dal pretraining, possiamo permetterci rollout piu' lunghi senza
    # ripartire da una direzione rumorosa.
    #
    # c_weight sale fino a 1/2 (non piu' 1/5) e viene scalato anche in base a
    # quanto e' lungo il rollout dell'epoca corrente: sui rollout piu' lunghi
    # (dove la deriva di fase secolare pesa di piu') la conservazione conta
    # relativamente di piu', spingendo la rete a restare fisicamente coerente
    # anche oltre l'orizzonte breve.
    # Il curriculum deve arrivare all'orizzonte massimo con margine sufficiente
    # PRIMA della fine di epochs_main, cosi' resta abbastanza training a rollout
    # lungo (non solo l'istante in cui lo si raggiunge). Il denominatore e' ora
    # derivato da epochs_main invece che fisso, cosi' i due non possono
    # disallinearsi come e' successo con epochs_main=2000 e ramp fisso a //40
    # (che raggiungeva total_t_max=80 solo a epoca ~3160, mai vista qui).
    max_horizon = 80
    ramp_fraction = 0.5  # il curriculum arriva al massimo entro il 50% di epochs_main
    ramp_denom = max(1, int(epochs_main * ramp_fraction / max_horizon))

    pbar = tqdm(range(epochs_main), desc="Training principale")
    for i in pbar:
        total_t_max = min(max_horizon, 1 + i // ramp_denom)

        # c_weight non e' piu' scalato dalla lunghezza del rollout corrente:
        # quello scaling penalizzava proprio la fase in cui total_t_max era
        # ancora sotto 30, riducendo il vincolo di conservazione quando invece
        # avrebbe dovuto restare quello (gia' validato) della ricetta precedente.
        c_weight = min(i / 100, 1) / 2

        result = _run_epoch(i, BodyNetwork, optimizer, scheduler, device,
                             batch_size, dt, total_t_max, c_weight=c_weight, dtype=dtype)
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
    # 1. Masse
    m1 = torch.rand(batch_size, 1, device=device, dtype=dtype) * 1.5 + 0.5
    m2 = torch.rand(batch_size, 1, device=device, dtype=dtype) * 1.5 + 0.5
    M_tot = m1 + m2

    # 2. Corpo 1 in (0,0)
    x1 = torch.zeros(batch_size, 1, device=device, dtype=dtype)
    y1 = torch.zeros(batch_size, 1, device=device, dtype=dtype)

    # 3. Distanza casuale r e angolo theta casuale nel semipiano destro (-pi/2 < theta < pi/2)
    dist = torch.rand(batch_size, 1, device=device, dtype=dtype) * 2.0 + 0.5
    theta = (torch.rand(batch_size, 1, device=device, dtype=dtype) - 0.5) * np.pi * 0.9  # in (-pi/2, pi/2)

    x2 = dist * torch.cos(theta)  # Sempre > 0 per costruzione!
    y2 = dist * torch.sin(theta)  # Può essere sia positivo che negativo

    # 4. Velocità orbitali relative coerenti con l'angolo theta
    v_circ = torch.sqrt(G * M_tot / dist)
    ecc = torch.rand(batch_size, 1, device=device, dtype=dtype) * 0.2 + 0.8
    v_tangential = v_circ * ecc
    v_radial = (torch.rand(batch_size, 1, device=device, dtype=dtype) - 0.5) * 0.1 * v_circ

    # Ruotiamo i vettori di velocità radiale/tangenziale secondo theta
    # Per una coordinata polare (r, theta):
    # e_r = (cos theta, sin theta), e_theta = (-sin theta, cos theta)
    vx_rel = v_radial * torch.cos(theta) - v_tangential * torch.sin(theta)
    vy_rel = v_radial * torch.sin(theta) + v_tangential * torch.cos(theta)

    # Velocità nel centro di massa
    vx1 = -(m2 / M_tot) * vx_rel
    vy1 = -(m2 / M_tot) * vy_rel
    vx2 = (m1 / M_tot) * vx_rel
    vy2 = (m1 / M_tot) * vy_rel

    raw_state = torch.cat([m1, x1, y1, vx1, vy1, m2, x2, y2, vx2, vy2], dim=1)
    
    # Assicuriamo la canonizzazione traslazionale
    state_canon, _ = canonicalize_translation(raw_state)
    return state_canon


def train_network(DEVICE: torch.device = torch.device('cpu')):

    print(f"Device {DEVICE}")

    # Parameters
    n_body = 2
    n_blocks = 4

    epochs_pretrain = 1000
    epochs_main = 2000
    batch_size = 256
    lr = 1e-3
    dt = 0.05

    dtype = torch.float64

    # Network initialization
    BodyNetwork = FullyEquivariant2BodyNet(
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

    PATH = "./PINN_savefile/save_equivariance.pt"

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
