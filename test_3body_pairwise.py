import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.animation import FuncAnimation
from scipy.integrate import solve_ivp

# Importiamo le reti dai vostri moduli esistenti
from networks import Acceleration2BodyNetv4, AccelerationNBodyNetv4
from networks_2 import Acceleration2BodyNetv5, AccelerationNBodyNetv5

# Importiamo le funzioni di utilità da train_nbody.py
from train_nbody import (
    generate_instance,
    compute_energy,
    compute_angular_momentum,
    canonicalize_translation,
    uncanonicalize_translation,
)


def compute_momentum(states):
    """Momento lineare totale per N corpi: p = sum(m_i * v_i)."""
    B = states.shape[0]
    N = states.shape[1] // 5
    x_r = states.view(B, N, 5)
    m = x_r[:, :, 0:1]
    v = x_r[:, :, 3:5]
    return torch.sum(m * v, dim=1)  # [B, 2]


class PairwiseWrapperNBody(nn.Module):
    """
    Wrapper che prende un modello a 2 corpi (es. Acceleration2BodyNetv4 o v5)
    e lo applica a coppie per simulare un sistema a N corpi (in questo caso N=3)
    sfruttando il principio di sovrapposizione lineare delle forze gravitazionali.
    """
    def __init__(self, net_2body, n_obj=3, dt=0.01):
        super().__init__()
        self.net_2body = net_2body
        self.n_obj = n_obj
        self.dt = dt

    def predict_acceleration(self, state: torch.Tensor) -> torch.Tensor:
        """
        Calcola l'accelerazione per ciascuno degli N corpi sommando le interazioni a coppie.
        State: [B, N * 5]
        """
        B = state.shape[0]
        N = self.n_obj
        x_r = state.view(B, N, 5)

        m = x_r[:, :, 0:1]  # [B, N, 1]
        p = x_r[:, :, 1:3]  # [B, N, 2]
        v = x_r[:, :, 3:5]  # [B, N, 2]

        # Inizializziamo le accelerazioni totali a zero per tutti gli N corpi
        a_total = torch.zeros((B, N, 2), dtype=state.dtype, device=state.device)

        # Iteriamo su tutte le coppie possibili (i, j) con i < j
        for i in range(N):
            for j in range(i + 1, N):
                # Ricostruiamo lo stato fittizio a 2 corpi per la coppia (i, j)
                state_pair = torch.cat([
                    m[:, i], p[:, i], v[:, i],
                    m[:, j], p[:, j], v[:, j]
                ], dim=-1)  # [B, 10]

                # Prediciamo l'accelerazione usando la rete a 2 corpi
                a_pair = self.net_2body.predict_acceleration(state_pair)  # [B, 2, 2]

                # Accumuliamo i contributi vettoriali rispettivi
                a_total[:, i] += a_pair[:, 0]
                a_total[:, j] += a_pair[:, 1]

        # Correzione Hard opzionale del Centro di Massa (Terzo Principio di Newton)
        M_tot = torch.sum(m, dim=1, keepdim=True)
        a_cm = torch.sum(m * a_total, dim=1, keepdim=True) / M_tot
        return a_total - a_cm

    def forward(self, x: torch.Tensor, dt: float = 0.01) -> torch.Tensor:
        """
        Evolve lo stato di dt usando lo schema di integrazione Velocity Verlet,
        coerentemente con le altre reti del progetto.
        """
        B = x.shape[0]
        N = self.n_obj
        x_r = x.view(B, N, 5)

        m = x_r[:, :, 0:1]
        p = x_r[:, :, 1:3]
        v = x_r[:, :, 3:5]

        # 1. Accelerazione al tempo t
        a_t = self.predict_acceleration(x)

        # 2. Aggiornamento delle posizioni: p(t+dt)
        p_next = p + v * dt + 0.5 * a_t * (dt ** 2)

        # 3. Stato intermedio per calcolare l'accelerazione futura
        state_temp = torch.cat([m, p_next, v], dim=-1).view(B, -1)
        a_next = self.predict_acceleration(state_temp)

        # 4. Aggiornamento delle velocità: v(t+dt)
        v_next = v + 0.5 * (a_t + a_next) * dt

        # Ricostruzione dello stato nel formato [B, N * 5]
        out_r = torch.cat([m, p_next, v_next], dim=-1)
        return out_r.view(B, -1)


# Solver di riferimento per N corpi (DOP853)
def n_body_rhs(t, state, masses, G=1.0, eps=1e-3):
    N = len(masses)
    pos = state[:2 * N].reshape(N, 2)
    vel = state[2 * N:].reshape(N, 2)

    acc = np.zeros((N, 2))
    for i in range(N):
        for j in range(N):
            if i == j:
                continue
            rij = pos[j] - pos[i]
            dist = np.sqrt(np.sum(rij ** 2) + eps ** 2)
            acc[i] += G * masses[j] * rij / dist ** 3

    return np.concatenate([vel.reshape(-1), acc.reshape(-1)])


def solve_nbody(state0, masses, dt, n_points, G=1.0):
    t_eval = np.arange(n_points) * dt
    sol = solve_ivp(
        n_body_rhs,
        (0, t_eval[-1]),
        state0,
        args=(masses, G),
        t_eval=t_eval,
        method="DOP853",
        rtol=1e-12,
        atol=1e-14,
    )
    return sol.t, sol.y.T


def test_pairwise_model(ModelWrapper, n_obj=3, device=torch.device("cpu"), rollout_steps=100, dt=0.01, dtype=torch.float64):
    ModelWrapper.eval()

    # Generiamo un'istanza iniziale standard a N corpi
    state = generate_instance(1, n_obj, device, dtype=dtype)
    # known problem 1
    # state = torch.tensor([[
    #     3.0, 1.0, 3.0, 0.0, 0.0,
    #     4.0, -2.0, -1.0, 0.0, 0.0,
    #     5.0, 1.0, -1.0, 0.0, 0.0
    # ]], dtype=torch.float64)

    # known problem 2
    # state = torch.tensor([[
    #     1.0, 1.0, 0.0, 0.0, np.sqrt(1/np.sqrt(3)),
    #     1.0, -0.5, np.sqrt(3)/2, -np.sqrt(3)/2 * np.sqrt(1/np.sqrt(3)), -0.5 * np.sqrt(1/np.sqrt(3)),
    #     1.0, -0.5, -np.sqrt(3)/2, np.sqrt(3)/2 * np.sqrt(1/np.sqrt(3)), -0.5 * np.sqrt(1/np.sqrt(3))
    # ]], dtype=torch.float64)

    # known problem 3
    state = torch.tensor([[
        1.0, 0.0, 0.0, -0.93240737, -0.86473146,
        1.0, 0.97000436, -0.24308753, 0.46620369, 0.43236573,
        1.0, -0.97000436, 0.24308753, 0.46620369, 0.43236573,
    ]], dtype=torch.float64)

    traj_net = []
    with torch.no_grad():
        s_abs = state.clone()
        for _ in range(rollout_steps):
            traj_net.append(s_abs.squeeze(0).cpu())
            s_canon, p_min = canonicalize_translation(s_abs)
            out_canon = ModelWrapper(s_canon, dt)
            s_abs = uncanonicalize_translation(out_canon, p_min)

    # Preparazione del solver di riferimento
    state_np = state.squeeze(0).detach().cpu().numpy().reshape(n_obj, 5)
    masses = state_np[:, 0]
    pos0 = state_np[:, 1:3].reshape(-1)
    vel0 = state_np[:, 3:5].reshape(-1)
    state0 = np.concatenate([pos0, vel0])

    _, sol = solve_nbody(state0, masses, dt=dt, n_points=rollout_steps)

    traj_solver = []
    for row in sol:
        pos = row[:2 * n_obj].reshape(n_obj, 2)
        vel = row[2 * n_obj:].reshape(n_obj, 2)
        body_rows = [torch.tensor([masses[k], pos[k, 0], pos[k, 1], vel[k, 0], vel[k, 1]], dtype=dtype)
                     for k in range(n_obj)]
        traj_solver.append(torch.cat(body_rows))

    # Calcolo dei drift fisici
    E0 = compute_energy(traj_net[0].unsqueeze(0))
    L0 = compute_angular_momentum(traj_net[0].unsqueeze(0))
    p0 = compute_momentum(traj_net[0].unsqueeze(0))

    drift_energy, drift_L, drift_p = [], [], []
    for s in traj_net:
        s_b = s.unsqueeze(0)
        drift_energy.append(torch.mean((compute_energy(s_b) - E0) ** 2).item())
        drift_L.append(torch.mean((compute_angular_momentum(s_b) - L0) ** 2).item())
        drift_p.append(torch.mean(torch.sum((compute_momentum(s_b) - p0) ** 2, dim=-1)).item())

    return {
        "traj_net": traj_net,
        "traj_solver": traj_solver,
        "energy_drift": drift_energy,
        "angular_drift": drift_L,
        "momentum_drift": drift_p,
        "n_obj": n_obj,
    }


def compute_stepwise_error(results):
    traj_net, traj_solver, n_obj = results["traj_net"], results["traj_solver"], results["n_obj"]
    n = min(len(traj_net), len(traj_solver))

    pos_err = [[] for _ in range(n_obj)]
    vel_err = [[] for _ in range(n_obj)]

    for i in range(n):
        sn = traj_net[i].view(n_obj, 5)
        ss = traj_solver[i].view(n_obj, 5)
        for k in range(n_obj):
            pos_err[k].append(torch.norm(sn[k, 1:3] - ss[k, 1:3]).item())
            vel_err[k].append(torch.norm(sn[k, 3:5] - ss[k, 3:5]).item())

    return {
        "pos_err": [np.array(e) for e in pos_err],
        "vel_err": [np.array(e) for e in vel_err],
        "n_obj": n_obj,
    }


def animate_trajectory(traj_net, traj_solver, n_obj, interval=30, save_path=None):
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.set_aspect("equal")

    colors = [cm.tab10(k % 10) for k in range(n_obj)]
    xs, ys = [], []
    for s in traj_net + traj_solver:
        s_r = s.view(n_obj, 5)
        xs.extend(s_r[:, 1].tolist())
        ys.extend(s_r[:, 2].tolist())

    margin = 0.5
    ax.set_xlim(min(xs) - margin, max(xs) + margin)
    ax.set_ylim(min(ys) - margin, max(ys) + margin)
    ax.grid(True)

    solver_lines, solver_dots, net_lines, net_dots = [], [], [], []
    for k in range(n_obj):
        sl, = ax.plot([], [], "-", color=colors[k], lw=2, label=f"Body {k+1} Solver")
        sd, = ax.plot([], [], "o", color=colors[k], ms=8)
        nl, = ax.plot([], [], "--", color=colors[k], lw=2, alpha=0.6, label=f"Body {k+1} Pairwise Net")
        nd, = ax.plot([], [], "s", color=colors[k], ms=6)
        solver_lines.append(sl); solver_dots.append(sd)
        net_lines.append(nl); net_dots.append(nd)

    ax.legend(fontsize=8)

    def update(frame):
        artists = []
        for k in range(n_obj):
            xs_s = [traj_solver[i].view(n_obj, 5)[k, 1].item() for i in range(frame + 1)]
            ys_s = [traj_solver[i].view(n_obj, 5)[k, 2].item() for i in range(frame + 1)]
            solver_lines[k].set_data(xs_s, ys_s)
            solver_dots[k].set_data([xs_s[-1]], [ys_s[-1]])

            xs_n = [traj_net[i].view(n_obj, 5)[k, 1].item() for i in range(frame + 1)]
            ys_n = [traj_net[i].view(n_obj, 5)[k, 2].item() for i in range(frame + 1)]
            net_lines[k].set_data(xs_n, ys_n)
            net_dots[k].set_data([xs_n[-1]], [ys_n[-1]])

            artists.extend([solver_lines[k], solver_dots[k], net_lines[k], net_dots[k]])
        return artists

    anim = FuncAnimation(fig, update, frames=min(len(traj_net), len(traj_solver)),
                          interval=interval, blit=True)

    if save_path is not None:
        anim.save(save_path, dpi=150)

    plt.show()
    return anim


if __name__ == "__main__":
    print("=" * 90)
    print("Avvio del test di confronto: Modello 2-corpi applicato a coppie per il sistema a 3 corpi")
    print("=" * 90)

    # Percorso del checkpoint del modello a 2 corpi già addestrato
    PATH_2BODY = "./PINN_savefile/save_equivariance_acc_v4.pt"  # Sostituisci se usi v5 o g-PINN

    torch.manual_seed(63)
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    n_blocks = 4
    dt = 0.01
    rollout_steps = 500
    dtype = torch.float64
    n_body = 3

    # 1. Inizializzazione della rete base a 2 corpi
    net_2body = Acceleration2BodyNetv4(
        num_blocks=n_blocks,
        dtype=dtype,
        device=DEVICE
    ).to(DEVICE)

    # Caricamento dei pesi
    ckpt = torch.load(PATH_2BODY, map_location=DEVICE)
    weights = ckpt.get("model", ckpt)
    net_2body.load_state_dict(weights)
    print(f"Modello a 2 corpi caricato correttamente da: {PATH_2BODY}")

    # 2. Istanziazione del wrapper pairwise per 3 corpi
    pairwise_model = PairwiseWrapperNBody(net_2body, n_obj=n_body, dt=dt).to(DEVICE)

    # 3. Esecuzione del test / rollout
    results = test_pairwise_model(pairwise_model, n_obj=n_body, device=DEVICE, rollout_steps=rollout_steps, dt=dt, dtype=dtype)

    # 4. Calcolo delle metriche di errore
    err = compute_stepwise_error(results)
    print("\n[Risultati Pairwise 3-Corpi]")
    print("Errore di posizione, ultimo step:", {f"body{k+1}": err["pos_err"][k][-1] for k in range(n_body)})
    print("Errore di posizione, primo step: ", {f"body{k+1}": err["pos_err"][k][0] for k in range(n_body)})

    print(f"\nDrift energia:          max={max(results['energy_drift']):.3e}")
    print(f"Drift momento angolare: max={max(results['angular_drift']):.3e}")
    print(f"Drift momento lineare:  max={max(results['momentum_drift']):.3e}")

    # 5. Visualizzazione grafica e animazione della traiettoria
    animate_trajectory(results["traj_net"], results["traj_solver"], n_body, interval=20)