import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.animation import FuncAnimation
from networks import AccelerationNBodyNetv4
from networks_2 import AccelerationNBodyNetv5
import random
from tqdm import tqdm
from scipy.integrate import solve_ivp
from train_nbody import (
    generate_instance,
    compute_energy,
    compute_angular_momentum,
    conservation_loss,
    canonicalize_translation,
    uncanonicalize_translation,
)


def compute_momentum(states):
    """Momento lineare totale, generalizzato a N corpi."""
    B = states.shape[0]
    N = states.shape[1] // 5
    x_r = states.view(B, N, 5)
    m = x_r[:, :, 0:1]
    v = x_r[:, :, 3:5]
    return torch.sum(m * v, dim=1)  # [B, 2]


# Animazione (generalizzata a N corpi con color cycle)

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
        nl, = ax.plot([], [], "--", color=colors[k], lw=2, alpha=0.6, label=f"Body {k+1} Network")
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


# Solver di riferimento (N corpi generico)

def n_body_rhs(t, state, masses, G=1.0, eps=1e-3):
    """state = [pos_flat(2N), vel_flat(2N)] -> dstate/dt nello stesso formato."""
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
    """Integrazione numerica di riferimento per N corpi. state0 = [pos_flat(2N), vel_flat(2N)]."""
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


def test(BodyNetwork, n_obj, device=torch.device("cpu"), rollout_steps=30, dt=0.01,
         seed_state=None, dtype=torch.float64):
    BodyNetwork.eval()

    state = seed_state if seed_state is not None else generate_instance(1, n_obj, device, dtype=dtype)

    traj_net = []
    with torch.no_grad():
        s_abs = state.clone()
        for _ in range(rollout_steps):
            traj_net.append(s_abs.squeeze(0).cpu())

            s_canon, p_min = canonicalize_translation(s_abs)
            out_canon = BodyNetwork(s_canon, dt)
            s_abs = uncanonicalize_translation(out_canon, p_min)

    # Condizioni iniziali per il solver di riferimento (N corpi generico)
    state_np = state.squeeze(0).detach().cpu().numpy().reshape(n_obj, 5)
    masses = state_np[:, 0]                       # [N]
    pos0 = state_np[:, 1:3].reshape(-1)            # [2N]
    vel0 = state_np[:, 3:5].reshape(-1)            # [2N]
    state0 = np.concatenate([pos0, vel0])

    _, sol = solve_nbody(state0, masses, dt=dt, n_points=rollout_steps)

    traj_solver = []
    for row in sol:
        pos = row[:2 * n_obj].reshape(n_obj, 2)
        vel = row[2 * n_obj:].reshape(n_obj, 2)
        body_rows = [torch.tensor([masses[k], pos[k, 0], pos[k, 1], vel[k, 0], vel[k, 1]], dtype=dtype)
                     for k in range(n_obj)]
        traj_solver.append(torch.cat(body_rows))

    # Drift e quantità conservate (stati assoluti)
    E0 = compute_energy(traj_net[0].unsqueeze(0))
    L0 = compute_angular_momentum(traj_net[0].unsqueeze(0))
    p0 = compute_momentum(traj_net[0].unsqueeze(0))

    drift_energy, drift_L, drift_p = [], [], []
    for s in traj_net:
        s_b = s.unsqueeze(0)
        drift_energy.append(torch.mean((compute_energy(s_b) - E0) ** 2).item())
        drift_L.append(torch.mean((compute_angular_momentum(s_b) - L0) ** 2).item())
        drift_p.append(torch.mean(torch.sum((compute_momentum(s_b) - p0) ** 2, dim=-1)).item())

    BodyNetwork.train()
    return {
        "traj_net": traj_net,
        "traj_solver": traj_solver,
        "energy_drift": drift_energy,
        "angular_drift": drift_L,
        "momentum_drift": drift_p,
        "n_obj": n_obj,
    }

# Errore rete-vs-solver, step per step (per ciascun corpo)

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
        "pos_err": [np.array(e) for e in pos_err],   # lista di array, uno per corpo
        "vel_err": [np.array(e) for e in vel_err],
        "n_obj": n_obj,
    }


# Diagnostica di DIREZIONE al primo step (generalizzata a N corpi)
def direction_diagnostic(BodyNetwork, n_obj, device, dt, n_samples=200, eps=1e-3, G=1.0, dtype=torch.float64):
    BodyNetwork.eval()
    angles = [[] for _ in range(n_obj)]
    ratios = [[] for _ in range(n_obj)]

    with torch.no_grad():
        for _ in range(n_samples):
            s0 = generate_instance(1, n_obj, device, dtype=dtype)
            s0_canon, p_min = canonicalize_translation(s0)
            s1_canon = BodyNetwork(s0_canon, dt)
            s1 = uncanonicalize_translation(s1_canon, p_min)

            s0_r = s0.view(n_obj, 5)
            s1_r = s1.view(n_obj, 5)

            for k in range(n_obj):
                disp_net = s1_r[k, 1:3] - s0_r[k, 1:3]
                disp_expected = s0_r[k, 3:5] * dt

                n_net = torch.norm(disp_net)
                n_exp = torch.norm(disp_expected)

                if n_net > 1e-10 and n_exp > 1e-10:
                    cos_angle = torch.dot(disp_net, disp_expected) / (n_net * n_exp)
                    cos_angle = torch.clamp(cos_angle, -1.0, 1.0)
                    angle_deg = torch.rad2deg(torch.arccos(cos_angle)).item()
                    angles[k].append(angle_deg)
                    ratios[k].append((n_net / n_exp).item())

    BodyNetwork.train()

    for k in range(n_obj):
        a, r = np.array(angles[k]), np.array(ratios[k])
        print(f"--- Body {k+1} ---")
        print(f"  angolo spostamento vs v*dt: media={a.mean():.1f}°  "
              f"mediana={np.median(a):.1f}°  std={a.std():.1f}°")
        print(f"  |spostamento_rete| / |v*dt|: media={r.mean():.3f}  mediana={np.median(r):.3f}")

    return {
        "angles": [np.array(a) for a in angles],
        "ratios": [np.array(r) for r in ratios],
        "n_obj": n_obj,
    }

# Plot diagnostici (generalizzati a N corpi)

def plot_diagnostics(results, err, dir_diag, save_prefix="diag"):
    n_obj = results["n_obj"]
    colors = [cm.tab10(k % 10) for k in range(n_obj)]
    n_steps = len(err["pos_err"][0])
    steps = np.arange(n_steps)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for k in range(n_obj):
        axes[0].plot(steps, err["pos_err"][k], label=f"Body {k+1}", color=colors[k])
        axes[1].plot(steps, err["vel_err"][k], label=f"Body {k+1}", color=colors[k])
    axes[0].set_yscale("log"); axes[0].set_xlabel("step"); axes[0].set_ylabel("errore posizione (norma L2)")
    axes[0].set_title("Errore di posizione rete vs solver"); axes[0].legend(); axes[0].grid(True, which="both", alpha=0.3)
    axes[1].set_yscale("log"); axes[1].set_xlabel("step"); axes[1].set_ylabel("errore velocità (norma L2)")
    axes[1].set_title("Errore di velocità rete vs solver"); axes[1].legend(); axes[1].grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    fig.savefig(f"{save_prefix}_stepwise_error.png", dpi=150)

    fig2, axes2 = plt.subplots(1, 3, figsize=(15, 4.5))
    for ax, key, title in zip(
        axes2,
        ["energy_drift", "angular_drift", "momentum_drift"],
        ["Drift energia (rete)", "Drift momento angolare (rete)", "Drift momento lineare (rete)"],
    ):
        vals = np.array(results[key])
        ax.plot(vals); ax.set_yscale("log"); ax.set_xlabel("step"); ax.set_title(title)
        ax.grid(True, which="both", alpha=0.3)
    fig2.tight_layout()
    fig2.savefig(f"{save_prefix}_conservation_drift.png", dpi=150)

    fig3, axes3 = plt.subplots(1, 2, figsize=(12, 4.5))
    for k in range(n_obj):
        axes3[0].hist(dir_diag["angles"][k], bins=30, alpha=0.5, label=f"Body {k+1}", color=colors[k])
        axes3[1].hist(dir_diag["ratios"][k], bins=30, alpha=0.5, label=f"Body {k+1}", color=colors[k])
    axes3[0].axvline(90, color="red", linestyle="--", label="90° (rotazione pura)")
    axes3[0].set_xlabel("angolo tra spostamento rete e v*dt atteso (°)")
    axes3[0].set_ylabel("conteggio"); axes3[0].set_title("Direzione dello spostamento al 1° step"); axes3[0].legend(fontsize=8)
    axes3[1].axvline(1.0, color="red", linestyle="--", label="1.0 (modulo corretto)")
    axes3[1].set_xlabel("|spostamento rete| / |v*dt|"); axes3[1].set_title("Modulo dello spostamento al 1° step")
    axes3[1].legend(fontsize=8)
    fig3.tight_layout()
    fig3.savefig(f"{save_prefix}_direction_check.png", dpi=150)

    plt.close("all")


if __name__ == "__main__":

    print("=" * 90)

    N_BODY = 3  # deve combaciare col checkpoint caricato
    PATH = f"./PINN_savefile/save_nbody_{N_BODY}_gpinn_acc_v4.pt"

    torch.manual_seed(42)

    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    n_blocks = 4
    dt = 0.01
    rollout_steps = 1000
    dtype = torch.float64

    BodyNetwork = AccelerationNBodyNetv4(
        n_obj=N_BODY,
        num_blocks=n_blocks,
        dtype=dtype,
        device=DEVICE
    ).to(DEVICE)

    ckpt = torch.load(PATH, map_location=DEVICE)
    weights = ckpt.get("model", ckpt)
    BodyNetwork.load_state_dict(weights)

    results = test(BodyNetwork, N_BODY, device=DEVICE, rollout_steps=rollout_steps, dt=dt, dtype=dtype)

    err = compute_stepwise_error(results)
    print("\nErrore di posizione, ultimo step:", {f"body{k+1}": err["pos_err"][k][-1] for k in range(N_BODY)})
    print("Errore di posizione, primo step: ", {f"body{k+1}": err["pos_err"][k][0] for k in range(N_BODY)})

    print(f"\nDrift energia:          max={max(results['energy_drift']):.3e}")
    print(f"Drift momento angolare: max={max(results['angular_drift']):.3e}")
    print(f"Drift momento lineare:  max={max(results['momentum_drift']):.3e}")

    print(f"\n--- Diagnostica di direzione (single-step, {rollout_steps} campioni indipendenti) ---")
    dir_diag = direction_diagnostic(BodyNetwork, N_BODY, DEVICE, dt, n_samples=rollout_steps, dtype=dtype)

    plot_diagnostics(results, err, dir_diag, save_prefix="diag_nbody")
    print("\nSalvati: diag_nbody_stepwise_error.png, diag_nbody_conservation_drift.png, diag_nbody_direction_check.png")

    animate_trajectory(results["traj_net"], results["traj_solver"], N_BODY, interval=20)
