import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from networks import AB2Net, FullyEquivariant2BodyNet, Acceleration2BodyNet
import random
from tqdm import tqdm
from scipy.integrate import solve_ivp
from train import (
    generate_instance,
    compute_energy,
    compute_angular_momentum,
    conservation_loss,
    canonicalize_translation,
    uncanonicalize_translation,
)


def compute_momentum(states):
    """Total linear momentum p = (sum m_i*vx_i, sum m_i*vy_i)."""
    m1, m2 = states[:, 0], states[:, 5]
    vx1, vy1 = states[:, 3], states[:, 4]
    vx2, vy2 = states[:, 8], states[:, 9]
    return torch.stack([m1 * vx1 + m2 * vx2, m1 * vy1 + m2 * vy2], dim=1)


# ============================================================
# Animazione (invariata)
# ============================================================

def animate_trajectory(traj_net, traj_solver, interval=30, save_path=None):
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.set_aspect("equal")

    xs, ys = [], []
    for s in traj_net:
        xs.extend([s[1].item(), s[6].item()])
        ys.extend([s[2].item(), s[7].item()])
    for s in traj_solver:
        xs.extend([s[1].item(), s[6].item()])
        ys.extend([s[2].item(), s[7].item()])

    margin = 0.5
    ax.set_xlim(min(xs) - margin, max(xs) + margin)
    ax.set_ylim(min(ys) - margin, max(ys) + margin)
    ax.grid(True)

    solver1_line, = ax.plot([], [], "b-", lw=2, label="Body 1 Solver")
    solver2_line, = ax.plot([], [], "r-", lw=2, label="Body 2 Solver")
    solver1_dot, = ax.plot([], [], "bo", ms=8)
    solver2_dot, = ax.plot([], [], "ro", ms=8)

    net1_line, = ax.plot([], [], "b--", lw=2, alpha=0.6, label="Body 1 Network")
    net2_line, = ax.plot([], [], "r--", lw=2, alpha=0.6, label="Body 2 Network")
    net1_dot, = ax.plot([], [], "bs", ms=6)
    net2_dot, = ax.plot([], [], "rs", ms=6)

    ax.legend()

    def update(frame):
        xs1 = [traj_solver[i][1].item() for i in range(frame + 1)]
        ys1 = [traj_solver[i][2].item() for i in range(frame + 1)]
        xs2 = [traj_solver[i][6].item() for i in range(frame + 1)]
        ys2 = [traj_solver[i][7].item() for i in range(frame + 1)]

        solver1_line.set_data(xs1, ys1)
        solver2_line.set_data(xs2, ys2)
        solver1_dot.set_data([xs1[-1]], [ys1[-1]])
        solver2_dot.set_data([xs2[-1]], [ys2[-1]])

        xn1 = [traj_net[i][1].item() for i in range(frame + 1)]
        yn1 = [traj_net[i][2].item() for i in range(frame + 1)]
        xn2 = [traj_net[i][6].item() for i in range(frame + 1)]
        yn2 = [traj_net[i][7].item() for i in range(frame + 1)]

        net1_line.set_data(xn1, yn1)
        net2_line.set_data(xn2, yn2)
        net1_dot.set_data([xn1[-1]], [yn1[-1]])
        net2_dot.set_data([xn2[-1]], [yn2[-1]])

        return (solver1_line, solver2_line, solver1_dot, solver2_dot,
                net1_line, net2_line, net1_dot, net2_dot)

    anim = FuncAnimation(fig, update, frames=min(len(traj_net), len(traj_solver)),
                          interval=interval, blit=True)

    if save_path is not None:
        anim.save(save_path, dpi=150)

    plt.show()
    return anim


# ============================================================
# Solver di riferimento
# ============================================================

def two_body_rhs(t, state, masses, G=1.0, eps=1e-3):
    x1, y1, x2, y2 = state[0:4]
    vx1, vy1, vx2, vy2 = state[4:8]
    m1, m2 = masses

    r12 = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2 + eps ** 2)
    ax1 = G * m2 * (x2 - x1) / r12 ** 3
    ay1 = G * m2 * (y2 - y1) / r12 ** 3
    ax2 = G * m1 * (x1 - x2) / r12 ** 3
    ay2 = G * m1 * (y1 - y2) / r12 ** 3

    return [vx1, vy1, vx2, vy2, ax1, ay1, ax2, ay2]


def solve_twobody(state0, masses, dt, n_points, G=1.0):
    """Integrazione numerica del problema dei due corpi.

    FIX rispetto alla versione precedente: qui i punti valutati sono ESATTAMENTE
    agli stessi istanti t = 0, dt, 2*dt, ... dei passi della rete (t_eval costruito
    da dt direttamente, non da un t_max diviso per n_points-1) -- prima c'era un
    disallineamento sistematico di circa dt/(n_points-1) per punto, che si
    accumulava lungo il confronto.
    """
    t_eval = np.arange(n_points) * dt

    sol = solve_ivp(
        two_body_rhs,
        (0, t_eval[-1]),
        state0,
        args=(masses, G),
        t_eval=t_eval,
        method="DOP853",
        rtol=1e-12,
        atol=1e-14,
    )
    return sol.t, sol.y.T



def test(BodyNetwork, device=torch.device("cpu"), rollout_steps=30, dt=0.01, seed_state=None, dtype=torch.float64):
    BodyNetwork.eval()

    # Stato iniziale (già canonizzato da generate_instance)
    state = seed_state if seed_state is not None else generate_instance(batch_size=1, device=device, dtype=dtype)

    traj_net = []
    with torch.no_grad():
        s_abs = state.clone() # Mantiene lo stato assoluto nel riferimento globale
        for _ in range(rollout_steps):
            traj_net.append(s_abs.squeeze(0).cpu())
            
            # 1. Canonizziamo prima di passare alla rete
            s_canon, p_min = canonicalize_translation(s_abs)
            
            # 2. La rete predice lo stato canonico al tempo t+1
            out_canon = BodyNetwork(s_canon, dt)
            
            # 3. Riportiamo l'output nel sistema di riferimento globale
            s_abs = uncanonicalize_translation(out_canon, p_min)

    # Preparazione per il solver di riferimento (condizioni iniziali t=0)
    state_np = state.squeeze(0).detach().cpu().numpy()
    masses = (state_np[0], state_np[5])
    state0 = np.array([
        state_np[1], state_np[2], state_np[6], state_np[7],
        state_np[3], state_np[4], state_np[8], state_np[9],
    ])

    _, sol = solve_twobody(state0, masses, dt=dt, n_points=rollout_steps)

    traj_solver = []
    for row in sol:
        traj_solver.append(torch.tensor([
            masses[0], row[0], row[1], row[4], row[5],
            masses[1], row[2], row[3], row[6], row[7]
        ], dtype=dtype))

    # Drift e quantità conservate (calcolate sugli stati assoluti)
    E0 = compute_energy(traj_net[0].unsqueeze(0))
    L0 = compute_angular_momentum(traj_net[0].unsqueeze(0))
    p0 = compute_momentum(traj_net[0].unsqueeze(0))

    drift_energy, drift_L, drift_p = [], [], []
    for s in traj_net:
        s_b = s.unsqueeze(0)
        drift_energy.append(torch.mean((compute_energy(s_b) - E0) ** 2).item())
        drift_L.append(torch.mean((compute_angular_momentum(s_b) - L0) ** 2).item())
        drift_p.append(torch.mean((compute_momentum(s_b) - p0) ** 2).item())

    BodyNetwork.train()
    return {
        "traj_net": traj_net,
        "traj_solver": traj_solver,
        "energy_drift": drift_energy,
        "angular_drift": drift_L,
        "momentum_drift": drift_p,
    }

# ============================================================
# Errore rete-vs-solver, step per step
# ============================================================

def compute_stepwise_error(results):
    """Errore euclideo posizione/velocità, rete vs solver, per ciascun corpo, ad ogni step."""
    traj_net, traj_solver = results["traj_net"], results["traj_solver"]
    n = min(len(traj_net), len(traj_solver))

    pos_err_1, pos_err_2 = [], []
    vel_err_1, vel_err_2 = [], []

    for i in range(n):
        sn, ss = traj_net[i], traj_solver[i]

        p1n, p1s = sn[1:3], ss[1:3]
        p2n, p2s = sn[6:8], ss[6:8]
        v1n, v1s = sn[3:5], ss[3:5]
        v2n, v2s = sn[8:10], ss[8:10]

        pos_err_1.append(torch.norm(p1n - p1s).item())
        pos_err_2.append(torch.norm(p2n - p2s).item())
        vel_err_1.append(torch.norm(v1n - v1s).item())
        vel_err_2.append(torch.norm(v2n - v2s).item())

    return {
        "pos_err_body1": np.array(pos_err_1),
        "pos_err_body2": np.array(pos_err_2),
        "vel_err_body1": np.array(vel_err_1),
        "vel_err_body2": np.array(vel_err_2),
    }


# ============================================================
# Diagnostica di DIREZIONE: la rete predice lo spostamento
# nella direzione giusta al PRIMO step, isolato dall'accumulo?
# ============================================================
def direction_diagnostic(BodyNetwork, device, dt, n_samples=200, eps=1e-3, G=1.0, dtype=torch.float64):
    BodyNetwork.eval()
    angles_body1, angles_body2 = [], []
    ratio_body1, ratio_body2 = [], []

    with torch.no_grad():
        for _ in range(n_samples):
            s0 = generate_instance(batch_size=1, device=device, dtype=dtype)
            
            # Canonizziamo prima del forward
            s0_canon, p_min = canonicalize_translation(s0)
            s1_canon = BodyNetwork(s0_canon, dt)
            s1 = uncanonicalize_translation(s1_canon, p_min)

            for body, (idx_pos, idx_vel, angles, ratios) in enumerate([
                ((1, 2), (3, 4), angles_body1, ratio_body1),
                ((6, 7), (8, 9), angles_body2, ratio_body2),
            ]):
                disp_net = torch.stack([
                    s1[0, idx_pos[0]] - s0[0, idx_pos[0]],
                    s1[0, idx_pos[1]] - s0[0, idx_pos[1]],
                ])
                disp_expected = torch.stack([
                    s0[0, idx_vel[0]] * dt,
                    s0[0, idx_vel[1]] * dt,
                ])

                n_net = torch.norm(disp_net)
                n_exp = torch.norm(disp_expected)

                if n_net > 1e-10 and n_exp > 1e-10:
                    cos_angle = torch.dot(disp_net, disp_expected) / (n_net * n_exp)
                    cos_angle = torch.clamp(cos_angle, -1.0, 1.0)
                    angle_deg = torch.rad2deg(torch.arccos(cos_angle)).item()
                    angles.append(angle_deg)
                    ratios.append((n_net / n_exp).item())

    BodyNetwork.train()

    def summarize(name, angles, ratios):
        angles = np.array(angles)
        ratios = np.array(ratios)
        print(f"--- {name} ---")
        print(f"  angolo spostamento vs v*dt: media={angles.mean():.1f}°  "
              f"mediana={np.median(angles):.1f}°  std={angles.std():.1f}°")
        print(f"  |spostamento_rete| / |v*dt|: media={ratios.mean():.3f}  "
              f"mediana={np.median(ratios):.3f}")

    summarize("Body 1", angles_body1, ratio_body1)
    summarize("Body 2", angles_body2, ratio_body2)

    return {
        "angles_body1": np.array(angles_body1), "ratios_body1": np.array(ratio_body1),
        "angles_body2": np.array(angles_body2), "ratios_body2": np.array(ratio_body2),
    }

# ============================================================
# Plot diagnostici
# ============================================================

def plot_diagnostics(results, err, dir_diag, save_prefix="diag"):
    n_steps = len(err["pos_err_body1"])
    steps = np.arange(n_steps)

    # 1. Errore di posizione/velocità, rete vs solver, per step (scala log)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    axes[0].plot(steps, err["pos_err_body1"], label="Body 1")
    axes[0].plot(steps, err["pos_err_body2"], label="Body 2")
    axes[0].set_yscale("log")
    axes[0].set_xlabel("step")
    axes[0].set_ylabel("errore posizione (norma L2)")
    axes[0].set_title("Errore di posizione rete vs solver")
    axes[0].legend()
    axes[0].grid(True, which="both", alpha=0.3)

    axes[1].plot(steps, err["vel_err_body1"], label="Body 1")
    axes[1].plot(steps, err["vel_err_body2"], label="Body 2")
    axes[1].set_yscale("log")
    axes[1].set_xlabel("step")
    axes[1].set_ylabel("errore velocità (norma L2)")
    axes[1].set_title("Errore di velocità rete vs solver")
    axes[1].legend()
    axes[1].grid(True, which="both", alpha=0.3)

    fig.tight_layout()
    fig.savefig(f"{save_prefix}_stepwise_error.png", dpi=150)

    # 2. Drift di energia / momento angolare / momento lineare
    fig2, axes2 = plt.subplots(1, 3, figsize=(15, 4.5))
    for ax, key, title in zip(
        axes2,
        ["energy_drift", "angular_drift", "momentum_drift"],
        ["Drift energia (rete)", "Drift momento angolare (rete)", "Drift momento lineare (rete)"],
    ):
        vals = np.array(results[key])
        ax.plot(vals)
        ax.set_yscale("log")
        ax.set_xlabel("step")
        ax.set_title(title)
        ax.grid(True, which="both", alpha=0.3)

    fig2.tight_layout()
    fig2.savefig(f"{save_prefix}_conservation_drift.png", dpi=150)

    # 3. Diagnostica di direzione (istogrammi)
    fig3, axes3 = plt.subplots(1, 2, figsize=(12, 4.5))
    axes3[0].hist(dir_diag["angles_body1"], bins=30, alpha=0.6, label="Body 1")
    axes3[0].hist(dir_diag["angles_body2"], bins=30, alpha=0.6, label="Body 2")
    axes3[0].axvline(90, color="red", linestyle="--", label="90° (rotazione pura)")
    axes3[0].set_xlabel("angolo tra spostamento rete e v*dt atteso (°)")
    axes3[0].set_ylabel("conteggio")
    axes3[0].set_title("Direzione dello spostamento al 1° step")
    axes3[0].legend()

    axes3[1].hist(dir_diag["ratios_body1"], bins=30, alpha=0.6, label="Body 1")
    axes3[1].hist(dir_diag["ratios_body2"], bins=30, alpha=0.6, label="Body 2")
    axes3[1].axvline(1.0, color="red", linestyle="--", label="1.0 (modulo corretto)")
    axes3[1].set_xlabel("|spostamento rete| / |v*dt|")
    axes3[1].set_title("Modulo dello spostamento al 1° step")
    axes3[1].legend()

    fig3.tight_layout()
    fig3.savefig(f"{save_prefix}_direction_check.png", dpi=150)

    plt.close("all")


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":

    print("=" * 90)

    PATH = "./PINN_savefile/save_equivariance_acc.pt"  # percorso aggiornato

    torch.manual_seed(63)

    #63 ellissi
    #resto di solito fionda (42)

    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    n_blocks = 4
    dt = 0.01
    rollout_steps = 1000
    dtype = torch.float64

    BodyNetwork = Acceleration2BodyNet(
        num_blocks=n_blocks,
        dtype=dtype,
        device=DEVICE
    ).to(DEVICE)

    ckpt = torch.load(PATH, map_location=DEVICE)
    weights = ckpt.get("model", ckpt)
    BodyNetwork.load_state_dict(weights)

    # 1. Rollout rete vs solver (con time alignment e traslazione canonica corrette)
    results = test(BodyNetwork, device=DEVICE, rollout_steps=rollout_steps, dt=dt, dtype=dtype)

    # 2. Errore stepwise
    err = compute_stepwise_error(results)
    print(f"\nErrore di posizione, ultimo step: body1={err['pos_err_body1'][-1]:.4f}  "
          f"body2={err['pos_err_body2'][-1]:.4f}")
    print(f"Errore di posizione, primo step:  body1={err['pos_err_body1'][0]:.6f}  "
          f"body2={err['pos_err_body2'][0]:.6f}")

    # 3. Drift di conservazione
    print(f"\nDrift energia:          max={max(results['energy_drift']):.3e}")
    print(f"Drift momento angolare: max={max(results['angular_drift']):.3e}")
    print(f"Drift momento lineare:  max={max(results['momentum_drift']):.3e}")

    # 4. Diagnostica di direzione
    print(f"\n--- Diagnostica di direzione (single-step, {rollout_steps} campioni indipendenti) ---")
    dir_diag = direction_diagnostic(BodyNetwork, DEVICE, dt, n_samples=rollout_steps, dtype=dtype)

    # 5. Plot
    plot_diagnostics(results, err, dir_diag, save_prefix="diag")
    print("\nSalvati: diag_stepwise_error.png, diag_conservation_drift.png, diag_direction_check.png")

    # 6. Animazione
    animate_trajectory(results["traj_net"], results["traj_solver"], interval=20)