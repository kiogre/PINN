import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from networks import AB2Net
import random
from tqdm import tqdm
from scipy.integrate import solve_ivp

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation


def animate_trajectory(traj_net, traj_solver, interval=30, save_path=None):
    """
    traj_net e traj_solver sono liste di stati nel formato

    [m1,x1,y1,vx1,vy1,m2,x2,y2,vx2,vy2]
    """

    fig, ax = plt.subplots(figsize=(7, 7))

    ax.set_aspect("equal")

    # ------------------------------
    # Calcolo limiti grafico
    # ------------------------------

    xs = []
    ys = []

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

    # ===================================
    # Solver
    # ===================================

    solver1_line, = ax.plot([], [], "b-", lw=2, label="Body 1 Solver")
    solver2_line, = ax.plot([], [], "r-", lw=2, label="Body 2 Solver")

    solver1_dot, = ax.plot([], [], "bo", ms=8)
    solver2_dot, = ax.plot([], [], "ro", ms=8)

    # ===================================
    # Network
    # ===================================

    net1_line, = ax.plot([], [], "b--", lw=2, alpha=0.6,
                         label="Body 1 Network")

    net2_line, = ax.plot([], [], "r--", lw=2, alpha=0.6,
                         label="Body 2 Network")

    net1_dot, = ax.plot([], [], "bs", ms=6)
    net2_dot, = ax.plot([], [], "rs", ms=6)

    ax.legend()

    # ===================================

    def update(frame):

        # ---------- Solver ----------

        xs1 = [traj_solver[i][1].item() for i in range(frame + 1)]
        ys1 = [traj_solver[i][2].item() for i in range(frame + 1)]

        xs2 = [traj_solver[i][6].item() for i in range(frame + 1)]
        ys2 = [traj_solver[i][7].item() for i in range(frame + 1)]

        solver1_line.set_data(xs1, ys1)
        solver2_line.set_data(xs2, ys2)

        solver1_dot.set_data([xs1[-1]], [ys1[-1]])
        solver2_dot.set_data([xs2[-1]], [ys2[-1]])

        # ---------- Network ----------

        xn1 = [traj_net[i][1].item() for i in range(frame + 1)]
        yn1 = [traj_net[i][2].item() for i in range(frame + 1)]

        xn2 = [traj_net[i][6].item() for i in range(frame + 1)]
        yn2 = [traj_net[i][7].item() for i in range(frame + 1)]

        net1_line.set_data(xn1, yn1)
        net2_line.set_data(xn2, yn2)

        net1_dot.set_data([xn1[-1]], [yn1[-1]])
        net2_dot.set_data([xn2[-1]], [yn2[-1]])

        return (
            solver1_line,
            solver2_line,
            solver1_dot,
            solver2_dot,
            net1_line,
            net2_line,
            net1_dot,
            net2_dot,
        )

    anim = FuncAnimation(
        fig,
        update,
        frames=min(len(traj_net), len(traj_solver)),
        interval=interval,
        blit=True,
    )

    if save_path is not None:
        anim.save(save_path, dpi=150)

    plt.show()

    return anim

def two_body_rhs(t, state, masses, G=1.0, eps=1e-3):
    """
    RHS per scipy.solve_ivp.
    state = [x1, y1, x2, y2, vx1, vy1, vx2, vy2]
    """

    x1, y1, x2, y2 = state[0:4]
    vx1, vy1, vx2, vy2 = state[4:8]

    m1, m2 = masses

    # distanza tra i due corpi
    r12 = np.sqrt((x2 - x1)**2 + (y2 - y1)**2 + eps**2)

    # accelerazioni
    ax1 = G * m2 * (x2 - x1) / r12**3
    ay1 = G * m2 * (y2 - y1) / r12**3

    ax2 = G * m1 * (x1 - x2) / r12**3
    ay2 = G * m1 * (y1 - y2) / r12**3

    return [
        vx1, vy1,
        vx2, vy2,
        ax1, ay1,
        ax2, ay2
    ]


def solve_twobody(state0, masses, t_max, G=1.0, n_points=2000):
    """
    Integrazione numerica del problema dei due corpi.
    """

    t_eval = np.linspace(0, t_max, n_points)

    sol = solve_ivp(
        two_body_rhs,
        (0, t_max),
        state0,
        args=(masses, G),
        t_eval=t_eval,
        method="DOP853",
        rtol=1e-12,
        atol=1e-14,
    )

    return sol.t, sol.y.T      # (n_points, 8)

def test(BodyNetwork,
         device=torch.device("cpu"),
         rollout_steps=200,
         dt=0.05):

    BodyNetwork.eval()

    # ===========================
    # Stato iniziale
    # ===========================

    state = generate_instance(batch_size=1, device=device)

    traj_net = []
    traj_solver = []

    # ===========================
    # Rollout della rete
    # ===========================

    with torch.no_grad():

        s = state.clone()

        for _ in range(rollout_steps):
            traj_net.append(s.squeeze(0).cpu())
            s = BodyNetwork(s)

    # ===========================
    # Solver numerico
    # ===========================

    state_np = state.squeeze(0).detach().cpu().numpy()

    masses = (
        state_np[0],
        state_np[5]
    )

    state0 = np.array([
        state_np[1],  # x1
        state_np[2],  # y1
        state_np[6],  # x2
        state_np[7],  # y2
        state_np[3],  # vx1
        state_np[4],  # vy1
        state_np[8],  # vx2
        state_np[9],  # vy2
    ])

    _, sol = solve_twobody(
        state0,
        masses,
        t_max=rollout_steps * dt,
        n_points=rollout_steps
    )

    # converto nel formato della rete:
    # [m1,x1,y1,vx1,vy1,m2,x2,y2,vx2,vy2]

    for row in sol:

        traj_solver.append(torch.tensor([
            masses[0],
            row[0],
            row[1],
            row[4],
            row[5],
            masses[1],
            row[2],
            row[3],
            row[6],
            row[7]
        ], dtype=torch.float64))

    # ===========================
    # Energia
    # ===========================

    E0_net = compute_energy(traj_net[0].unsqueeze(0))

    drift_energy = []

    for s in traj_net:

        drift_energy.append(
            torch.mean(
                (compute_energy(s.unsqueeze(0)) - E0_net) ** 2
            ).item()
        )

    # ===========================
    # Momento angolare
    # ===========================

    L0_net = compute_angular_momentum(traj_net[0].unsqueeze(0))

    drift_L = []

    for s in traj_net:

        drift_L.append(
            torch.mean(
                (compute_angular_momentum(s.unsqueeze(0)) - L0_net) ** 2
            ).item()
        )

    BodyNetwork.train()

    return {
        "traj_net": traj_net,
        "traj_solver": traj_solver,
        "energy_drift": drift_energy,
        "angular_drift": drift_L
    }

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
    """Penalise energy and angular-momentum drift between input state and output state."""
    E, E0 = compute_energy(output, G, eps), compute_energy(input, G, eps)
    L, L0 = compute_angular_momentum(output), compute_angular_momentum(input)

    loss_E = torch.mean((E - E0) ** 2)
    loss_L = torch.mean((L - L0) ** 2)
    return loss_E + loss_L

if __name__ == "__main__":

    print("="*90)

    PATH = "./PINN_savefile/save.pt"

    torch.manual_seed(69)
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    dim = 10
    
    n_blocks = 4

    BodyNetwork = AB2Net(
            in_features=dim,
            out_features=8,
            num_blocks=n_blocks,
            dtype=torch.float64,
        ).to(DEVICE)

    ckpt = torch.load(PATH)
    weights = ckpt.get("model", ckpt)

    BodyNetwork.load_state_dict(weights)

    results = test(BodyNetwork, device=DEVICE)

    animate_trajectory(
        results["traj_net"],
        results["traj_solver"],
        interval=20
    )