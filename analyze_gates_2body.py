"""
Analisi focalizzata sui gate (theta e fattori di scala) delle reti
2-corpi equivarianti (Acceleration2BodyNetv2/v3/v4 e v5).

Obiettivo (progetto magistrale):
  SOLO alle mini-reti che producono quantita' scalari:
    - theta          (RotationGate)
    - fattori di scala lambda (InvariantGate)

Inoltre si fa probing rispetto alle variabili angolo-azione del problema
di Keplero (energia, |L|, eccentricita', anomalia media / anomalia vera),
che sono le quantita' fisiche piu' naturali da controllare.

Uso:
    python analyze_gates.py --checkpoint path/to/ckpt.pt [--arch v4|v5]
"""

import argparse
from collections import defaultdict
from sklearn.decomposition import PCA

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree

# ------------------------------------------------------------------
# Import delle reti e del generatore di stati
# ------------------------------------------------------------------
from train import generate_instance as generate_instance_2body

try:
    from networks import (
        Acceleration2BodyNetv2,
        Acceleration2BodyNetv3,
        Acceleration2BodyNetv4,
    )
except ImportError:
    Acceleration2BodyNetv2 = Acceleration2BodyNetv3 = Acceleration2BodyNetv4 = None

try:
    from networks_2 import Acceleration2BodyNetv5
except ImportError:
    Acceleration2BodyNetv5 = None


# ------------------------------------------------------------------
# Elementi orbitali (variabili angolo-azione del problema di Keplero)
# ------------------------------------------------------------------
def orbital_elements(state, G=1.0):
    """Restituisce un dizionario con le principali quantita' angolo-azione
    (e l'anomalia vera) per un batch di stati 2-corpi.
    state: [B, 10]
    """
    m1, m2 = state[:, 0], state[:, 5]
    mu = G * (m1 + m2)

    r = state[:, 6:8] - state[:, 1:3]          # posizione relativa
    vrel = state[:, 8:10] - state[:, 3:5]      # velocita' relativa

    r_norm = torch.norm(r, dim=-1)
    v2 = torch.sum(vrel ** 2, dim=-1)
    h = r[:, 0] * vrel[:, 1] - r[:, 1] * vrel[:, 0]   # momento angolare specifico

    # energia specifica e semiasse maggiore
    eps = 0.5 * v2 - mu / r_norm
    a = -mu / (2 * eps)

    # eccentricita'
    e_sq = 1.0 + 2 * eps * h**2 / mu**2
    e = torch.sqrt(torch.clamp(e_sq, min=0.0))

    # vettore eccentricita' e anomalia vera
    ex = (h * vrel[:, 1]) / mu - r[:, 0] / r_norm
    ey = (-h * vrel[:, 0]) / mu - r[:, 1] / r_norm
    nu = torch.atan2(
        r[:, 1] * ex - r[:, 0] * ey,
        r[:, 0] * ex + r[:, 1] * ey,
    ) * -1.0

    # periodo (solo orbite legate)
    T = 2 * np.pi * torch.sqrt(torch.clamp(a, min=0.0) ** 3 / mu)
    T = torch.where(e < 1.0, T, torch.full_like(T, float("nan")))

    return {
        "energy": eps,
        "h": h,
        "e": e,
        "nu": nu,          # anomalia vera
        "T": T,
        "a": a,
        "r_norm": r_norm,
    }


def mean_anomaly_from_true(nu, e):
    """Conversione anomalia vera -> anomalia media (Keplero)."""
    # soluzione dell'equazione di Keplero per E (anomalia eccentrica)
    Ecc = 2 * np.arctan2(
        np.sqrt(max(1 - e, 0.0)) * np.sin(nu / 2),
        np.sqrt(max(1 + e, 0.0)) * np.cos(nu / 2),
    )
    M = Ecc - e * np.sin(Ecc)
    return M


# ------------------------------------------------------------------
# Estrazione di theta e lambda dai gate
# ------------------------------------------------------------------
def extract_gate_outputs(net, state, arch="v4"):
    """Esegue un forward parziale e restituisce, per ogni layer:
        - theta   : [B, 2, C]   angoli prodotti dal RotationGate
        - lambda_ : [B, 2, C]   fattori di scala prodotti dall'InvariantGate
    (per le architetture che hanno un solo tipo di gate, uno dei due
    sara' None).
    """
    B = state.shape[0]
    x_r = state.view(B, 2, 5)
    m = x_r[:, :, 0:1]          # [B, 2, 1]
    p = x_r[:, :, 1:3]
    v = x_r[:, :, 3:5]

    vecs = torch.stack([p, v], dim=2)   # [B, 2, 2, 2]  (canali p/v)

    thetas = []
    lambdas = []

    if arch == "v5" and hasattr(net, "gates2"):
        # v5: gate invariante (gates2) + gate di rotazione (gates),
        # entrambi condizionati anche su r_ij
        r_ij = 1.0 / (torch.sum((p[:, 0] - p[:, 1]) ** 2, dim=-1, keepdim=True) + 1e-4)
        out = vecs
        for layer, g_inv, g_rot in zip(net.layers[:-1], net.gates2, net.gates):
            out = layer(out)
            # InvariantGate restituisce il vettore scalato;
            # per ottenere lambda ricostruiamo il fattore di scala
            out_scaled = g_inv(out, m, r_ij)
            # lambda = norma(out_scaled) / (norma(out) + eps)
            n_in = torch.norm(out, dim=-1, keepdim=True) + 1e-12
            n_out = torch.norm(out_scaled, dim=-1, keepdim=True)
            lam = (n_out / n_in).squeeze(-1)          # [B, 2, C]
            out = out_scaled
            # RotationGate
            out_rot = g_rot(out, m, r_ij)
            # theta si ottiene dall'angolo tra out e out_rot
            # (oppure, se il gate espone theta, usarlo direttamente)
            theta = _extract_theta_from_rotation(out, out_rot)  # [B, 2, C]
            out = out_rot
            thetas.append(theta.detach())
            lambdas.append(lam.detach())
        # ultimo layer lineare (niente gate)
        return thetas, lambdas

    elif hasattr(net, "gates") and hasattr(net, "gates2"):
        # v3 / v4: due gate (ordine dipende dalla versione)
        out = vecs
        for layer, g_rot, g_inv in zip(net.layers[:-1], net.gates, net.gates2):
            out = layer(out)
            # prima rotazione (v3) oppure prima scala (v4) – gestiamo entrambi
            if arch == "v4":
                out_scaled = g_inv(out, m)
                n_in = torch.norm(out, dim=-1, keepdim=True) + 1e-12
                n_out = torch.norm(out_scaled, dim=-1, keepdim=True)
                lam = (n_out / n_in).squeeze(-1)
                out = out_scaled
                out_rot = g_rot(out, m)
                theta = _extract_theta_from_rotation(out, out_rot)
                out = out_rot
            else:  # v3
                out_rot = g_rot(out, m)
                theta = _extract_theta_from_rotation(out, out_rot)
                out = out_rot
                out_scaled = g_inv(out, m)
                n_in = torch.norm(out, dim=-1, keepdim=True) + 1e-12
                n_out = torch.norm(out_scaled, dim=-1, keepdim=True)
                lam = (n_out / n_in).squeeze(-1)
                out = out_scaled
            thetas.append(theta.detach())
            lambdas.append(lam.detach())
        return thetas, lambdas

    elif hasattr(net, "gates"):
        # v2: solo RotationGate
        out = vecs
        for layer, g_rot in zip(net.layers[:-1], net.gates):
            out = layer(out)
            out_rot = g_rot(out, m)
            theta = _extract_theta_from_rotation(out, out_rot)
            out = out_rot
            thetas.append(theta.detach())
            lambdas.append(None)
        return thetas, lambdas

    else:
        raise RuntimeError(
            "Architettura non riconosciuta: serve almeno un attributo "
            "`.gates` (RotationGate) o `.gates2` (InvariantGate)."
        )


def _extract_theta_from_rotation(v_in, v_out):
    """Stima l'angolo di rotazione applicato canale-per-canale
    confrontando v_in e v_out (entrambi [B, 2, C, 2]).
    Usa atan2 sul prodotto vettoriale / scalare.
    """
    # media sui due corpi non e' necessaria: calcoliamo per ogni corpo
    cross = v_in[..., 0] * v_out[..., 1] - v_in[..., 1] * v_out[..., 0]
    dot = (v_in * v_out).sum(dim=-1)
    theta = torch.atan2(cross, dot)          # [B, 2, C]
    return theta


# ------------------------------------------------------------------
# Two-NN (Facco et al. 2017) – identico allo spirito del PDF
# ------------------------------------------------------------------
def two_nn_id(X, discard_fraction=0.1):
    """Stima la dimensione intrinseca di un point-cloud X [n_samples, n_feat]."""
    X = np.unique(X, axis=0)
    if len(X) < 20:
        return float("nan")
    tree = cKDTree(X)
    dists, _ = tree.query(X, k=3)
    r1, r2 = dists[:, 1], dists[:, 2]
    valid = r1 > 1e-12
    mu = r2[valid] / r1[valid]
    mu = mu[mu > 1.0]
    if len(mu) < 10:
        return float("nan")
    mu_sorted = np.sort(mu)
    cutoff = max(int(len(mu_sorted) * (1.0 - discard_fraction)), 10)
    mu_used = mu_sorted[:cutoff]
    return float(len(mu_used) / np.sum(np.log(mu_used)))


# ------------------------------------------------------------------
# Neighborhood Overlap (versione semplice, continua)
# ------------------------------------------------------------------
def neighborhood_overlap(feats, target, k=15, n_components = 10):
    n = feats.shape[0]
    tree_f = cKDTree(feats)
    _, idx_f = tree_f.query(feats, k=k + 1)
    idx_f = idx_f[:, 1:]

    tree_t = cKDTree(target.reshape(-1, 1))
    _, idx_t = tree_t.query(target.reshape(-1, 1), k=k + 1)
    idx_t = idx_t[:, 1:]

    overlaps = np.array(
        [len(set(idx_f[i]) & set(idx_t[i])) for i in range(n)]
    ) / k
    return float(overlaps.mean())


# ------------------------------------------------------------------
# Probe principali
# ------------------------------------------------------------------
def probe_intrinsic_dimension_gates(thetas, lambdas):
    """Two-NN su theta e su lambda, layer per layer."""
    print("\n[1] Dimensione intrinseca (Two-NN) dei gate")

    n_layers = len(thetas)
    for li in range(n_layers):
        # theta: [B, 2, C] -> flatten
        th = thetas[li]
        if th is not None:
            X_th = th.reshape(th.shape[0], -1).cpu().numpy()
            d_th = two_nn_id(X_th)
            print(f"    layer {li}  theta   : embedding={X_th.shape[1]:3d}   ID≈{d_th:.2f}")
        else:
            print(f"    layer {li}  theta   : non presente")

        # lambda
        lam = lambdas[li]
        if lam is not None:
            X_lam = lam.reshape(lam.shape[0], -1).cpu().numpy()
            d_lam = two_nn_id(X_lam)
            print(f"    layer {li}  lambda  : embedding={X_lam.shape[1]:3d}   ID≈{d_lam:.2f}")
        else:
            print(f"    layer {li}  lambda  : non presente")


def probe_linear_vs_actions(thetas, lambdas, elems):
    print("\n[2] Probing lineare: gate -> variabili angolo-azione")

    targets = {
        "energia": elems["energy"].cpu().numpy(),
        "|L|": elems["h"].abs().cpu().numpy(),
        "eccentricita'": elems["e"].cpu().numpy(),
        # "anomalia_vera" tolta da qui, gestita a parte perché circolare
    }
    nu_xy = elems["nu_xy"].cpu().numpy()   # [B, 2] = (cos nu, sin nu)

    n_layers = len(thetas)
    n_train = int(0.8 * len(elems["energy"]))

    for li in range(n_layers):
        print(f"  --- layer {li} ---")
        for name, gate in [("theta", thetas[li]), ("lambda", lambdas[li])]:
            if gate is None:
                continue
            X = gate.reshape(gate.shape[0], -1).cpu().numpy()
            X_tr = np.concatenate([X[:n_train], np.ones((n_train, 1))], axis=1)
            X_te = np.concatenate([X[n_train:], np.ones((X.shape[0] - n_train, 1))], axis=1)

            r2s = []
            for tname, y in targets.items():
                y_tr, y_te = y[:n_train], y[n_train:]
                coef, *_ = np.linalg.lstsq(X_tr, y_tr, rcond=None)
                pred = X_te @ coef
                ss_res = np.sum((y_te - pred) ** 2)
                ss_tot = np.sum((y_te - y_te.mean()) ** 2)
                r2 = 1.0 - ss_res / (ss_tot + 1e-12)
                r2s.append(f"{tname}: R²={r2:.3f}")

            # --- caso speciale: anomalia vera (circolare) ---
            y_tr_xy, y_te_xy = nu_xy[:n_train], nu_xy[n_train:]
            coef_xy, *_ = np.linalg.lstsq(X_tr, y_tr_xy, rcond=None)   # [n_feat, 2]
            pred_xy = X_te @ coef_xy
            nu_pred = np.arctan2(pred_xy[:, 1], pred_xy[:, 0])
            nu_true = np.arctan2(y_te_xy[:, 1], y_te_xy[:, 0])
            ang_err = np.arctan2(np.sin(nu_pred - nu_true), np.cos(nu_pred - nu_true))
            r2_circ = 1.0 - np.mean(ang_err ** 2) / (np.var(nu_true) + 1e-12)
            r2s.append(f"anomalia_vera: R²circ={r2_circ:.3f}")

            print(f"    {name:6s}  " + "  ".join(r2s))


def probe_neighborhood_overlap_gates(thetas, lambdas, elems, k=15):
    """Neighborhood Overlap tra i gate e le variabili angolo-azione.
    Cattura relazioni non lineari che la regressione lineare non vede.
    """
    print(f"\n[3] Neighborhood Overlap (k={k}) gate vs variabili angolo-azione")
    print("    (valori vicini a 1 = i vicini nello spazio del gate "
          "sono anche vicini nella quantita' fisica)")

    targets = {
        "energia": elems["energy"].cpu().numpy(),
        "|L|": elems["h"].abs().cpu().numpy(),
        "eccentricita'": elems["e"].cpu().numpy(),
        "anomalia_vera": elems["nu_xy"].cpu().numpy()
    }

    n_layers = len(thetas)
    for li in range(n_layers):
        print(f"  --- layer {li} ---")
        for name, gate in [("theta", thetas[li]), ("lambda", lambdas[li])]:
            if gate is None:
                continue
            X = gate.reshape(gate.shape[0], -1).cpu().numpy()
            parts = []
            for tname, tgt in targets.items():
                no = neighborhood_overlap(X, tgt, k=k)
                parts.append(f"{tname}: NO={no:.3f}")
            print(f"    {name:6s}  " + "  ".join(parts))


def probe_theta_vs_true_anomaly(thetas, elems, n_traj_plot=6):
    print("\n[4] Correlazione theta vs anomalia vera (punto per punto)")

    nu = elems["nu"].cpu().numpy()
    n_layers = len(thetas)

    best_overall = None
    for li in range(n_layers):
        th = thetas[li]
        if th is None:
            continue
        th_np = th.cpu().numpy()
        B, Nbody, C = th_np.shape
        for c in range(C):
            theta_c = th_np[:, 0, c]

            # differenza circolare punto per punto, NIENTE unwrap cumulativo
            diff = theta_c - nu
            diff = np.arctan2(np.sin(diff), np.cos(diff))   # wrap in (-pi, pi]

            # offset costante = media circolare della differenza
            offset = np.arctan2(np.mean(np.sin(diff)), np.mean(np.cos(diff)))

            resid = diff - offset
            resid = np.arctan2(np.sin(resid), np.cos(resid))  # ri-wrappa il residuo
            std = np.std(resid)

            if best_overall is None or std < best_overall[0]:
                best_overall = (std, li, c, offset)

    if best_overall is not None:
        std, li, c, offset = best_overall
        print(f"    Miglior canale: layer {li}, canale {c}")
        print(f"    std(theta - nu - offset) = {std:.4f} rad  "
              f"(offset = {offset:.3f} rad)")
        if std < 0.3:
            print("    → segnale forte: questo canale sembra tracciare l'anomalia vera.")
        elif std < 0.8:
            print("    → segnale moderato.")
        else:
            print("    → nessun canale segue in modo convincente l'anomalia vera.")
    else:
        print("    Nessun theta disponibile.")


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--arch", type=str, default="v4",
                        choices=["v2", "v3", "v4", "v5"])
    parser.add_argument("--n_blocks", type=int, default=4)
    parser.add_argument("--n_traj", type=int, default=1500,
                        help="Numero di stati casuali su cui fare l'analisi")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    DTYPE = torch.float64

    # Costruzione rete
    if args.arch == "v5":
        assert Acceleration2BodyNetv5 is not None, "Acceleration2BodyNetv5 non trovata"
        net = Acceleration2BodyNetv5(
            num_blocks=args.n_blocks, dtype=DTYPE, device=DEVICE
        ).to(DEVICE)
    elif args.arch == "v4":
        assert Acceleration2BodyNetv4 is not None
        net = Acceleration2BodyNetv4(
            num_blocks=args.n_blocks, dtype=DTYPE, device=DEVICE
        ).to(DEVICE)
    elif args.arch == "v3":
        assert Acceleration2BodyNetv3 is not None
        net = Acceleration2BodyNetv3(
            num_blocks=args.n_blocks, dtype=DTYPE, device=DEVICE
        ).to(DEVICE)
    else:
        assert Acceleration2BodyNetv2 is not None
        net = Acceleration2BodyNetv2(
            num_blocks=args.n_blocks, dtype=DTYPE, device=DEVICE
        ).to(DEVICE)

    ckpt = torch.load(args.checkpoint, map_location=DEVICE)
    net.load_state_dict(ckpt.get("model", ckpt))
    net.eval()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    print(f"Caricato checkpoint: {args.checkpoint}")
    print(f"Architettura: {args.arch}  |  n_blocks={args.n_blocks}")
    print(f"Genero {args.n_traj} stati casuali ...")

    states = generate_instance_2body(args.n_traj, DEVICE, dtype=DTYPE, G=1.0)
    elems = orbital_elements(states, G=1.0)

    # Estrazione gate
    print("Estrazione di theta e lambda dai gate ...")
    thetas, lambdas = extract_gate_outputs(net, states, arch=args.arch)

    # Probe
    elems["nu_xy"] = torch.stack([torch.cos(elems["nu"]), torch.sin(elems["nu"])], dim=-1)
    probe_intrinsic_dimension_gates(thetas, lambdas)
    probe_linear_vs_actions(thetas, lambdas, elems)
    probe_neighborhood_overlap_gates(thetas, lambdas, elems, k=15)
    probe_theta_vs_true_anomaly(thetas, elems)

    print("\nAnalisi completata.")


if __name__ == "__main__":
    main()
