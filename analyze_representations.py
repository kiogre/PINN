"""
Analisi di cosa la rete rappresenta internamente, specifica per l'architettura
equivariante (AB_Block / PermutationVectorABBlock / PermutationInvariantGate).

Tre probe, indipendenti:

  1. Decomposizione simmetrica/antisimmetrica di ogni AB_Block. La matrice
     [[A,B],[B,A]] ha autovettori noti a priori: modo simmetrico (1,1) con
     autovalore A+B (~ "centro di massa"), modo antisimmetrico (1,-1) con
     autovalore A-B (~ "coordinata relativa", dove vive tutta la dinamica
     non banale). Ci si aspetta ||A-B|| >> ||A+B|| se la rete ha imparato
     la struttura giusta.

  2. Probing lineare: le norme dei canali (invarianti, confrontabili tra
     sample diversi) ad ogni layer vengono regredite linearmente contro le
     quantità conservate vere (E, L, eccentricità) calcolate analiticamente.
     R^2 alto per un layer = quel layer porta un sottoinsieme di canali che
     si comporta da "variabile d'azione" (funzione delle sole costanti del
     moto, quasi costante lungo una traiettoria).

  3. Probe angolare: per ogni canale, verifica se il suo angolo nel frame di
     laboratorio avanza LINEARMENTE nel tempo lungo un rollout (come
     l'anomalia media M(t), non come l'anomalia vera che si muove più
     veloce al pericentro) e confronta il rate con n = 2*pi / T atteso
     dalla terza legge di Keplero.

Uso:
    python analyze_representations.py --checkpoint ./PINN_savefile/save_equivariance_acc.pt
"""

import argparse
import numpy as np
import torch
import matplotlib.pyplot as plt

from train import generate_instance
from networks import Acceleration2BodyNet


# ----------------------------------------------------------------------
# 1. Decomposizione simmetrica / antisimmetrica di ogni AB_Block
# ----------------------------------------------------------------------
def find_ab_blocks(module, prefix=""):
    """Trova ricorsivamente tutti gli AB_Block nella rete (funziona anche
    se sono annidati dentro PermutationVectorABBlock/PermutationInvariantGate)."""
    found = []
    for name, child in module.named_children():
        full_name = f"{prefix}.{name}" if prefix else name
        if hasattr(child, "A") and hasattr(child, "B") and isinstance(child.A, torch.nn.Parameter):
            found.append((full_name, child))
        else:
            found.extend(find_ab_blocks(child, full_name))
    return found


def _sym_antisym_ratios(net):
    blocks = find_ab_blocks(net)
    ratios = {}
    for name, block in blocks:
        A, B = block.A.detach(), block.B.detach()
        sym_norm = (A + B).norm().item()
        antisym_norm = (A - B).norm().item()
        ratios[name] = (sym_norm, antisym_norm, antisym_norm / (sym_norm + 1e-8))
    return ratios


def report_symmetric_antisymmetric(net, net_random_ctor, n_random_seeds=5):
    """Confronta i rapporti sim/antisim della rete allenata contro n_random_seeds
    reti fresche (stessa architettura, pesi mai allenati). Dato che A e B sono
    inizializzati i.i.d. Gaussiani, A+B e A-B hanno la STESSA distribuzione per
    costruzione: un rapporto ~1 e' quello che ci si aspetta ANCHE senza training.
    Solo uno scostamento sistematico dal range random e' evidenza di struttura
    appresa, non il valore assoluto del rapporto."""
    print("\n[1] Decomposizione simmetrica/antisimmetrica di ogni AB_Block")
    trained = _sym_antisym_ratios(net)

    random_ratios = {name: [] for name in trained}
    for seed in range(n_random_seeds):
        torch.manual_seed(1000 + seed)
        net_r = net_random_ctor()
        for name, (_, _, r) in _sym_antisym_ratios(net_r).items():
            random_ratios[name].append(r)

    print(f"    {'layer':40s}  {'rapporto allenato':>18s}  {'range random (n=' + str(n_random_seeds) + ')':>22s}  fuori range?")
    for name, (sym_n, antisym_n, ratio) in trained.items():
        rr = random_ratios[name]
        lo, hi = min(rr), max(rr)
        out_of_range = ratio < lo or ratio > hi
        flag = "  <-- SI, scostamento reale" if out_of_range else "  no, compatibile col caso"
        print(f"    {name:40s}  {ratio:18.3f}  [{lo:.3f}, {hi:.3f}]{'':>6s}{flag}")


# ----------------------------------------------------------------------
# Estrazione delle attivazioni intermedie (norme dei canali, per layer)
# ----------------------------------------------------------------------
def extract_layer_activations(net, state):
    """Replica manualmente predict_acceleration, salvando l'output vettoriale
    (v: [B, 2, C, 2]) DOPO ogni gate. Ritorna una lista di tensori.
    NOTA: assume la firma attuale di PermutationInvariantGate.forward(v, m)
    (due argomenti). Se hai aggiunto inv_r12_sq come terzo argomento, aggiorna
    la chiamata qui sotto di conseguenza."""
    B = state.shape[0]
    x_r = state.view(B, 2, 5)
    m = x_r[:, :, 0:1]
    p = x_r[:, :, 1:3]
    v = x_r[:, :, 3:5]

    vecs = torch.stack([p, v], dim=2)  # [B, 2, 2, 2]

    activations = []
    out = vecs
    for layer, gate in zip(net.layers[:-1], net.gates):
        out = layer(out)
        out = gate(out, m)
        activations.append(out.detach().clone())

    out = net.layers[-1](out)  # ultimo layer, nessun gate (residuo puro)
    activations.append(out.detach().clone())

    return activations  # lista di [B, 2, C, 2], C variabile per layer


# ----------------------------------------------------------------------
# Elementi orbitali analitici (moto relativo, problema di Keplero)
# ----------------------------------------------------------------------
def orbital_elements(state, G=1.0):
    """Da uno stato [B, 10] calcola mu, semiasse a, eccentricità e,
    momento angolare specifico h, anomalia vera nu, periodo T (solo per
    orbite legate e<1; per e>=1 ritorna T=nan)."""
    m1, m2 = state[:, 0], state[:, 5]
    mu = G * (m1 + m2)

    r = state[:, 6:8] - state[:, 1:3]   # r12 [B, 2]
    vrel = state[:, 8:10] - state[:, 3:5]

    r_norm = torch.norm(r, dim=-1)
    v2 = torch.sum(vrel ** 2, dim=-1)

    h = r[:, 0] * vrel[:, 1] - r[:, 1] * vrel[:, 0]  # momento angolare specifico (scalare, 2D)

    eps_energy = 0.5 * v2 - mu / r_norm  # energia specifica
    a = -mu / (2 * eps_energy)

    e_sq = 1 + 2 * eps_energy * h ** 2 / mu ** 2
    e = torch.sqrt(torch.clamp(e_sq, min=0.0))

    # Vettore eccentricità (per l'anomalia vera)
    # e_vec = (vrel x h)/mu - r/|r|, in 2D con h scalare: vrel x h_z = h*(vrel_y, -vrel_x)
    ex = (h * vrel[:, 1]) / mu - r[:, 0] / r_norm
    ey = (-h * vrel[:, 0]) / mu - r[:, 1] / r_norm
    nu = torch.atan2(r[:, 1] * ex - r[:, 0] * ey, r[:, 0] * ex + r[:, 1] * ey) * -1  # angolo tra r e e_vec

    T = 2 * np.pi * torch.sqrt(torch.clamp(a, min=0.0) ** 3 / mu)
    T = torch.where(e < 1.0, T, torch.full_like(T, float('nan')))

    return dict(mu=mu, a=a, e=e, h=h, energy=eps_energy, nu=nu, T=T, r_norm=r_norm)


# ----------------------------------------------------------------------
# 2. Probing lineare: norme dei canali -> quantità conservate
# ----------------------------------------------------------------------
def probe_linear_regression(net, device, dtype, n_traj=500, G=1.0):
    print("\n[2] Probing lineare: norme dei canali per layer -> (E, |h|, e)")
    states = generate_instance(n_traj, device, dtype=dtype, G=G)
    elems = orbital_elements(states, G=G)
    targets = torch.stack([elems["energy"], elems["h"].abs(), elems["e"]], dim=1).cpu().numpy()
    target_names = ["energia", "|momento angolare|", "eccentricita'"]

    activations = extract_layer_activations(net, states)

    n_train = int(0.8 * n_traj)
    for layer_idx, act in enumerate(activations):
        # act: [B, 2, C, 2] -> norme per canale, poi concateno i 2 corpi come feature
        norms = torch.norm(act, dim=-1)              # [B, 2, C]
        feats = norms.reshape(n_traj, -1).cpu().numpy()  # [B, 2*C]

        r2s = []
        for k in range(targets.shape[1]):
            y = targets[:, k]
            X_tr, y_tr = feats[:n_train], y[:n_train]
            X_te, y_te = feats[n_train:], y[n_train:]

            X_tr_ = np.concatenate([X_tr, np.ones((X_tr.shape[0], 1))], axis=1)
            X_te_ = np.concatenate([X_te, np.ones((X_te.shape[0], 1))], axis=1)
            coef, *_ = np.linalg.lstsq(X_tr_, y_tr, rcond=None)
            pred = X_te_ @ coef
            ss_res = np.sum((y_te - pred) ** 2)
            ss_tot = np.sum((y_te - y_te.mean()) ** 2)
            r2 = 1 - ss_res / (ss_tot + 1e-12)
            r2s.append(r2)

        r2_str = "  ".join(f"{n}: R2={r:.3f}" for n, r in zip(target_names, r2s))
        print(f"    layer {layer_idx} ({feats.shape[1]} feature)  {r2_str}")


# ----------------------------------------------------------------------
# 3. Probe angolare: qualche canale si comporta da "angolo" (crescita lineare)?
# ----------------------------------------------------------------------
def probe_angle_channels(net, device, dtype, total_t=60, dt=0.01, G=1.0):
    print("\n[3] Probe angolare: canali con crescita ~lineare dell'angolo nel tempo")
    state0 = generate_instance(8, device, dtype=dtype, G=G)
    elems0 = orbital_elements(state0, G=G)
    bound = elems0["e"] < 1.0
    if bound.sum() == 0:
        print("    Nessuna orbita legata nel campione, riprova.")
        return
    idx = torch.nonzero(bound, as_tuple=True)[0][0].item()  # prima orbita legata
    n_expected = (2 * np.pi / elems0["T"][idx]).item()
    print(f"    Traiettoria scelta: e={elems0['e'][idx].item():.3f}  T={elems0['T'][idx].item():.3f}  "
          f"n atteso = 2*pi/T = {n_expected:.4f} rad/unita' di tempo")

    current = state0[idx:idx + 1].clone()
    layer_angle_history = None
    t_axis = []

    with torch.no_grad():
        for step in range(total_t):
            acts = extract_layer_activations(net, current)
            if layer_angle_history is None:
                layer_angle_history = [[] for _ in acts]
            for li, act in enumerate(acts):
                # act: [1, 2, C, 2] -- angolo nel frame di laboratorio per ogni canale, corpo 0
                ang = torch.atan2(act[0, 0, :, 1], act[0, 0, :, 0])  # [C]
                layer_angle_history[li].append(ang.cpu().numpy())
            t_axis.append(step * dt)
            current = net(current, dt)

    t_axis = np.array(t_axis)
    best_fit = None  # (layer, channel, slope, r2)
    for li, hist in enumerate(layer_angle_history):
        hist = np.stack(hist, axis=0)          # [total_t, C]
        hist_unwrapped = np.unwrap(hist, axis=0)
        n_channels = hist.shape[1]
        for c in range(n_channels):
            y = hist_unwrapped[:, c]
            slope, intercept = np.polyfit(t_axis, y, 1)
            pred = slope * t_axis + intercept
            ss_res = np.sum((y - pred) ** 2)
            ss_tot = np.sum((y - y.mean()) ** 2)
            r2 = 1 - ss_res / (ss_tot + 1e-12)
            if r2 > 0.98:  # crescita quasi perfettamente lineare
                rel_err_rate = abs(abs(slope) - abs(n_expected)) / abs(n_expected)
                if best_fit is None or rel_err_rate < best_fit[4]:
                    best_fit = (li, c, slope, r2, rel_err_rate)

    if best_fit is None:
        print("    Nessun canale con crescita angolare lineare (R2>0.98) trovato.")
    else:
        li, c, slope, r2, rel_err = best_fit
        print(f"    Miglior candidato 'angolo': layer {li}, canale {c}  "
              f"rate={slope:.4f}  R2={r2:.4f}  errore vs n atteso = {rel_err*100:.1f}%")
        plt.figure()
        hist = np.unwrap(np.stack(layer_angle_history[li], axis=0)[:, c])
        plt.plot(t_axis, hist, label='angolo canale (lab frame, unwrapped)')
        plt.plot(t_axis, slope * t_axis + hist[0], '--', label=f'fit lineare (rate={slope:.3f})')
        plt.plot(t_axis, n_expected * t_axis + hist[0], ':', label=f'n atteso={n_expected:.3f}')
        plt.xlabel('tempo')
        plt.ylabel('angolo (rad, unwrapped)')
        plt.legend()
        plt.title(f'Layer {li}, canale {c}: candidato variabile angolo')
        plt.savefig('diag_4_angle_probe.png', dpi=120)
        plt.close()
        print("    Salvato diag_4_angle_probe.png")


# ----------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default="./PINN_savefile/save_equivariance_acc.pt")
    parser.add_argument("--n_blocks", type=int, default=4)
    args = parser.parse_args()

    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    DTYPE = torch.float64

    net = Acceleration2BodyNet(num_blocks=args.n_blocks, dtype=DTYPE, device=DEVICE).to(DEVICE)
    ckpt = torch.load(args.checkpoint, map_location=DEVICE)
    net.load_state_dict(ckpt["model"])
    net.eval()

    torch.manual_seed(0)
    np.random.seed(0)

    def net_random_ctor():
        return Acceleration2BodyNet(
            num_blocks=args.n_blocks,
            dtype=DTYPE,
            device=DEVICE
        ).to(DEVICE)


    report_symmetric_antisymmetric(net, net_random_ctor)
    probe_linear_regression(net, DEVICE, DTYPE)
    probe_angle_channels(net, DEVICE, DTYPE)