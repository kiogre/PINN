"""
Analisi di cosa la rete rappresenta internamente, generalizzata per
funzionare su Acceleration2BodyNet (v2-v4, AB_Block, gate condizionato solo
sulla massa), Acceleration2BodyNetv5 (gate doppio condizionato anche su
r_ij), e NBodyAccelerationNet (N corpi, NBodyPermutationBlock).

Probe disponibili (i probe [3] e [4] restano specifici del 2-corpi: si
basano su elementi orbitali Kepleriani -- periodo, anomalia media -- che
non esistono in generale per N>=3, dove il problema e' tipicamente caotico
e non-integrabile):

  0. Self-check di equivarianza SO(2) dell'estrazione delle attivazioni.
     Va eseguito PRIMA di fidarsi di ogni altro probe: se extract_layer_
     activations non rispecchia esattamente il forward pass reale (es.
     nuova architettura non ancora gestita, o gate applicato con ordine/
     argomenti sbagliati), i probe [2],[5],[6],[8] -- che assumono norme e
     angoli relativi invarianti -- danno risultati silenziosamente falsi.

  1. Decomposizione simmetrica/antisimmetrica di ogni blocco di mixing
     (AB_Block per N=2, NBodyPermutationBlock per N generico), con
     baseline random per interpretare il rapporto ||self-other||/||self+(N-1)other||.

  2. Probing lineare: norme dei canali per layer -> quantita' conservate
     (energia, |momento angolare|, e per N=2 anche eccentricita').

  3. Probe angolare per canale singolo, fase assoluta vs moto medio atteso
     (solo N=2; soffre della non-linearita' dell'equazione di Keplero per
     orbite eccentriche -- vedi probe [8] per un'alternativa invariante).

  4. Decodifica lineare globale dell'angolo, con estrapolazione temporale
     (solo N=2).

  5. Dimensione intrinseca (Two-NN) per layer, su feature invarianti
     (norme + angoli relativi canale-vs-r_01).

  6. Neighborhood overlap rappresentazione vs quantita' fisiche, sulle
     stesse feature invarianti del probe [5].

  7. Confronto diretto dei parametri self/other tra reti a N diverso.

  8. Probe dell'angolo relativo (canale vs r_01), invariante punto per
     punto: confrontato direttamente con l'anomalia vera istantanea
     (nessuna assunzione di crescita lineare nel tempo, quindi non soffre
     del problema Kepleriano del probe [3]). Solo N=2 per il confronto col
     ground truth; per N>=3 non esiste un target Kepleriano di riferimento.

Uso:
    python analyze_representations.py --checkpoint <path> --n_obj 2
    python analyze_representations.py --checkpoint <path> --n_obj 3
"""

import argparse
from collections import Counter

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

from train import generate_instance as generate_instance_2body

from networks import Acceleration2BodyNetv2, Acceleration2BodyNetv3, Acceleration2BodyNetv4

try:
    from networks import Acceleration2BodyNet
except ImportError:
    Acceleration2BodyNet = None

try:
    from networks_2 import Acceleration2BodyNetv5
except ImportError:
    Acceleration2BodyNetv5 = None

try:
    from networks import AccelerationNBodyNetv4
except ImportError:
    AccelerationNBodyNetv4 = None

try:
    import train_nbody as tnb
except ImportError:
    tnb = None

try:
    from networks_2 import AccelerationNBodyNetv5
except ImportError:
    AccelerationNBodyNetv5 = None


def _generate_states(n, n_obj, device, dtype, G=1.0):
    """Wrapper unico per generare stati iniziali, 2 corpi o N corpi."""
    if n_obj == 2:
        return generate_instance_2body(n, device, dtype=dtype, G=G)
    assert tnb is not None, "train_nbody.py non trovato: serve per generate_instance a N corpi"
    return tnb.generate_instance(n, n_obj, device, dtype=dtype, G=G)


# ----------------------------------------------------------------------
# 1. Decomposizione simmetrica / antisimmetrica (AB_Block o NBodyPermutationBlock)
# ----------------------------------------------------------------------
def find_mixing_blocks(module, prefix=""):
    """Trova ricorsivamente ogni blocco di mixing self/other, sia che sia
    un AB_Block (.A/.B, N=2 implicito) sia un NBodyPermutationBlock
    (.lin_self/.lin_other, N generico)."""
    found = []
    for name, child in module.named_children():
        full_name = f"{prefix}.{name}" if prefix else name
        if hasattr(child, "A") and hasattr(child, "B") and isinstance(getattr(child, "A", None), torch.nn.Parameter):
            found.append((full_name, "ab_block", child))
        elif hasattr(child, "lin_self") and hasattr(child, "lin_other"):
            found.append((full_name, "nbody_block", child))
        else:
            found.extend(find_mixing_blocks(child, full_name))
    return found


def _sym_antisym_ratios(net, n_obj):
    """other_mult = N-1: per N=2 (AB_Block) si riduce esattamente ad A+B, A-B."""
    other_mult = max(n_obj - 1, 1)
    blocks = find_mixing_blocks(net)
    ratios = {}
    for name, kind, block in blocks:
        if kind == "ab_block":
            self_w, other_w = block.A.detach(), block.B.detach()
        else:
            self_w, other_w = block.lin_self.weight.detach(), block.lin_other.weight.detach()
        sym_norm = (self_w + other_mult * other_w).norm().item()
        antisym_norm = (self_w - other_w).norm().item()
        ratios[name] = (sym_norm, antisym_norm, antisym_norm / (sym_norm + 1e-8))
    return ratios


def report_symmetric_antisymmetric(net, net_random_ctor, n_obj, n_random_seeds=5):
    """Confronta i rapporti sim/antisim della rete allenata contro reti
    fresche (stessa architettura, pesi mai allenati) per capire se lo
    scostamento e' reale o compatibile col caso (A, B sono i.i.d. Gaussiani
    all'init, quindi un rapporto ~1 e' atteso anche senza training)."""
    print(f"\n[1] Decomposizione simmetrica/antisimmetrica (N={n_obj})")
    trained = _sym_antisym_ratios(net, n_obj)
    if not trained:
        print("    Nessun blocco di mixing trovato (architettura non riconosciuta).")
        return

    random_ratios = {name: [] for name in trained}
    for seed in range(n_random_seeds):
        torch.manual_seed(1000 + seed)
        net_r = net_random_ctor()
        for name, (_, _, r) in _sym_antisym_ratios(net_r, n_obj).items():
            random_ratios[name].append(r)

    print(f"    {'layer':40s}  {'rapporto allenato':>18s}  {'range random (n=' + str(n_random_seeds) + ')':>22s}  fuori range?")
    for name, (sym_n, antisym_n, ratio) in trained.items():
        rr = random_ratios[name]
        lo, hi = min(rr), max(rr)
        out_of_range = ratio < lo or ratio > hi
        flag = "  <-- SI, scostamento reale" if out_of_range else "  no, compatibile col caso"
        print(f"    {name:40s}  {ratio:18.3f}  [{lo:.3f}, {hi:.3f}]{'':>6s}{flag}")


# ----------------------------------------------------------------------
# Estrazione delle attivazioni intermedie, unificata a [B, N, C, 2]
# ----------------------------------------------------------------------
def _r_ij_inv_sq(p, eps=1e-4):
    """1/(r_01^2 + eps), stesso identico calcolo di predict_acceleration
    in Acceleration2BodyNetv5 -- deve restare sincronizzato con quello."""
    return 1.0 / (torch.sum((p[:, 0] - p[:, 1]) ** 2, dim=-1, keepdim=True) + eps)


def extract_layer_activations(net, state, n_obj):
    B = state.shape[0]
    N = n_obj
    x_r = state.view(B, N, 5)
    m = x_r[:, :, 0:1]
    p = x_r[:, :, 1:3]
    v = x_r[:, :, 3:5]

    activations = []

    # 1. Rete 2-corpi v5 (gate condizionati su m e r_ij scalare)
    if n_obj == 2 and hasattr(net, "gates2"):
        vecs = torch.stack([p, v], dim=2)  # [B, 2, C, 2]
        r_ij = _r_ij_inv_sq(p)
        out = vecs
        for layer, gate2, gate in zip(net.layers[:-1], net.gates2, net.gates):
            out = layer(out)
            out = gate2(out, m, r_ij)
            out = gate(out, m, r_ij)
            activations.append(out.detach().clone())
        out = net.layers[-1](out)
        activations.append(out.detach().clone())

    # 2. Rete 2-corpi v2-v4 (singolo gate su m)
    elif n_obj == 2 and hasattr(net, "gates"):
        vecs = torch.stack([p, v], dim=2)
        out = vecs
        for layer, gate in zip(net.layers[:-1], net.gates):
            out = layer(out)
            out = gate(out, m)
            activations.append(out.detach().clone())
        out = net.layers[-1](out)
        activations.append(out.detach().clone())

    # 3. Nuova rete N-corpi v5 (gate condizionati su m e r_features aggregate)
    elif hasattr(net, "gates_inv") and hasattr(net, "_compute_r_features"):
        vecs = torch.stack([p, v], dim=-1)  # [B, N, 2, 2]
        r_feats = net._compute_r_features(p, m)
        out = vecs
        for layer, g_inv, g_rot in zip(net.layers[:-1], net.gates_inv, net.gates_rot):
            out = layer(out)
            out = g_inv(out, m, r_feats)
            out = g_rot(out, m, r_feats)
            activations.append(out.transpose(-1, -2).detach().clone())  # -> [B, N, C, 2]
        out = net.layers[-1](out)
        activations.append(out.transpose(-1, -2).detach().clone())

    # 4. Rete N-corpi v4 (gates_inv e gates_rot senza distanze r)
    elif hasattr(net, "gates_inv"):
        vecs = torch.stack([p, v], dim=-1)
        out = vecs
        for layer, g_inv, g_rot in zip(net.layers[:-1], net.gates_inv, net.gates_rot):
            out = layer(out)
            out = g_inv(out, m)
            out = g_rot(out, m)
            activations.append(out.transpose(-1, -2).detach().clone())
        out = net.layers[-1](out)
        activations.append(out.transpose(-1, -2).detach().clone())

    # 5. Vecchia NBodyAccelerationNet senza gate (solo SiLU)
    else:
        vecs = torch.stack([p, v], dim=-1)
        out = vecs
        for layer in net.layers[:-1]:
            out = layer(out)
            out = F.silu(out)
            activations.append(out.transpose(-1, -2).detach().clone())
        out = net.layers[-1](out)
        activations.append(out.transpose(-1, -2).detach().clone())

    return activations
    """Ritorna una lista di tensori [B, N, C, 2], indipendentemente
    dall'architettura. Replica ESATTAMENTE l'ordine di operazioni del
    forward/predict_acceleration reale di ciascuna rete -- e' il pezzo
    piu' delicato dello script: se non rispecchia il forward vero, tutti
    i probe a valle (che assumono norme/angoli invarianti) sono silenziosamente
    sbagliati. Usare probe [0] per verificarlo empiricamente dopo ogni
    modifica qui o ogni nuova architettura aggiunta."""
    B = state.shape[0]
    N = n_obj
    x_r = state.view(B, N, 5)
    m = x_r[:, :, 0:1]
    p = x_r[:, :, 1:3]
    v = x_r[:, :, 3:5]

    activations = []

    if n_obj == 2 and hasattr(net, "gates2"):
        # Acceleration2BodyNetv5: layer -> gate invariante (gates2, scala) ->
        # gate di rotazione (gates), entrambi condizionati su m e su r_ij.
        vecs = torch.stack([p, v], dim=2)  # [B, 2, 2(canale p/v), 2(xy)]
        r_ij = _r_ij_inv_sq(p)
        out = vecs
        for layer, gate2, gate in zip(net.layers[:-1], net.gates2, net.gates):
            out = layer(out)
            out = gate2(out, m, r_ij)
            out = gate(out, m, r_ij)
            activations.append(out.detach().clone())
        out = net.layers[-1](out)
        activations.append(out.detach().clone())

    elif n_obj == 2 and hasattr(net, "gates"):
        # Acceleration2BodyNet (v2-v4): singolo gate condizionato solo su m.
        vecs = torch.stack([p, v], dim=2)
        out = vecs
        for layer, gate in zip(net.layers[:-1], net.gates):
            out = layer(out)
            out = gate(out, m)
            activations.append(out.detach().clone())
        out = net.layers[-1](out)
        activations.append(out.detach().clone())

    else:
        # NBodyAccelerationNet: vecs = [B, N, 2(xy), C], niente gate, solo SiLU.
        # ATTENZIONE: SiLU applicata elemento per elemento sulle componenti x,y
        # grezze non e' equivariante SO(2) in generale -- se probe [0] segnala
        # errore qui, la rete vera probabilmente non passa da questo branch
        # cosi' com'e' (es. usa un gate norm-based non replicato in questa
        # funzione) e va corretta prima di fidarsi di [2],[5],[6],[8].
        vecs = torch.stack([p, v], dim=-1)  # [B, N, 2, 2(canale p/v)]
        out = vecs
        for layer in net.layers[:-1]:
            out = layer(out)
            out = F.silu(out)
            activations.append(out.transpose(-1, -2).detach().clone())  # -> [B, N, C, 2]
        out = net.layers[-1](out)
        activations.append(out.transpose(-1, -2).detach().clone())

    return activations


# ----------------------------------------------------------------------
# 0. Self-check di equivarianza SO(2) dell'estrazione
# ----------------------------------------------------------------------
def check_equivariance(net, device, dtype, n_obj, n_traj=8, seed=123, theta_val=0.7):
    """Ruota lo stato di un angolo fisso e confronta le attivazioni estratte
    con le attivazioni originali ruotate della stessa quantita'. Se
    extract_layer_activations rispecchia il vero forward pass equivariante,
    lo scarto deve essere a precisione numerica (~1e-14/1e-6 a seconda del
    dtype), esattamente come gia' verificato per la rete intera. Se non lo
    e', i probe successivi basati su norme/angoli invarianti NON sono validi."""
    print(f"\n[0] Self-check equivarianza SO(2) dell'estrazione (N={n_obj})")
    torch.manual_seed(seed)
    state = _generate_states(n_traj, n_obj, device, dtype, G=1.0)

    theta = torch.tensor(theta_val, dtype=dtype, device=device)
    c, s = torch.cos(theta), torch.sin(theta)
    R = torch.stack([torch.stack([c, -s]), torch.stack([s, c])]).to(dtype=dtype, device=device)

    B = state.shape[0]
    x_r = state.view(B, n_obj, 5)
    m, p, v = x_r[:, :, 0:1], x_r[:, :, 1:3], x_r[:, :, 3:5]
    p_rot = p @ R.T
    v_rot = v @ R.T
    state_rot = torch.cat([m, p_rot, v_rot], dim=-1).view(B, -1)

    acts_orig = extract_layer_activations(net, state, n_obj)
    acts_rot = extract_layer_activations(net, state_rot, n_obj)

    max_err = 0.0
    for li, (a0, a1) in enumerate(zip(acts_orig, acts_rot)):
        a0_rot = a0 @ R.T  # ruota le componenti x,y delle attivazioni originali
        err = (a0_rot - a1).abs().max().item()
        max_err = max(max_err, err)
        print(f"    layer {li}: errore max |R@act(x) - act(R@x)| = {err:.3e}")

    if max_err < 1e-6:
        print("    OK: estrazione equivariante a precisione numerica -- probe [2],[5],[6],[8] validi.")
    else:
        print("    !!! ATTENZIONE: estrazione NON equivariante -- probe [2],[5],[6],[8] "
              "NON sono affidabili cosi' come sono. Correggere extract_layer_activations "
              "prima di interpretare i risultati sotto.")
    return max_err


# ----------------------------------------------------------------------
# Angolo relativo tra vettori 2D, invariante SO(2) punto per punto
# ----------------------------------------------------------------------
def _relative_angle(u, v):
    """Angolo con segno tra u e v (ultimo asse = x,y), via atan2(cross, dot).
    A differenza dell'angolo assoluto di un canale, e' invariante per
    rotazioni globali: ruotare u e v della stessa quantita' non lo cambia.
    Supporta il broadcasting standard di torch tra le shape di u e v."""
    cross = u[..., 0] * v[..., 1] - u[..., 1] * v[..., 0]
    dot = (u * v).sum(-1)
    return torch.atan2(cross, dot)


def _invariant_features(act, r_ref):
    """Feature invarianti a rotazione per un singolo layer: norme dei canali
    + coseno/seno dell'angolo di ogni canale rispetto a r_ref (posizione
    relativa corpo1-corpo0). Le norme da sole catturano solo le quantita'
    scalari conservate (energia, |L|, eccentricita'); l'angolo aggiunge la
    fase/orientazione orbitale, che e' invariante ma non e' una norma."""
    n = act.shape[0]
    norms = torch.norm(act, dim=-1).reshape(n, -1)
    ang = _relative_angle(act, r_ref)  # broadcasting: r_ref e' [B,1,1,2]
    feats = torch.cat([norms, torch.cos(ang).reshape(n, -1), torch.sin(ang).reshape(n, -1)], dim=-1)
    return feats.cpu().numpy()


# ----------------------------------------------------------------------
# Elementi orbitali analitici (SOLO 2 corpi, problema di Keplero)
# ----------------------------------------------------------------------
def orbital_elements(state, G=1.0):
    m1, m2 = state[:, 0], state[:, 5]
    mu = G * (m1 + m2)

    r = state[:, 6:8] - state[:, 1:3]
    vrel = state[:, 8:10] - state[:, 3:5]

    r_norm = torch.norm(r, dim=-1)
    v2 = torch.sum(vrel ** 2, dim=-1)

    h = r[:, 0] * vrel[:, 1] - r[:, 1] * vrel[:, 0]

    eps_energy = 0.5 * v2 - mu / r_norm
    a = -mu / (2 * eps_energy)

    e_sq = 1 + 2 * eps_energy * h ** 2 / mu ** 2
    e = torch.sqrt(torch.clamp(e_sq, min=0.0))

    ex = (h * vrel[:, 1]) / mu - r[:, 0] / r_norm
    ey = (-h * vrel[:, 0]) / mu - r[:, 1] / r_norm
    nu = torch.atan2(r[:, 1] * ex - r[:, 0] * ey, r[:, 0] * ex + r[:, 1] * ey) * -1

    T = 2 * np.pi * torch.sqrt(torch.clamp(a, min=0.0) ** 3 / mu)
    T = torch.where(e < 1.0, T, torch.full_like(T, float('nan')))

    return dict(mu=mu, a=a, e=e, h=h, energy=eps_energy, nu=nu, T=T, r_norm=r_norm)


def _mean_anomaly_from_true(nu, e):
    Ecc = 2 * np.arctan2(np.sqrt(max(1 - e, 0.0)) * np.sin(nu / 2),
                          np.sqrt(max(1 + e, 0.0)) * np.cos(nu / 2))
    M = Ecc - e * np.sin(Ecc)
    return M


# ----------------------------------------------------------------------
# 2. Probing lineare: norme dei canali -> quantita' conservate
# ----------------------------------------------------------------------
def probe_linear_regression(net, device, dtype, n_obj, n_traj=500, G=1.0):
    print(f"\n[2] Probing lineare: norme dei canali per layer -> quantita' conservate (N={n_obj})")

    if n_obj == 2:
        states = generate_instance_2body(n_traj, device, dtype=dtype, G=G)
        elems = orbital_elements(states, G=G)
        targets = np.stack([elems["energy"].cpu().numpy(),
                             elems["h"].abs().cpu().numpy(),
                             elems["e"].cpu().numpy()], axis=1)
        target_names = ["energia", "|momento angolare|", "eccentricita'"]
    else:
        assert tnb is not None, "train_nbody.py non trovato: serve per generate_instance/compute_energy a N corpi"
        states = tnb.generate_instance(n_traj, n_obj, device, dtype=dtype, G=G)
        E = tnb.compute_energy(states, G=G)
        L = tnb.compute_angular_momentum(states)
        targets = np.stack([E.cpu().numpy(), L.abs().cpu().numpy()], axis=1)
        target_names = ["energia", "|momento angolare|"]

    activations = extract_layer_activations(net, states, n_obj)

    n_train = int(0.8 * n_traj)
    for layer_idx, act in enumerate(activations):
        norms = torch.norm(act, dim=-1)              # [B, N, C]
        feats = norms.reshape(n_traj, -1).cpu().numpy()

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
# 3. Probe angolare per canale singolo (SOLO 2 corpi)
# ----------------------------------------------------------------------
def _angle_probe_single(net, state, T_true, dt, total_t, n_obj):
    n_expected = 2 * np.pi / T_true
    current = state.clone()
    layer_angle_history = None
    t_axis = []

    with torch.no_grad():
        for step in range(total_t):
            acts = extract_layer_activations(net, current, n_obj)
            if layer_angle_history is None:
                layer_angle_history = [[] for _ in acts]
            for li, act in enumerate(acts):
                ang = torch.atan2(act[0, 0, :, 1], act[0, 0, :, 0])
                layer_angle_history[li].append(ang.cpu().numpy())
            t_axis.append(step * dt)
            current = net(current, dt)

    t_axis = np.array(t_axis)
    best = None
    for li, hist in enumerate(layer_angle_history):
        hist = np.unwrap(np.stack(hist, axis=0), axis=0)
        for c in range(hist.shape[1]):
            y = hist[:, c]
            slope, intercept = np.polyfit(t_axis, y, 1)
            pred = slope * t_axis + intercept
            ss_res = np.sum((y - pred) ** 2)
            ss_tot = np.sum((y - y.mean()) ** 2)
            r2 = 1 - ss_res / (ss_tot + 1e-12)
            if r2 > 0.98:
                rel_err = abs(abs(slope) - abs(n_expected)) / abs(n_expected)
                if best is None or rel_err < best[4]:
                    best = (li, c, slope, r2, rel_err)
    return best, t_axis, layer_angle_history


def probe_angle_channels(net, device, dtype, total_t=60, dt=0.01, G=1.0, n_trajectories=15):
    print(f"\n[3] Probe angolare su {n_trajectories} traiettorie legate diverse (solo 2 corpi)")
    candidates = generate_instance_2body(n_trajectories * 4, device, dtype=dtype, G=G)
    elems = orbital_elements(candidates, G=G)
    bound_idx = torch.nonzero(elems["e"] < 1.0, as_tuple=True)[0]
    if bound_idx.numel() < n_trajectories:
        print(f"    Solo {bound_idx.numel()} orbite legate trovate, procedo con quelle disponibili.")
    bound_idx = bound_idx[:n_trajectories]

    results = []
    for i in bound_idx.tolist():
        state = candidates[i:i + 1]
        T_true = elems["T"][i].item()
        best, t_axis, hist = _angle_probe_single(net, state, T_true, dt, total_t, n_obj=2)
        e_i = elems["e"][i].item()
        if best is None:
            print(f"    e={e_i:.3f}  T={T_true:.3f}  -> nessun canale con R2>0.98")
        else:
            li, c, slope, r2, rel_err = best
            print(f"    e={e_i:.3f}  T={T_true:.3f}  -> layer {li}, canale {c}  "
                  f"rate={slope:.4f}  R2={r2:.4f}  errore={rel_err*100:.1f}%")
            results.append((li, c, rel_err, r2, e_i, T_true))

    if not results:
        print("    Nessun candidato trovato su nessuna traiettoria.")
        return

    from collections import Counter
    winner_counts = Counter((li, c) for li, c, *_ in results)
    top_winner, count = winner_counts.most_common(1)[0]
    errs = [r for li, c, r, r2, e_i, T in results if (li, c) == top_winner]
    print(f"\n    Layer/canale piu' frequente: {top_winner}  ({count}/{len(results)} traiettorie)")
    print(f"    Errore vs n atteso su quelle traiettorie: mediana={np.median(errs)*100:.1f}%  max={np.max(errs)*100:.1f}%")
    if count < len(results) * 0.5:
        print("    ATTENZIONE: nessun canale vince in modo consistente.")


# ----------------------------------------------------------------------
# 4. Decodifica lineare GLOBALE dell'angolo (SOLO 2 corpi)
# ----------------------------------------------------------------------
def probe_global_angle_decoder(net, device, dtype, n_traj=10, dt=0.01, G=1.0,
                                k_periods=1.5, min_steps=250, max_steps=600, ridge_frac=0.1):
    print(f"\n[4] Decodifica lineare globale dell'angolo (solo 2 corpi)")
    candidates = generate_instance_2body(n_traj * 4, device, dtype=dtype, G=G)
    elems = orbital_elements(candidates, G=G)
    bound_idx = torch.nonzero(elems["e"] < 1.0, as_tuple=True)[0][:n_traj]

    n_layers = None
    per_layer_r2 = None
    per_layer_ecc = None

    for i in bound_idx.tolist():
        state0 = candidates[i:i + 1].clone()
        e_i = elems["e"][i].item()
        T_i = elems["T"][i].item()
        n_i = 2 * np.pi / T_i
        M0 = _mean_anomaly_from_true(elems["nu"][i].item(), e_i)

        total_t = int(np.clip(k_periods * T_i / dt, min_steps, max_steps))

        current = state0.clone()
        layer_feats = None
        t_axis = []
        with torch.no_grad():
            for step in range(total_t):
                acts = extract_layer_activations(net, current, n_obj=2)
                if layer_feats is None:
                    n_layers = len(acts)
                    layer_feats = [[] for _ in acts]
                    if per_layer_r2 is None:
                        per_layer_r2 = [[] for _ in acts]
                        per_layer_ecc = [[] for _ in acts]
                for li, act in enumerate(acts):
                    layer_feats[li].append(act[0].reshape(-1).cpu().numpy())
                t_axis.append(step * dt)
                current = net(current, dt)

        t_axis = np.array(t_axis)
        M_t = M0 + n_i * t_axis
        target = np.stack([np.cos(M_t), np.sin(M_t)], axis=1)

        n_train = int(0.7 * total_t)
        print(f"    e={e_i:.3f}  T={T_i:.3f}  steps={total_t} ({total_t*dt:.2f} unita' di tempo, "
              f"~{total_t*dt/T_i:.1f} periodi, n_train={n_train})")
        for li in range(n_layers):
            X = np.stack(layer_feats[li], axis=0)
            X_tr, X_te = X[:n_train], X[n_train:]
            y_tr, y_te = target[:n_train], target[n_train:]

            F_ = X.shape[1]
            X_tr_ = np.concatenate([X_tr, np.ones((X_tr.shape[0], 1))], axis=1)
            X_te_ = np.concatenate([X_te, np.ones((X_te.shape[0], 1))], axis=1)
            gram = X_tr_.T @ X_tr_
            ridge = ridge_frac * np.trace(gram) / (F_ + 1)
            W = np.linalg.solve(gram + ridge * np.eye(F_ + 1), X_tr_.T @ y_tr)
            pred = X_te_ @ W
            ss_res = np.sum((y_te - pred) ** 2)
            ss_tot = np.sum((y_te - y_te.mean(axis=0)) ** 2)
            r2 = 1 - ss_res / (ss_tot + 1e-12)
            per_layer_r2[li].append(r2)
            per_layer_ecc[li].append(e_i)
            ratio = n_train / (F_ + 1)
            print(f"      layer {li} ({F_} feature, n_train/F={ratio:.1f}): R2 estrapolazione = {r2:.3f}")

    print("\n    Riepilogo per layer (su tutte le traiettorie):")
    for li in range(n_layers):
        rs = np.array(per_layer_r2[li])
        es = np.array(per_layer_ecc[li])
        corr = np.corrcoef(es, rs)[0, 1] if len(rs) > 2 else float('nan')
        print(f"      layer {li}: R2 mediano={np.median(rs):.3f}  min={np.min(rs):.3f}  max={np.max(rs):.3f}  "
              f"corr(eccentricita', R2)={corr:.3f}")


# ----------------------------------------------------------------------
# 5. Dimensione intrinseca (Two-NN, Facco et al. 2017)
# ----------------------------------------------------------------------
def _two_nn_id(X, discard_fraction=0.1):
    """Stima la dimensione intrinseca via Two-NN. X: [n_samples, n_features].
    Rimuove duplicati esatti (altrimenti r1=0, mu indefinito) e scarta la coda
    superiore di mu (outlier locali), come nel paper originale."""
    from scipy.spatial import cKDTree
    X = np.unique(X, axis=0)
    tree = cKDTree(X)
    dists, _ = tree.query(X, k=3)
    r1, r2 = dists[:, 1], dists[:, 2]
    valid = r1 > 1e-12
    mu = r2[valid] / r1[valid]
    mu = mu[mu > 1.0]
    mu_sorted = np.sort(mu)
    cutoff = max(int(len(mu_sorted) * (1 - discard_fraction)), 10)
    mu_used = mu_sorted[:cutoff]
    return len(mu_used) / np.sum(np.log(mu_used))


def probe_intrinsic_dimension(net, device, dtype, n_obj, n_traj=2000, G=1.0):
    """Dimensione intrinseca delle rappresentazioni per layer, su feature
    invarianti (norme + angoli relativi, vedi _invariant_features): usare
    solo le norme sottostimerebbe l'ID di almeno 1 dimensione per traiettoria,
    perche' la fase/orientazione orbitale e' invariante a rotazione ma non e'
    una norma. Le componenti x,y grezze invece gonfierebbero l'ID di +1 per
    l'orientazione globale arbitraria del sample, che non e' un grado di
    liberta' fisico."""
    print(f"\n[5] Dimensione intrinseca (Two-NN) per layer (N={n_obj})")
    states = _generate_states(n_traj, n_obj, device, dtype, G=G)
    x_r = states.view(n_traj, n_obj, 5)
    r_ref = (x_r[:, 1, 1:3] - x_r[:, 0, 1:3]).unsqueeze(1).unsqueeze(1)  # [n_traj,1,1,2]

    activations = extract_layer_activations(net, states, n_obj)
    for li, act in enumerate(activations):
        feats = _invariant_features(act, r_ref)
        try:
            d_hat = _two_nn_id(feats)
        except Exception:
            d_hat = float('nan')
        print(f"    layer {li}: dimensione embedding={feats.shape[1]:4d}   ID stimata (Two-NN)={d_hat:.2f}")


# ----------------------------------------------------------------------
# 6. Neighborhood Overlap con target continuo (generalizzazione di NO)
# ----------------------------------------------------------------------
def _neighborhood_overlap(feats, target, k=10):
    """Frazione media di vicini condivisi tra il ranking per distanza nella
    rappresentazione (feats) e il ranking per vicinanza nel valore di una
    grandezza fisica continua (target). Cattura struttura NON lineare che
    la regressione lineare del probe [2] non vede."""
    from scipy.spatial import cKDTree
    n = feats.shape[0]
    tree_feat = cKDTree(feats)
    _, idx_feat = tree_feat.query(feats, k=k + 1)
    idx_feat = idx_feat[:, 1:]

    tree_gt = cKDTree(target.reshape(-1, 1))
    _, idx_gt = tree_gt.query(target.reshape(-1, 1), k=k + 1)
    idx_gt = idx_gt[:, 1:]

    overlaps = np.array([len(set(idx_feat[i]) & set(idx_gt[i])) for i in range(n)]) / k
    return overlaps.mean()


def probe_neighborhood_overlap(net, device, dtype, n_obj, n_traj=1000, G=1.0, k=10):
    print(f"\n[6] Neighborhood overlap (k={k}) rappresentazione vs quantita' fisiche (N={n_obj})")
    states = _generate_states(n_traj, n_obj, device, dtype, G=G)
    if n_obj == 2:
        elems = orbital_elements(states, G=G)
        targets = {"energia": elems["energy"].cpu().numpy(),
                   "|momento angolare|": elems["h"].abs().cpu().numpy(),
                   "eccentricita'": elems["e"].cpu().numpy()}
    else:
        targets = {"energia": tnb.compute_energy(states, G=G).cpu().numpy(),
                   "|momento angolare|": tnb.compute_angular_momentum(states).abs().cpu().numpy()}

    x_r = states.view(n_traj, n_obj, 5)
    r_ref = (x_r[:, 1, 1:3] - x_r[:, 0, 1:3]).unsqueeze(1).unsqueeze(1)

    activations = extract_layer_activations(net, states, n_obj)
    print(f"    (overlap atteso per caso puro ~ {k}/{n_traj:.0f} ~= {k/n_traj:.4f}; valori vicini a 1 = struttura forte)")
    for li, act in enumerate(activations):
        feats = _invariant_features(act, r_ref)
        parts = []
        for name, tgt in targets.items():
            no = _neighborhood_overlap(feats, tgt, k=k)
            parts.append(f"{name}: NO={no:.3f}")
        print(f"    layer {li}:  " + "  ".join(parts))


# ----------------------------------------------------------------------
# 7. Confronto diretto dei parametri self/other tra reti a N diverso
# ----------------------------------------------------------------------
def _get_self_other(block, kind):
    if kind == "ab_block":
        return block.A.detach(), block.B.detach()
    return block.lin_self.weight.detach(), block.lin_other.weight.detach()


def compare_mixing_across_networks(net_a, n_obj_a, label_a, net_b, n_obj_b, label_b):
    """Confronta blocco per blocco i pesi self/other tra due reti (tipicamente
    2 corpi vs 3 corpi, stessa hidden_channels/num_blocks). Norme, cosine
    similarity (se le shape combaciano) e spettro dei valori singolari (la
    "forma" della trasformazione, anche quando i valori esatti non coincidono
    per via del training separato)."""
    print(f"\n[7] Confronto self/other: {label_a} (N={n_obj_a}) vs {label_b} (N={n_obj_b})")
    blocks_a = find_mixing_blocks(net_a)
    blocks_b = find_mixing_blocks(net_b)
    n = min(len(blocks_a), len(blocks_b))
    if len(blocks_a) != len(blocks_b):
        print(f"    Attenzione: {len(blocks_a)} blocchi vs {len(blocks_b)}, confronto solo i primi {n}.")

    for idx in range(n):
        name_a, kind_a, block_a = blocks_a[idx]
        name_b, kind_b, block_b = blocks_b[idx]
        self_a, other_a = _get_self_other(block_a, kind_a)
        self_b, other_b = _get_self_other(block_b, kind_b)

        print(f"    blocco {idx}  [{label_a}:{name_a}]  vs  [{label_b}:{name_b}]")
        print(f"      ||self||  {label_a}={self_a.norm().item():.3f}   {label_b}={self_b.norm().item():.3f}")
        print(f"      ||other|| {label_a}={other_a.norm().item():.3f}   {label_b}={other_b.norm().item():.3f}")

        if self_a.shape == self_b.shape:
            cos_self = F.cosine_similarity(self_a.flatten(), self_b.flatten(), dim=0).item()
            cos_other = F.cosine_similarity(other_a.flatten(), other_b.flatten(), dim=0).item()
            print(f"      cosine(self_a, self_b)={cos_self:.3f}   cosine(other_a, other_b)={cos_other:.3f}  "
                  f"(shape identica: confronto diretto valido)")
        else:
            print(f"      shape diversa ({tuple(self_a.shape)} vs {tuple(self_b.shape)}): "
                  f"cosine non ha senso, salto")

        k = min(5, self_a.shape[0], self_a.shape[1], self_b.shape[0], self_b.shape[1])
        sv_a = torch.linalg.svdvals(self_a).cpu().numpy()[:k]
        sv_b = torch.linalg.svdvals(self_b).cpu().numpy()[:k]
        print(f"      top-{k} valori singolari self:  {label_a}={np.round(sv_a, 3)}   {label_b}={np.round(sv_b, 3)}")


# ----------------------------------------------------------------------
# 8. Probe dell'angolo relativo (canale vs r_01), invariante punto per punto
# ----------------------------------------------------------------------
def probe_relative_angle_channels(net, device, dtype, n_obj, total_t=60, dt=0.01, G=1.0, n_trajectories=15):
    """Alternativa al probe [3]: invece della fase ASSOLUTA di un canale
    rispetto a un asse fisso arbitrario (valida solo assumendo crescita
    lineare nel tempo, che fallisce per orbite eccentriche per via
    dell'equazione di Keplero), guarda l'angolo RELATIVO tra ogni canale e
    r_01(t) -- invariante SO(2) punto per punto, nessuna assunzione di
    linearita'. Per N=2 lo confronta direttamente con l'anomalia vera
    istantanea (calcolata dallo stato corrente, non estrapolata)."""
    print(f"\n[8] Probe angolo relativo (canale vs r_01) su {n_trajectories} traiettorie (N={n_obj})")

    if n_obj == 2:
        candidates = generate_instance_2body(n_trajectories * 4, device, dtype=dtype, G=G)
        elems0 = orbital_elements(candidates, G=G)
        bound_idx = torch.nonzero(elems0["e"] < 1.0, as_tuple=True)[0]
        if bound_idx.numel() < n_trajectories:
            print(f"    Solo {bound_idx.numel()} orbite legate trovate, procedo con quelle disponibili.")
        bound_idx = bound_idx[:n_trajectories].tolist()
    else:
        candidates = tnb.generate_instance(n_trajectories, n_obj, device, dtype=dtype, G=G)
        bound_idx = list(range(n_trajectories))
        print("    N>=3: nessun target Kepleriano di riferimento, riporto solo la variabilita' "
              "dell'angolo relativo nel tempo (non un confronto con ground truth).")

    results = []
    for i in bound_idx:
        state = candidates[i:i + 1].clone()
        angle_hist = None
        nu_hist = []
        with torch.no_grad():
            for _ in range(total_t):
                x_r = state.view(1, n_obj, 5)
                r_ref = (x_r[:, 1, 1:3] - x_r[:, 0, 1:3]).unsqueeze(1)  # [1,1,2]
                acts = extract_layer_activations(net, state, n_obj)
                if angle_hist is None:
                    angle_hist = [[] for _ in acts]
                for li, act in enumerate(acts):
                    ang = _relative_angle(act[0, 0], r_ref[0])  # [C]
                    angle_hist[li].append(ang.cpu().numpy())
                if n_obj == 2:
                    nu_hist.append(orbital_elements(state, G=G)["nu"].item())
                state = net(state, dt)

        if n_obj == 2:
            nu_t = np.unwrap(np.array(nu_hist))
            best = None
            for li, hist in enumerate(angle_hist):
                hist = np.unwrap(np.stack(hist, axis=0), axis=0)
                for c in range(hist.shape[1]):
                    diff = hist[:, c] - nu_t
                    diff = diff - np.round(diff.mean() / (2 * np.pi)) * 2 * np.pi
                    resid_std = diff.std()
                    if best is None or resid_std < best[2]:
                        best = (li, c, resid_std)
            li, c, resid_std = best
            e_i = elems0["e"][i].item()
            print(f"    e={e_i:.3f}  -> layer {li}, canale {c}  "
                  f"std(angolo_canale - anomalia_vera) = {resid_std:.4f} rad")
            results.append((li, c, resid_std, e_i))

    if n_obj == 2 and results:
        winner_counts = Counter((li, c) for li, c, *_ in results)
        top_winner, count = winner_counts.most_common(1)[0]
        stds = [s for li, c, s, e in results if (li, c) == top_winner]
        print(f"\n    Layer/canale piu' frequente: {top_winner}  ({count}/{len(results)} traiettorie)")
        print(f"    std(angolo - nu_vera) su quelle traiettorie: "
              f"mediana={np.median(stds):.4f}  max={np.max(stds):.4f} rad")
        if count < len(results) * 0.5:
            print("    ATTENZIONE: nessun canale vince in modo consistente.")


# ----------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--n_obj", type=int, default=2)
    parser.add_argument("--n_blocks", type=int, default=4)
    parser.add_argument("--checkpoint2", type=str, default=None,
                         help="Secondo checkpoint opzionale (es. rete a 3 corpi) per il confronto [7]")
    parser.add_argument("--n_obj2", type=int, default=None,
                         help="n_obj del secondo checkpoint (richiesto se --checkpoint2 e' passato)")
    parser.add_argument("--arch", type=str, default="v5", choices=["v4", "v5"],
                         help="Versione dell'architettura 2-corpi da istanziare (ignorato per n_obj>2)")
    args = parser.parse_args()

    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    DTYPE = torch.float64

    def build_net(n_obj, arch=args.arch):
        if n_obj == 2:
            if arch == "v5":
                assert Acceleration2BodyNetv5 is not None, "Acceleration2BodyNetv5 non trovata in networks_2.py"
                cls = Acceleration2BodyNetv5
            else:
                assert Acceleration2BodyNetv4 is not None, "Acceleration2BodyNetv4 non trovata in networks.py"
                cls = Acceleration2BodyNetv4
            net_ = cls(num_blocks=args.n_blocks, dtype=DTYPE, device=DEVICE).to(DEVICE)
            ctor_ = lambda: cls(num_blocks=args.n_blocks, dtype=DTYPE, device=DEVICE).to(DEVICE)
        else:
            if arch == "v5":
                assert AccelerationNBodyNetv5 is not None, "AccelerationNBodyNetv5 non trovata in networks_2.py"
                cls = AccelerationNBodyNetv5
            else:
                assert AccelerationNBodyNetv4 is not None, "AccelerationNBodyNetv4 non trovata in networks.py"
                cls = AccelerationNBodyNetv4
            net_ = cls(n_obj=n_obj, num_blocks=args.n_blocks, dtype=DTYPE, device=DEVICE).to(DEVICE)
            ctor_ = lambda: cls(n_obj=n_obj, num_blocks=args.n_blocks, dtype=DTYPE, device=DEVICE).to(DEVICE)
        return net_, ctor_

    net, net_random_ctor = build_net(args.n_obj)
    ckpt = torch.load(args.checkpoint, map_location=DEVICE)
    net.load_state_dict(ckpt.get("model", ckpt))
    net.eval()

    torch.manual_seed(0)
    np.random.seed(0)

    max_err = check_equivariance(net, DEVICE, DTYPE, args.n_obj)
    if max_err >= 1e-6:
        print("\n    Procedo comunque con i probe sottostanti, ma i risultati di [2],[5],[6],[8] "
              "vanno considerati inaffidabili finche' il problema sopra non e' risolto.\n")

    report_symmetric_antisymmetric(net, net_random_ctor, args.n_obj)
    probe_linear_regression(net, DEVICE, DTYPE, args.n_obj)
    probe_intrinsic_dimension(net, DEVICE, DTYPE, args.n_obj)
    probe_neighborhood_overlap(net, DEVICE, DTYPE, args.n_obj)
    probe_relative_angle_channels(net, DEVICE, DTYPE, args.n_obj)

    if args.n_obj == 2:
        probe_angle_channels(net, DEVICE, DTYPE)
        probe_global_angle_decoder(net, DEVICE, DTYPE)
    else:
        print(f"\n[3]/[4] Saltati per N={args.n_obj}: si basano su elementi orbitali Kepleriani "
              f"(periodo, anomalia media) che non esistono in generale per N>=3 non-integrabile.")

    if args.checkpoint2 is not None:
        assert args.n_obj2 is not None, "--n_obj2 richiesto insieme a --checkpoint2"
        net2, _ = build_net(args.n_obj2)
        ckpt2 = torch.load(args.checkpoint2, map_location=DEVICE)
        net2.load_state_dict(ckpt2.get("model", ckpt2))
        net2.eval()
        compare_mixing_across_networks(net, args.n_obj, f"N={args.n_obj}", net2, args.n_obj2, f"N={args.n_obj2}")
