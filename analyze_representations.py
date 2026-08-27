"""
Analisi di cosa la rete rappresenta internamente, generalizzata per
funzionare sia su Acceleration2BodyNet (2 corpi, AB_Block) sia su
NBodyAccelerationNet (N corpi, NBodyPermutationBlock).

Quattro probe (i probe [3] e [4] restano specifici del 2-corpi: si basano
su elementi orbitali Kepleriani -- periodo, anomalia media -- che non
esistono in generale per N>=3, dove il problema e' tipicamente caotico
e non-integrabile):

  1. Decomposizione simmetrica/antisimmetrica di ogni blocco di mixing
     (AB_Block per N=2, NBodyPermutationBlock per N generico), con
     baseline random per interpretare il rapporto ||self-other||/||self+(N-1)other||.

  2. Probing lineare: norme dei canali per layer -> quantita' conservate
     (energia, |momento angolare|, e per N=2 anche eccentricita').

  3. Probe angolare per canale singolo (solo N=2).

  4. Decodifica lineare globale dell'angolo, con estrapolazione temporale
     (solo N=2).

Uso:
    python analyze_representations.py --checkpoint <path> --n_obj 2
    python analyze_representations.py --checkpoint <path> --n_obj 3
"""

import argparse
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
    from networks import AccelerationNBodyNetv4
except ImportError:
    AccelerationNBodyNetv4 = None

try:
    import train_nbody as tnb
except ImportError:
    tnb = None


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
def extract_layer_activations(net, state, n_obj):
    """Ritorna una lista di tensori [B, N, C, 2] (norma sull'ultimo asse
    per ottenere invarianti), indipendentemente dall'architettura."""
    B = state.shape[0]
    N = n_obj
    x_r = state.view(B, N, 5)
    m = x_r[:, :, 0:1]
    p = x_r[:, :, 1:3]
    v = x_r[:, :, 3:5]

    activations = []

    if n_obj == 2 and hasattr(net, "gates"):
        # Acceleration2BodyNet: vecs = [B, 2, 2, 2] (canale prima delle coordinate x/y)
        vecs = torch.stack([p, v], dim=2)  # [B, 2, 2(canale p/v), 2(xy)]
        out = vecs
        for layer, gate in zip(net.layers[:-1], net.gates):
            out = layer(out)
            out = gate(out, m)
            activations.append(out.detach().clone())
        out = net.layers[-1](out)
        activations.append(out.detach().clone())
    else:
        # NBodyAccelerationNet: vecs = [B, N, 2(xy), C], niente gate, solo SiLU
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
    """Dimensione intrinseca delle rappresentazioni per layer, usando le
    NORME dei canali (non i vettori grezzi x,y): le norme sono invarianti
    per rotazione, quindi la stima riflette gradi di liberta' fisici veri
    (es. energia, momento angolare, fase) e non il grado di liberta' banale
    dell'orientazione globale casuale del sample (che gonfierebbe l'ID di
    circa +1 senza motivo fisico se si usassero le componenti x,y grezze)."""
    print(f"\n[5] Dimensione intrinseca (Two-NN) per layer (N={n_obj})")
    if n_obj == 2:
        states = generate_instance_2body(n_traj, device, dtype=dtype, G=G)
    else:
        states = tnb.generate_instance(n_traj, n_obj, device, dtype=dtype, G=G)

    activations = extract_layer_activations(net, states, n_obj)
    for li, act in enumerate(activations):
        norms = torch.norm(act, dim=-1)
        feats = norms.reshape(n_traj, -1).cpu().numpy()
        try:
            d_hat = _two_nn_id(feats)
        except Exception as e:
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
    if n_obj == 2:
        states = generate_instance_2body(n_traj, device, dtype=dtype, G=G)
        elems = orbital_elements(states, G=G)
        targets = {"energia": elems["energy"].cpu().numpy(),
                   "|momento angolare|": elems["h"].abs().cpu().numpy(),
                   "eccentricita'": elems["e"].cpu().numpy()}
    else:
        states = tnb.generate_instance(n_traj, n_obj, device, dtype=dtype, G=G)
        targets = {"energia": tnb.compute_energy(states, G=G).cpu().numpy(),
                   "|momento angolare|": tnb.compute_angular_momentum(states).abs().cpu().numpy()}

    activations = extract_layer_activations(net, states, n_obj)
    print(f"    (overlap atteso per caso puro ~ {k}/{n_traj:.0f} ~= {k/n_traj:.4f}; valori vicini a 1 = struttura forte)")
    for li, act in enumerate(activations):
        norms = torch.norm(act, dim=-1)
        feats = norms.reshape(n_traj, -1).cpu().numpy()
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
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--n_obj", type=int, default=2)
    parser.add_argument("--n_blocks", type=int, default=4)
    parser.add_argument("--checkpoint2", type=str, default=None,
                         help="Secondo checkpoint opzionale (es. rete a 3 corpi) per il confronto [7]")
    parser.add_argument("--n_obj2", type=int, default=None,
                         help="n_obj del secondo checkpoint (richiesto se --checkpoint2 e' passato)")
    args = parser.parse_args()

    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    DTYPE = torch.float64

    def build_net(n_obj):
        if n_obj == 2:
            assert Acceleration2BodyNetv4 is not None, "Acceleration2BodyNet non trovata in networks.py"
            net_ = Acceleration2BodyNetv4(num_blocks=args.n_blocks, dtype=DTYPE, device=DEVICE).to(DEVICE)
            ctor_ = lambda: Acceleration2BodyNetv4(num_blocks=args.n_blocks, dtype=DTYPE, device=DEVICE).to(DEVICE)
        else:
            assert AccelerationNBodyNetv4 is not None, "AccelerationNBodyNetv4 non trovata in networks.py"
            net_ = AccelerationNBodyNetv4(n_obj=n_obj, num_blocks=args.n_blocks, dtype=DTYPE, device=DEVICE).to(DEVICE)
            ctor_ = lambda: AccelerationNBodyNetv4(n_obj=n_obj, num_blocks=args.n_blocks, dtype=DTYPE, device=DEVICE).to(DEVICE)
        return net_, ctor_

    net, net_random_ctor = build_net(args.n_obj)
    ckpt = torch.load(args.checkpoint, map_location=DEVICE)
    net.load_state_dict(ckpt.get("model", ckpt))
    net.eval()

    torch.manual_seed(0)
    np.random.seed(0)

    report_symmetric_antisymmetric(net, net_random_ctor, args.n_obj)
    probe_linear_regression(net, DEVICE, DTYPE, args.n_obj)
    probe_intrinsic_dimension(net, DEVICE, DTYPE, args.n_obj)
    probe_neighborhood_overlap(net, DEVICE, DTYPE, args.n_obj)

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
