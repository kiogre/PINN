import torch
import numpy as np
from train import generate_instance

"""
Misura la distribuzione dei periodi orbitali (in numero di STEP, non tempo
fisico) sulla distribuzione attuale di generate_instance, per calibrare
max_horizon nel curriculum di train.py invece di sceglierlo a caso.

Uso l'equazione vis-viva + terza legge di Keplero: entrambe richiedono solo
massa totale, distanza e velocita' relativa all'istante iniziale -- gia'
tutti disponibili nello stato generato, nessuna integrazione necessaria.
"""

def orbital_period_steps(state, dt, G=1.0):
    """state: [B, 10] = [m1,x1,y1,vx1,vy1,m2,x2,y2,vx2,vy2] (frame canonico:
    corpo 1 in (0,0), quindi la posizione relativa e' esattamente p2).
    Ritorna (steps, bound): steps=NaN dove l'orbita non e' legata.
    """
    m1, m2 = state[:, 0], state[:, 5]
    M_tot = m1 + m2

    x2, y2 = state[:, 6], state[:, 7]
    dist = torch.sqrt(x2**2 + y2**2)

    vx1, vy1 = state[:, 3], state[:, 4]
    vx2, vy2 = state[:, 8], state[:, 9]
    v_rel2 = (vx2 - vx1)**2 + (vy2 - vy1)**2  # |v_rel|^2

    inv_a = 2.0 / dist - v_rel2 / (G * M_tot)
    bound = inv_a > 0

    a = torch.where(bound, 1.0 / inv_a.clamp(min=1e-8), torch.full_like(inv_a, float('nan')))
    T = 2 * torch.pi * torch.sqrt(a**3 / (G * M_tot))
    steps = T / dt

    return steps, bound


if __name__ == "__main__":
    torch.manual_seed(0)
    DEVICE = torch.device('cpu')
    dt = 0.01
    n_samples = 20000

    state = generate_instance(batch_size=n_samples, device=DEVICE, dtype=torch.float64)
    steps, bound = orbital_period_steps(state, dt)

    frac_bound = bound.float().mean().item()
    print(f"Frazione di orbite legate: {frac_bound*100:.1f}%")
    print(f"Frazione di traiettorie aperte (fionda/iperboliche): {(1-frac_bound)*100:.1f}%")

    steps_bound = steps[bound].numpy()
    steps_bound = steps_bound[np.isfinite(steps_bound)]

    print(f"\nPeriodo orbitale (in numero di step, solo orbite legate):")
    print(f"  minimo:            {steps_bound.min():.1f}")
    print(f"  5° percentile:     {np.percentile(steps_bound, 5):.1f}")
    print(f"  mediana:           {np.percentile(steps_bound, 50):.1f}")
    print(f"  90° percentile:    {np.percentile(steps_bound, 90):.1f}")
    print(f"  95° percentile:    {np.percentile(steps_bound, 95):.1f}")
    print(f"  massimo:           {steps_bound.max():.1f}")

    print(f"\nSuggerimento per max_horizon (curriculum):")
    print(f"  copre il 90% delle orbite legate con 1 periodo intero: {np.percentile(steps_bound, 90):.0f} step")
    print(f"  copre il 90% delle orbite legate con 1.5 periodi:      {1.5*np.percentile(steps_bound, 90):.0f} step")
