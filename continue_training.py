import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from networks import AB2Net
import random
from tqdm import tqdm
from scipy.integrate import solve_ivp

from train import (physics_loss_2_body,
                   compute_angular_momentum,
                   conservation_loss,
                   compute_energy,
                   generate_instance)


def train(epochs_beginning: int,
          new_epochs: int,
          BodyNetwork: nn.Module,
          optimizer: torch.optim.Optimizer,
          scheduler,
          device: torch.device = torch.device('cpu'),
          batch_size: int = 256,
          dt: float = 0.05):

    loss_history = []
    # Ricordarsi questo per l'input, altrimenti niente gradiente all'indietro: input.requires_grad_(True)
    pbar = tqdm(range(epochs_beginning, epochs_beginning + new_epochs), desc="Training")

    for i in pbar:
        total_t = random.randint(1, min(30, 1 + i // 50))
        optimizer.zero_grad()

        c_weight = min(i / 100, 1)

        total = 0
        input = generate_instance(batch_size, device)
        input.requires_grad_(True)

        for _ in range(total_t):
            output = BodyNetwork(input)
            # ATTENZIONE A INPUT E OUTPUT COSA CONTENGONO
            p_loss = physics_loss_2_body(input, output, dt)
            c_loss = conservation_loss(input, output)

            total += p_loss + c_weight * c_loss
            input = output

        total /= total_t

        # Controllo di sanità: se un batch produce una loss non finita (nan/inf, es. per un
        # incontro ravvicinato tra i due corpi con r12 -> 0), saltiamo lo step invece di
        # propagare gradienti corrotti che possono far collassare la rete per il resto del
        # training.
        if not torch.isfinite(total):
            tqdm.write(f"epoch {i}: loss non finita ({total.item()}), skip step")
            continue

        total.backward()
        torch.nn.utils.clip_grad_norm_(BodyNetwork.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step(total.item())
        loss_history.append(total.item())

        pbar.set_postfix(loss=f"{total.item():.4e}", lr=f"{optimizer.param_groups[0]['lr']:.1e}")

    return loss_history


def continue_training(PATH, DEVICE, new_epochs=1000, batch_size=256, dt=0.05):

    dim = 10
    n_blocks = 4

    BodyNetwork = AB2Net(
        in_features=dim,
        out_features=8,
        num_blocks=n_blocks,
        dtype=torch.float64,
        device=DEVICE,
    ).to(DEVICE)

    ckpt = torch.load(PATH, map_location=DEVICE)

    # Pesi del modello: vanno caricati PRIMA di creare l'optimizer, cosi' i suoi
    # riferimenti ai parametri sono gia' sui tensori giusti.
    BodyNetwork.load_state_dict(ckpt["model"])

    # L'optimizer va ricreato (non "ricostruito" dalla classe) e poi il suo stato
    # (momenti di Adam, ecc.) va ripristinato con load_state_dict su un'ISTANZA.
    optimizer = torch.optim.Adam(BodyNetwork.parameters(), lr=1e-3)
    optimizer.load_state_dict(ckpt["optimizer"])

    # Stesso discorso per lo scheduler: va ricreato con gli stessi iperparametri
    # usati in train.py, poi il suo stato interno viene ripristinato.
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=50, min_lr=1e-6
    )
    scheduler.load_state_dict(ckpt["scheduler"])

    losses = ckpt["losses"]
    epoch = ckpt["epoch"]

    new_losses = train(
        epoch, new_epochs, BodyNetwork, optimizer, scheduler, DEVICE,
        batch_size=batch_size, dt=dt
    )

    return BodyNetwork, optimizer, scheduler, epoch + new_epochs, losses + new_losses


if __name__ == "__main__":

    print("=" * 90)

    PATH = "./PINN_savefile/save.pt"

    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)  # fissa anche l'RNG della GPU, non solo quello CPU

    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    net, optimizer, scheduler, epoch, losses = continue_training(
        PATH, DEVICE, new_epochs=1000, batch_size=256
    )

    torch.save({
        "model": net.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "epoch": epoch,
        "losses": losses
    }, PATH)

    plt.plot(losses)
    plt.yscale('log')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.savefig('loss_curve.png')
