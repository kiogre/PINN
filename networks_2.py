import torch
import torch.nn as nn
import torch.nn.functional as F
from Equi_module.new_layer import AB_Block, NBodyPermutationBlock
from networks import PermutationVectorABBlock, NBodyPermutationInvariantGate, NBodyPermutationRotationGate
from networks import NBodyPermutationBlock


class NBodyPermutationInvariantGate_r(nn.Module):
    """
    Gate di Scaling Equivariante per N corpi.
    Scala i canali vettoriali usando masse, norme ed embedding delle distanze in modo equivariante S_N.
    """
    def __init__(self, num_channels: int, r_dim: int = 1, hidden: int = 64, dtype=torch.float, device=None):
        super().__init__()
        self.num_channels = num_channels
        
        # Features per ciascun corpo: 1 (massa) + num_channels (norme) + r_dim (embedding distanze)
        body_feat_dim = 1 + num_channels + r_dim

        self.block1 = NBodyPermutationBlock(body_feat_dim, hidden, dtype=dtype, device=device)
        self.block2 = NBodyPermutationBlock(hidden, num_channels, dtype=dtype, device=device)
        self.act = nn.SiLU()

    def forward(self, v: torch.Tensor, m: torch.Tensor, r_ij) -> torch.Tensor:
        # v: [B, N, 2, C], m: [B, N, 1], r_ij: [B, N, r_dim]
        norms = torch.norm(v, dim=2)  # [B, N, C]
        
        # Concatenazione massa + norme + embedding distanze per ogni corpo
        body_feats = torch.cat([m, norms, r_ij], dim=-1)  # [B, N, 1 + C + r_dim]
        
        h = self.act(self.block1(body_feats))       # [B, N, hidden]
        gates = torch.sigmoid(self.block2(h))       # [B, N, C]
        
        return v * gates.unsqueeze(2)


class NBodyPermutationRotationGate_r(nn.Module):
    """
    Gate di Rotazione Equivariante per N corpi.
    Ruota ciascun canale vettoriale di un angolo theta appreso in modo equivariante S_N.
    """
    def __init__(self, num_channels: int, r_dim: int = 1, hidden: int = 64, dtype=torch.float, device=None):
        super().__init__()
        self.num_channels = num_channels
        
        # Features per ciascun corpo: 1 (massa) + num_channels (norme) + r_dim (embedding distanze)
        body_feat_dim = 1 + num_channels + r_dim

        self.block1 = NBodyPermutationBlock(body_feat_dim, hidden, dtype=dtype, device=device)
        self.block2 = NBodyPermutationBlock(hidden, num_channels, dtype=dtype, device=device)
        self.act = nn.SiLU()

        with torch.no_grad():
            for p in self.block2.parameters():
                p.mul_(1e-2)

    def forward(self, v: torch.Tensor, m: torch.Tensor, r_ij) -> torch.Tensor:
        norms = torch.norm(v, dim=2)  # [B, N, C]

        body_feats = torch.cat([m, norms, r_ij], dim=-1)  # [B, N, 1 + C + r_dim]

        h = self.act(self.block1(body_feats))
        theta = torch.tanh(self.block2(h)) * torch.pi  # [B, N, C]
        theta = theta.unsqueeze(2)  # [B, N, 1, C]

        cos_t, sin_t = torch.cos(theta), torch.sin(theta)
        vx, vy = v[:, :, 0:1, :], v[:, :, 1:2, :]
        
        vx_out = cos_t * vx - sin_t * vy
        vy_out = sin_t * vx + cos_t * vy
        
        return torch.cat([vx_out, vy_out], dim=2)  # [B, N, 2, C]


class PermutationInvariantGate_r(nn.Module):
    def __init__(self, num_channels, hidden=64, dtype=torch.float, device=None):
        super().__init__()
        self.ab1 = AB_Block(in_feature=2 * num_channels + 2 + 2, out_feature=2 * hidden, dtype=dtype, device=device)
        self.ab2 = AB_Block(in_feature=2 * hidden, out_feature=2 * num_channels, dtype=dtype, device=device)
        self.act = nn.SiLU()

    def forward(self, v, m, r_ij):          # <-- nuovo argomento
        B, N, C, _ = v.shape
        norms = torch.norm(v, dim=-1)
        body1_feats = torch.cat([m[:, 0], norms[:, 0], r_ij], dim=-1)
        body2_feats = torch.cat([m[:, 1], norms[:, 1], r_ij], dim=-1)
        ab_in = torch.cat([body1_feats, body2_feats], dim=-1)
        h = self.act(self.ab1(ab_in))
        gates = torch.sigmoid(self.ab2(h)).view(B, 2, C).unsqueeze(-1)
        return v * gates


class PermutationRotationGate_r(nn.Module):
    def __init__(self, num_channels, hidden=64, dtype=torch.float, device=None):
        super().__init__()
        self.ab1 = AB_Block(in_feature=2 * num_channels + 2 + 2, out_feature=2 * hidden, dtype=dtype, device=device)
        self.ab2 = AB_Block(in_feature=2 * hidden, out_feature=2 * num_channels, dtype=dtype, device=device)
        self.act = nn.SiLU()

        # theta ~ 0 all'inizio -> il gate parte come rotazione identica,
        # non come rotazione casuale per canale.
        with torch.no_grad():
            self.ab2.A.mul_(1e-2)
            self.ab2.B.mul_(1e-2)

    def forward(self, v, m, r_ij):
        # v: [B, 2, C, 2]
        B, N, C, _ = v.shape
        norms = torch.norm(v, dim=-1)  # [B, 2, C]

        body1_feats = torch.cat([m[:, 0], norms[:, 0], r_ij], dim=-1)
        body2_feats = torch.cat([m[:, 1], norms[:, 1], r_ij], dim=-1)
        ab_in = torch.cat([body1_feats, body2_feats], dim=-1)

        h = self.act(self.ab1(ab_in))
        theta = torch.tanh(self.ab2(h)) * torch.pi   # [B, 2*C], bound in [-pi, pi]
        theta = theta.view(B, 2, C)

        cos_t, sin_t = torch.cos(theta), torch.sin(theta)
        vx, vy = v[..., 0], v[..., 1]
        vx_out = cos_t * vx - sin_t * vy
        vy_out = sin_t * vx + cos_t * vy
        return torch.stack([vx_out, vy_out], dim=-1)



class Acceleration2BodyNetv5(nn.Module):
    """
    Rete per 2 corpi equivariante SO(2) e S_2.
    Calcola internamente l'accelerazione a(t) e usa Velocity Verlet per avanzare di dt,
    restituendo lo stato successivo nello stesso formato di input [B, 10].
    """
    def __init__(self, hidden_channels=32, num_blocks=4, dtype=torch.float64, device=torch.device("cpu")):
        super().__init__()
        self.dtype = dtype
        self.device = device

        in_channels = 2   # Input vettoriali: [p_i, v_i]
        out_channels = 1  # Output vettoriale: [a_i] (solo accelerazione)

        layers = [PermutationVectorABBlock(in_channels, hidden_channels, dtype=dtype, device=device)]
        gates = [PermutationRotationGate_r(hidden_channels, dtype=dtype, device=device)]
        gates2 = [PermutationInvariantGate_r(hidden_channels, dtype=dtype, device=device)]

        for _ in range(num_blocks - 2):
            layers.append(PermutationVectorABBlock(hidden_channels, hidden_channels, dtype=dtype, device=device))
            gates.append(PermutationRotationGate_r(hidden_channels, dtype=dtype, device=device))
            gates2.append(PermutationInvariantGate_r(hidden_channels, dtype=dtype, device=device))

        layers.append(PermutationVectorABBlock(hidden_channels, out_channels, dtype=dtype, device=device))

        self.layers = nn.ModuleList(layers)
        self.gates = nn.ModuleList(gates)
        self.gates2 = nn.ModuleList(gates2)

    def predict_acceleration(self, state: torch.Tensor) -> torch.Tensor:
        """Helper interno per calcolare solo l'accelerazione a_corrected da uno stato."""
        B = state.shape[0]
        x_r = state.view(B, 2, 5)

        m = x_r[:, :, 0:1]  # [B, 2, 1]
        p = x_r[:, :, 1:3]  # [B, 2, 2]
        v = x_r[:, :, 3:5]  # [B, 2, 2]

        vecs = torch.stack([p, v], dim=2)  # [B, 2, 2, 2]

        eps = 1e-3

        r_ij = 1.0 / (torch.sum((p[:, 0] - p[:, 1])**2, dim=-1, keepdim=True) + eps) 

        out = vecs
        for layer, gate, gates2 in zip(self.layers[:-1], self.gates, self.gates2):
            out = layer(out)
            out = gates2(out, m, r_ij)
            out = gate(out, m, r_ij)
        
        out = self.layers[-1](out)
        a = out.squeeze(2)  # [B, 2, 2]

        # Correzione Hard: a_cm = 0 (3° Principio di Newton)
        M_tot = torch.sum(m, dim=1, keepdim=True)
        a_cm = torch.sum(m * a, dim=1, keepdim=True) / M_tot
        return a - a_cm

    def forward(self, x: torch.Tensor, dt: float = 0.01) -> torch.Tensor:
        """
        Input:  x  [B, 10] al tempo t
        Output: x_next [B, 10] al tempo t + dt (usando Velocity Verlet)
        """
        B = x.shape[0]
        x_r = x.view(B, 2, 5)

        m = x_r[:, :, 0:1]
        p = x_r[:, :, 1:3]
        v = x_r[:, :, 3:5]

        # Accelerazione al tempo t: a_t
        a_t = self.predict_acceleration(x)

        # Aggiornamento posizione: p(t+dt) = p(t) + v(t)*dt + 0.5*a(t)*dt^2
        p_next = p + v * dt + 0.5 * a_t * (dt ** 2)

        # Stato intermedio per calcolare l'accelerazione futura
        state_temp = torch.cat([m, p_next, v], dim=-1).view(B, -1)

        # Accelerazione al tempo t+dt: a_{t+dt}
        a_next = self.predict_acceleration(state_temp)

        # Aggiornamento velocità: v(t+dt) = v(t) + 0.5*(a_t + a_{t+dt})*dt
        v_next = v + 0.5 * (a_t + a_next) * dt

        # Ricostruzione output nello stesso formato di input [B, 10]
        out_r = torch.cat([m, p_next, v_next], dim=-1)
        return out_r.view(B, 10)




class AccelerationNBodyNetv5(nn.Module):
    """
    Rete per N corpi equivariante a Rotazioni SO(2) e Permutazioni S_N.
    Evoluzione di NBodyv4: inietta nei gate le distanze a coppie aggregate (r_dim=2),
    abilitando il message passing dipendente dalla geometria locale.
    """
    def __init__(self, n_obj: int, hidden_channels: int = 32, num_blocks: int = 4, 
                 r_dim: int = 2, dtype=torch.float64, device=torch.device("cpu")):
        super().__init__()
        self.n_obj = n_obj
        self.dtype = dtype
        self.device = device
        self.r_dim = r_dim

        in_channels = 2   # [p_i, v_i]
        out_channels = 1  # [a_i]

        layers = [NBodyPermutationBlock(in_channels, hidden_channels, dtype=dtype, device=device)]
        gates_inv = [NBodyPermutationInvariantGate_r(hidden_channels, r_dim=r_dim, dtype=dtype, device=device)]
        gates_rot = [NBodyPermutationRotationGate_r(hidden_channels, r_dim=r_dim, dtype=dtype, device=device)]

        for _ in range(num_blocks - 2):
            layers.append(NBodyPermutationBlock(hidden_channels, hidden_channels, dtype=dtype, device=device))
            gates_inv.append(NBodyPermutationInvariantGate_r(hidden_channels, r_dim=r_dim, dtype=dtype, device=device))
            gates_rot.append(NBodyPermutationRotationGate_r(hidden_channels, r_dim=r_dim, dtype=dtype, device=device))

        layers.append(NBodyPermutationBlock(hidden_channels, out_channels, dtype=dtype, device=device))

        self.layers = nn.ModuleList(layers)
        self.gates_inv = nn.ModuleList(gates_inv)
        self.gates_rot = nn.ModuleList(gates_rot)

    def _compute_r_features(self, p: torch.Tensor, m: torch.Tensor, eps: float = 1e-3) -> torch.Tensor:
        """
        Calcola feature scalari invarianti SO(2) e permutazione-invarianti sugli altri corpi.
        Restituisce un tensore [B, N, 2] con l'intensità locale gravitazionale (1/r^2 e 1/r^3 pesati su m_j).
        """
        B, N, _ = p.shape
        # diff[b, i, j] = p_j - p_i
        diff = p.unsqueeze(1) - p.unsqueeze(2)                          # [B, N, N, 2]
        dist_sq = torch.sum(diff ** 2, dim=-1, keepdim=True) + eps      # [B, N, N, 1]
        dist = torch.sqrt(dist_sq)                                      # [B, N, N, 1]

        # Maschera per escludere il self-interaction (j == i)
        mask = 1.0 - torch.eye(N, device=p.device, dtype=p.dtype).view(1, N, N, 1)

        m_j = m.view(B, 1, N, 1)  # massa del corpo sorgente j
        # Feature 1: potenziale locale ~ sum m_j / r_ij
        feat_inv_r = torch.sum((m_j / dist) * mask, dim=2)             # [B, N, 1]
        # Feature 2: intensità di campo ~ sum m_j / r_ij^2
        feat_inv_r2 = torch.sum((m_j / dist_sq) * mask, dim=2)         # [B, N, 1]

        return torch.cat([feat_inv_r, feat_inv_r2], dim=-1)             # [B, N, 2]

    def predict_acceleration(self, state: torch.Tensor) -> torch.Tensor:
        B = state.shape[0]
        N = self.n_obj
        x_r = state.view(B, N, 5)

        m = x_r[:, :, 0:1]  # [B, N, 1]
        p = x_r[:, :, 1:3]  # [B, N, 2]
        v = x_r[:, :, 3:5]  # [B, N, 2]

        vecs = torch.stack([p, v], dim=-1)  # [B, N, 2, 2]

        # Calcolo dell'aggregato delle distanze a coppie
        r_ij_feats = self._compute_r_features(p, m)  # [B, N, 2]

        out = vecs
        for layer, g_inv, g_rot in zip(self.layers[:-1], self.gates_inv, self.gates_rot):
            out = layer(out)
            out = g_inv(out, m, r_ij_feats)
            out = g_rot(out, m, r_ij_feats)

        out = self.layers[-1](out)
        a = out.squeeze(-1)  # [B, N, 2]

        # Correzione Hard: 3° Principio di Newton (a_cm = 0)
        M_tot = torch.sum(m, dim=1, keepdim=True)
        a_cm = torch.sum(m * a, dim=1, keepdim=True) / M_tot
        return a - a_cm

    def forward(self, x: torch.Tensor, dt: float = 0.01) -> torch.Tensor:
        B = x.shape[0]
        N = self.n_obj
        x_r = x.view(B, N, 5)

        m = x_r[:, :, 0:1]
        p = x_r[:, :, 1:3]
        v = x_r[:, :, 3:5]

        a_t = self.predict_acceleration(x)
        p_next = p + v * dt + 0.5 * a_t * (dt ** 2)

        state_temp = torch.cat([m, p_next, v], dim=-1).view(B, -1)
        a_next = self.predict_acceleration(state_temp)
        v_next = v + 0.5 * (a_t + a_next) * dt

        out_r = torch.cat([m, p_next, v_next], dim=-1)
        return out_r.view(B, -1)