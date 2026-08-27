import torch
import torch.nn as nn
import torch.nn.functional as F
from Equi_module.new_layer import AB_Block, NBodyPermutationBlock


class PermutationVectorABBlock(nn.Module):
    """
    Layer Lineare Vettoriale per 2 corpi.
    Equivariante sia per Rotazioni SO(2) che per Permutazioni S_2.
    Input:  [Batch, 2, C_in, 2]   (2 corpi, C_in canali per corpo, coord 2D)
    Output: [Batch, 2, C_out, 2]
    """
    def __init__(self, in_channels: int, out_channels: int, dtype=torch.float, device=None):
        super().__init__()
        # AB_Block gestisce la permutazione tra 2 corpi (in_feature=2*C_in, out_feature=2*C_out)
        self.ab_layer = AB_Block(
            in_feature=2 * in_channels,
            out_feature=2 * out_channels,
            bias=False, # Nessun bias per non rompere l'equivarianza spaziale
            dtype=dtype,
            device=device
        )

    def forward(self, v: torch.Tensor) -> torch.Tensor:
        # v shape: [Batch, 2, C_in, 2]
        B, N, C_in, spatial_dim = v.shape  # N = 2, spatial_dim = 2 (x, y)

        # Portiamo (x, y) alla prima dimensione per far lavorare AB_Block solo sulla dimensione dei canali dei corpi
        v_perm = v.permute(0, 3, 1, 2).reshape(B * spatial_dim, 2 * C_in)
        
        # Applicazione della matrice [A B; B A] sui canali dei 2 corpi
        out_flat = self.ab_layer(v_perm)  # [B * spatial_dim, 2 * C_out]

        # Riconfiguriamo nel tensor 4D originale [Batch, 2, C_out, 2]
        out = out_flat.view(B, spatial_dim, 2, -1).permute(0, 2, 3, 1)
        return out


class PermutationInvariantGate(nn.Module):
    """
    Gate di non-linearità equivariante per Rotazioni e Permutazioni.
    Invece di un MLP generico, scala i canali vettoriali usando sia le norme individuali
    sia le masse in modo simmetrico (usando AB_Block sugli scalari).
    """
    def __init__(self, num_channels, hidden=64, dtype=torch.float, device=None):
        super().__init__()
        self.num_channels = num_channels
        
        # 1. Block per elaborare [norme_corpo1, norme_corpo2] mantenendo la simmetria S_2
        self.ab1 = AB_Block(in_feature=2 * num_channels + 2, out_feature=2 * hidden, dtype=dtype, device=device)
        self.ab2 = AB_Block(in_feature=2 * hidden, out_feature=2 * num_channels, dtype=dtype, device=device)
        self.act = nn.SiLU()

    def forward(self, v, m):
        # v: [B, 2, C, 2], m: [B, 2, 1]
        B, N, C, _ = v.shape
        
        # Norme dei vettori per ciascun corpo (invarianti per rotazione SO(2))
        norms = torch.norm(v, dim=-1)  # [B, 2, C]
        
        # Concateniamo la massa di ciascun corpo alle sue norme
        body1_feats = torch.cat([m[:, 0], norms[:, 0]], dim=-1) # [B, 1 + C]
        body2_feats = torch.cat([m[:, 1], norms[:, 1]], dim=-1) # [B, 1 + C]
        
        # Input simmetrico per AB_Block: [Batch, 2 * (1 + C)]
        ab_in = torch.cat([body1_feats, body2_feats], dim=-1)
        
        # Passaggio simmetrico
        h = self.act(self.ab1(ab_in))
        gates_flat = torch.sigmoid(self.ab2(h)) # [B, 2 * C]
        
        # Reshape dei gates per i 2 corpi: [B, 2, C, 1]
        gates = gates_flat.view(B, 2, C).unsqueeze(-1)
        
        return v * gates







class Acceleration2BodyNet(nn.Module):
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
        gates = [PermutationInvariantGate(hidden_channels, dtype=dtype, device=device)]

        for _ in range(num_blocks - 2):
            layers.append(PermutationVectorABBlock(hidden_channels, hidden_channels, dtype=dtype, device=device))
            gates.append(PermutationInvariantGate(hidden_channels, dtype=dtype, device=device))

        layers.append(PermutationVectorABBlock(hidden_channels, out_channels, dtype=dtype, device=device))

        self.layers = nn.ModuleList(layers)
        self.gates = nn.ModuleList(gates)

    def predict_acceleration(self, state: torch.Tensor) -> torch.Tensor:
        """Helper interno per calcolare solo l'accelerazione a_corrected da uno stato."""
        B = state.shape[0]
        x_r = state.view(B, 2, 5)

        m = x_r[:, :, 0:1]  # [B, 2, 1]
        p = x_r[:, :, 1:3]  # [B, 2, 2]
        v = x_r[:, :, 3:5]  # [B, 2, 2]

        vecs = torch.stack([p, v], dim=2)  # [B, 2, 2, 2]

        out = vecs
        for layer, gate in zip(self.layers[:-1], self.gates):
            out = layer(out)
            out = gate(out, m)
        
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

        # 1. Accelerazione al tempo t: a_t
        a_t = self.predict_acceleration(x)

        # 2. Aggiornamento posizione: p(t+dt) = p(t) + v(t)*dt + 0.5*a(t)*dt^2
        p_next = p + v * dt + 0.5 * a_t * (dt ** 2)

        # Stato intermedio per calcolare l'accelerazione futura
        state_temp = torch.cat([m, p_next, v], dim=-1).view(B, -1)

        # 3. Accelerazione al tempo t+dt: a_{t+dt}
        a_next = self.predict_acceleration(state_temp)

        # 4. Aggiornamento velocità: v(t+dt) = v(t) + 0.5*(a_t + a_{t+dt})*dt
        v_next = v + 0.5 * (a_t + a_next) * dt

        # Ricostruzione output nello stesso formato di input [B, 10]
        out_r = torch.cat([m, p_next, v_next], dim=-1)
        return out_r.view(B, 10)




class NBodyAccelerationNet(nn.Module):
    """
    Rete per N corpi equivariante a permutazioni (S_N) e rotazioni SO(2).
    Prende in input uno stato [Batch, N * 5] dove ogni corpo ha [m, x, y, vx, vy].
    Avanza lo stato di dt usando Velocity Verlet integrato.
    """
    def __init__(self, n_obj: int, hidden_channels: int = 32, num_blocks: int = 4, 
                 dtype=torch.float64, device=None):
        super().__init__()
        self.n_obj = n_obj
        self.dtype = dtype
        self.device = device

        # Input per ogni corpo: 2 vettori (posizione p e velocità v)
        in_vec_channels = 2   
        out_vec_channels = 1  # Output: 1 vettore (accelerazione a)

        # Costruzione dei blocchi della rete
        layers = [NBodyPermutationBlock(in_vec_channels, hidden_channels, dtype=dtype, device=device)]
        
        # Strati intermedi con attivazioni
        for _ in range(num_blocks - 2):
            layers.append(NBodyPermutationBlock(hidden_channels, hidden_channels, dtype=dtype, device=device))
        
        # Strato di output
        layers.append(NBodyPermutationBlock(hidden_channels, out_vec_channels, dtype=dtype, device=device))
        
        self.layers = nn.ModuleList(layers)

    def predict_acceleration(self, state: torch.Tensor) -> torch.Tensor:
        """
        Calcola l'accelerazione a_i per ciascuno degli N corpi.
        Input: state [B, N * 5]
        Output: a [B, N, 2]
        """
        B = state.shape[0]
        N = self.n_obj
        
        x_r = state.view(B, N, 5)
        m = x_r[:, :, 0:1]       # [B, N, 1]
        p = x_r[:, :, 1:3]       # [B, N, 2]
        v = x_r[:, :, 3:5]       # [B, N, 2]

        # Stack dei vettori p e v -> [B, N, 2, 2] (Dimensione 2 e' x,y; Dimensione 3 e' [p, v])
        vecs = torch.stack([p, v], dim=-1)  # [B, N, 2, 2]

        out = vecs
        for i, layer in enumerate(self.layers[:-1]):
            out = layer(out)
            out = F.silu(out)  # Attivazione non lineare
            
        out = self.layers[-1](out)  # [B, N, 2, 1]
        a = out.squeeze(-1)         # [B, N, 2] (Accelerazione grezza)

        # Correzione HARD del 3° Principio di Newton: a_cm = 0
        M_tot = torch.sum(m, dim=1, keepdim=True)             # [B, 1, 1]
        a_cm = torch.sum(m * a, dim=1, keepdim=True) / M_tot   # [B, 1, 2]
        return a - a_cm

    def forward(self, x: torch.Tensor, dt: float = 0.01) -> torch.Tensor:
        """
        Input:  x [B, N * 5] al tempo t
        Output: x_next [B, N * 5] al tempo t + dt via Velocity Verlet
        """
        B = x.shape[0]
        N = self.n_obj
        x_r = x.view(B, N, 5)

        m = x_r[:, :, 0:1]
        p = x_r[:, :, 1:3]
        v = x_r[:, :, 3:5]

        # 1. Accelerazione al tempo t: a_t
        a_t = self.predict_acceleration(x)

        # 2. Aggiornamento posizione: p(t+dt) = p(t) + v(t)*dt + 0.5*a(t)*dt^2
        p_next = p + v * dt + 0.5 * a_t * (dt ** 2)

        # Stato temporaneo per calcolare la nuova accelerazione
        state_temp = torch.cat([m, p_next, v], dim=-1).view(B, -1)

        # 3. Accelerazione al tempo t+dt: a_{t+dt}
        a_next = self.predict_acceleration(state_temp)

        # 4. Aggiornamento velocità: v(t+dt) = v(t) + 0.5*(a_t + a_{t+dt})*dt
        v_next = v + 0.5 * (a_t + a_next) * dt

        # Ricostruzione output [B, N * 5]
        out_r = torch.cat([m, p_next, v_next], dim=-1)
        return out_r.view(B, -1)


class PermutationRotationGate(nn.Module):
    def __init__(self, num_channels, hidden=64, dtype=torch.float, device=None):
        super().__init__()
        self.ab1 = AB_Block(in_feature=2*num_channels + 2, out_feature=2*hidden, dtype=dtype, device=device)
        self.ab2 = AB_Block(in_feature=2*hidden, out_feature=2*num_channels, dtype=dtype, device=device)
        self.act = nn.SiLU()

        # theta ~ 0 all'inizio -> il gate parte come rotazione identica,
        # non come rotazione casuale per canale.
        with torch.no_grad():
            self.ab2.A.mul_(1e-2)
            self.ab2.B.mul_(1e-2)

    def forward(self, v, m):
        # v: [B, 2, C, 2]
        B, N, C, _ = v.shape
        norms = torch.norm(v, dim=-1)  # [B, 2, C]

        body1_feats = torch.cat([m[:, 0], norms[:, 0]], dim=-1)
        body2_feats = torch.cat([m[:, 1], norms[:, 1]], dim=-1)
        ab_in = torch.cat([body1_feats, body2_feats], dim=-1)

        h = self.act(self.ab1(ab_in))
        theta = torch.tanh(self.ab2(h)) * torch.pi   # [B, 2*C], bound in [-pi, pi]
        theta = theta.view(B, 2, C)

        cos_t, sin_t = torch.cos(theta), torch.sin(theta)
        vx, vy = v[..., 0], v[..., 1]
        vx_out = cos_t * vx - sin_t * vy
        vy_out = sin_t * vx + cos_t * vy
        return torch.stack([vx_out, vy_out], dim=-1)


class Acceleration2BodyNetv2(nn.Module):
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
        gates = [PermutationRotationGate(hidden_channels, dtype=dtype, device=device)]

        for _ in range(num_blocks - 2):
            layers.append(PermutationVectorABBlock(hidden_channels, hidden_channels, dtype=dtype, device=device))
            gates.append(PermutationRotationGate(hidden_channels, dtype=dtype, device=device))

        layers.append(PermutationVectorABBlock(hidden_channels, out_channels, dtype=dtype, device=device))

        self.layers = nn.ModuleList(layers)
        self.gates = nn.ModuleList(gates)

    def predict_acceleration(self, state: torch.Tensor) -> torch.Tensor:
        """Helper interno per calcolare solo l'accelerazione a_corrected da uno stato."""
        B = state.shape[0]
        x_r = state.view(B, 2, 5)

        m = x_r[:, :, 0:1]  # [B, 2, 1]
        p = x_r[:, :, 1:3]  # [B, 2, 2]
        v = x_r[:, :, 3:5]  # [B, 2, 2]

        vecs = torch.stack([p, v], dim=2)  # [B, 2, 2, 2]

        out = vecs
        for layer, gate in zip(self.layers[:-1], self.gates):
            out = layer(out)
            out = gate(out, m)
        
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

        # 1. Accelerazione al tempo t: a_t
        a_t = self.predict_acceleration(x)

        # 2. Aggiornamento posizione: p(t+dt) = p(t) + v(t)*dt + 0.5*a(t)*dt^2
        p_next = p + v * dt + 0.5 * a_t * (dt ** 2)

        # Stato intermedio per calcolare l'accelerazione futura
        state_temp = torch.cat([m, p_next, v], dim=-1).view(B, -1)

        # 3. Accelerazione al tempo t+dt: a_{t+dt}
        a_next = self.predict_acceleration(state_temp)

        # 4. Aggiornamento velocità: v(t+dt) = v(t) + 0.5*(a_t + a_{t+dt})*dt
        v_next = v + 0.5 * (a_t + a_next) * dt

        # Ricostruzione output nello stesso formato di input [B, 10]
        out_r = torch.cat([m, p_next, v_next], dim=-1)
        return out_r.view(B, 10)



class Acceleration2BodyNetv3(nn.Module):
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
        gates = [PermutationRotationGate(hidden_channels, dtype=dtype, device=device)]
        gates2 = [PermutationInvariantGate(hidden_channels, dtype=dtype, device=device)]

        for _ in range(num_blocks - 2):
            layers.append(PermutationVectorABBlock(hidden_channels, hidden_channels, dtype=dtype, device=device))
            gates.append(PermutationRotationGate(hidden_channels, dtype=dtype, device=device))
            gates2.append(PermutationInvariantGate(hidden_channels, dtype=dtype, device=device))

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

        out = vecs
        for layer, gate, gates2 in zip(self.layers[:-1], self.gates, self.gates2):
            out = layer(out)
            out = gate(out, m)
            out = gates2(out, m)
        
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

        # 1. Accelerazione al tempo t: a_t
        a_t = self.predict_acceleration(x)

        # 2. Aggiornamento posizione: p(t+dt) = p(t) + v(t)*dt + 0.5*a(t)*dt^2
        p_next = p + v * dt + 0.5 * a_t * (dt ** 2)

        # Stato intermedio per calcolare l'accelerazione futura
        state_temp = torch.cat([m, p_next, v], dim=-1).view(B, -1)

        # 3. Accelerazione al tempo t+dt: a_{t+dt}
        a_next = self.predict_acceleration(state_temp)

        # 4. Aggiornamento velocità: v(t+dt) = v(t) + 0.5*(a_t + a_{t+dt})*dt
        v_next = v + 0.5 * (a_t + a_next) * dt

        # Ricostruzione output nello stesso formato di input [B, 10]
        out_r = torch.cat([m, p_next, v_next], dim=-1)
        return out_r.view(B, 10)



class Acceleration2BodyNetv4(nn.Module):
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
        gates = [PermutationRotationGate(hidden_channels, dtype=dtype, device=device)]
        gates2 = [PermutationInvariantGate(hidden_channels, dtype=dtype, device=device)]

        for _ in range(num_blocks - 2):
            layers.append(PermutationVectorABBlock(hidden_channels, hidden_channels, dtype=dtype, device=device))
            gates.append(PermutationRotationGate(hidden_channels, dtype=dtype, device=device))
            gates2.append(PermutationInvariantGate(hidden_channels, dtype=dtype, device=device))

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

        out = vecs
        for layer, gate, gates2 in zip(self.layers[:-1], self.gates, self.gates2):
            out = layer(out)
            out = gates2(out, m)
            out = gate(out, m)
        
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

        # 1. Accelerazione al tempo t: a_t
        a_t = self.predict_acceleration(x)

        # 2. Aggiornamento posizione: p(t+dt) = p(t) + v(t)*dt + 0.5*a(t)*dt^2
        p_next = p + v * dt + 0.5 * a_t * (dt ** 2)

        # Stato intermedio per calcolare l'accelerazione futura
        state_temp = torch.cat([m, p_next, v], dim=-1).view(B, -1)

        # 3. Accelerazione al tempo t+dt: a_{t+dt}
        a_next = self.predict_acceleration(state_temp)

        # 4. Aggiornamento velocità: v(t+dt) = v(t) + 0.5*(a_t + a_{t+dt})*dt
        v_next = v + 0.5 * (a_t + a_next) * dt

        # Ricostruzione output nello stesso formato di input [B, 10]
        out_r = torch.cat([m, p_next, v_next], dim=-1)
        return out_r.view(B, 10)

class NBodyPermutationInvariantGate(nn.Module):
    """
    Gate di Scaling Equivariante per N corpi.
    Scala i canali vettoriali usando masse e norme in modo equivariante S_N.
    """
    def __init__(self, num_channels: int, hidden: int = 64, dtype=torch.float, device=None):
        super().__init__()
        self.num_channels = num_channels
        
        # Features per ciascun corpo: 1 (massa) + num_channels (norme dei vettori)
        body_feat_dim = 1 + num_channels

        self.block1 = NBodyPermutationBlock(body_feat_dim, hidden, dtype=dtype, device=device)
        self.block2 = NBodyPermutationBlock(hidden, num_channels, dtype=dtype, device=device)
        self.act = nn.SiLU()

    def forward(self, v: torch.Tensor, m: torch.Tensor) -> torch.Tensor:
        # v: [B, N, 2, C], m: [B, N, 1]
        # Norme dei vettori per ogni corpo lungo la dimensione spaziale (dim 2)
        norms = torch.norm(v, dim=2)  # [B, N, C]
        
        # Concatenazione massa + norme per ogni corpo
        body_feats = torch.cat([m, norms], dim=-1)  # [B, N, 1 + C]
        
        h = self.act(self.block1(body_feats))       # [B, N, hidden]
        gates = torch.sigmoid(self.block2(h))       # [B, N, C]
        
        # Reshape per lo scaling vettoriale [B, N, 1, C]
        return v * gates.unsqueeze(2)


class NBodyPermutationRotationGate(nn.Module):
    """
    Gate di Rotazione Equivariante per N corpi.
    Ruota ciascun canale vettoriale di un angolo theta appreso in modo equivariante S_N.
    """
    def __init__(self, num_channels: int, hidden: int = 64, dtype=torch.float, device=None):
        super().__init__()
        self.num_channels = num_channels
        
        body_feat_dim = 1 + num_channels

        self.block1 = NBodyPermutationBlock(body_feat_dim, hidden, dtype=dtype, device=device)
        self.block2 = NBodyPermutationBlock(hidden, num_channels, dtype=dtype, device=device)
        self.act = nn.SiLU()

        # Inizializzazione piccola per far partire il gate come trasformazione identica (theta ~ 0)
        with torch.no_grad():
            for p in self.block2.parameters():
                p.mul_(1e-2)

    def forward(self, v: torch.Tensor, m: torch.Tensor) -> torch.Tensor:
        # v: [B, N, 2, C], m: [B, N, 1]
        norms = torch.norm(v, dim=2)  # [B, N, C]

        body_feats = torch.cat([m, norms], dim=-1)  # [B, N, 1 + C]

        h = self.act(self.block1(body_feats))
        theta = torch.tanh(self.block2(h)) * torch.pi  # [B, N, C] nell'intervallo [-pi, pi]
        theta = theta.unsqueeze(2)  # [B, N, 1, C]

        cos_t, sin_t = torch.cos(theta), torch.sin(theta)
        vx, vy = v[:, :, 0:1, :], v[:, :, 1:2, :]
        
        vx_out = cos_t * vx - sin_t * vy
        vy_out = sin_t * vx + cos_t * vy
        
        return torch.cat([vx_out, vy_out], dim=2)  # [B, N, 2, C]


class AccelerationNBodyNetv4(nn.Module):
    """
    Rete per N corpi equivariante a Rotazioni SO(2) e Permutazioni S_N.
    Implementa la logica v4 (alternanza Layer -> InvariantGate -> RotationGate)
    ed evolve lo stato tramite Velocity Verlet.
    """
    def __init__(self, n_obj: int, hidden_channels: int = 32, num_blocks: int = 4, 
                 dtype=torch.float64, device=torch.device("cpu")):
        super().__init__()
        self.n_obj = n_obj
        self.dtype = dtype
        self.device = device

        in_channels = 2   # Canali vettoriali in ingresso: [p_i, v_i]
        out_channels = 1  # Canale vettoriale in uscita: [a_i]

        layers = [NBodyPermutationBlock(in_channels, hidden_channels, dtype=dtype, device=device)]
        gates_inv = [NBodyPermutationInvariantGate(hidden_channels, dtype=dtype, device=device)]
        gates_rot = [NBodyPermutationRotationGate(hidden_channels, dtype=dtype, device=device)]

        for _ in range(num_blocks - 2):
            layers.append(NBodyPermutationBlock(hidden_channels, hidden_channels, dtype=dtype, device=device))
            gates_inv.append(NBodyPermutationInvariantGate(hidden_channels, dtype=dtype, device=device))
            gates_rot.append(NBodyPermutationRotationGate(hidden_channels, dtype=dtype, device=device))

        layers.append(NBodyPermutationBlock(hidden_channels, out_channels, dtype=dtype, device=device))

        self.layers = nn.ModuleList(layers)
        self.gates_inv = nn.ModuleList(gates_inv)
        self.gates_rot = nn.ModuleList(gates_rot)

    def predict_acceleration(self, state: torch.Tensor) -> torch.Tensor:
        """Calcola l'accelerazione a_i con correzione hard del centro di massa (a_cm = 0)."""
        B = state.shape[0]
        N = self.n_obj
        x_r = state.view(B, N, 5)

        m = x_r[:, :, 0:1]  # [B, N, 1]
        p = x_r[:, :, 1:3]  # [B, N, 2]
        v = x_r[:, :, 3:5]  # [B, N, 2]

        # Stack dei vettori di posizione e velocità -> [B, N, 2, 2]
        # dim 2: coordinate spaziali (x, y); dim 3: canali vettoriali (p, v)
        vecs = torch.stack([p, v], dim=-1)

        out = vecs
        # Sequenza v4: Layer -> InvariantGate -> RotationGate
        for layer, g_inv, g_rot in zip(self.layers[:-1], self.gates_inv, self.gates_rot):
            out = layer(out)
            out = g_inv(out, m)
            out = g_rot(out, m)
        
        out = self.layers[-1](out)
        a = out.squeeze(-1)  # [B, N, 2]

        # Correzione Hard: a_cm = 0 (Terzo Principio di Newton)
        M_tot = torch.sum(m, dim=1, keepdim=True)
        a_cm = torch.sum(m * a, dim=1, keepdim=True) / M_tot
        return a - a_cm

    def forward(self, x: torch.Tensor, dt: float = 0.01) -> torch.Tensor:
        """Avanza lo stato di dt usando Velocity Verlet."""
        B = x.shape[0]
        N = self.n_obj
        x_r = x.view(B, N, 5)

        m = x_r[:, :, 0:1]
        p = x_r[:, :, 1:3]
        v = x_r[:, :, 3:5]

        # 1. Accelerazione a(t)
        a_t = self.predict_acceleration(x)

        # 2. Aggiornamento posizione p(t+dt)
        p_next = p + v * dt + 0.5 * a_t * (dt ** 2)

        # Stato temporaneo
        state_temp = torch.cat([m, p_next, v], dim=-1).view(B, -1)

        # 3. Accelerazione a(t+dt)
        a_next = self.predict_acceleration(state_temp)

        # 4. Aggiornamento velocità v(t+dt)
        v_next = v + 0.5 * (a_t + a_next) * dt

        out_r = torch.cat([m, p_next, v_next], dim=-1)
        return out_r.view(B, -1)