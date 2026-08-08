import torch
import torch.nn as nn
import torch.nn.functional as F
from Equi_module.new_layer import AB_Block, NBodyPermutationBlock

class AB2Net(nn.Module):

    def __init__(
        self,
        in_features: int = 10,
        out_features: int = 8,
        num_blocks: int = 3,
        hidden_dim: int = 256,
        dtype: torch.dtype = torch.float,
        device: torch.device = torch.device("cpu")
    ) -> None:

        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.num_blocks = num_blocks

        ls = [AB_Block(in_feature=self.in_features, out_feature=hidden_dim, dtype=dtype, device=device)]
        ls += [nn.SiLU()]

        for _ in range(num_blocks - 2):
            ls += [AB_Block(hidden_dim, hidden_dim, dtype=dtype, device=device)]
            ls += [nn.SiLU()]

        ls += [AB_Block(hidden_dim, out_features, dtype=dtype, device=device)]

        self.network_modules = nn.ModuleList(ls)

        # L'ultimo layer produce il residuo (kinematic_next = kinematic_in + out):
        # lo inizializziamo con pesi piccoli così all'inizio del training out ~ 0,
        # cioe' la rete parte da "quasi non cambiare nulla" invece che da un salto
        # casuale di scala O(1) rispetto a un target che e' O(dt) -- molto più
        # facile da correggere via discesa del gradiente.
        with torch.no_grad():
            self.network_modules[-1].A.mul_(1e-2)
            self.network_modules[-1].B.mul_(1e-2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        m1 = x[:, 0:1]
        m2 = x[:, 5:6]

        # Momento lineare dello stato in INGRESSO -- riferimento per il vincolo hard
        vx1_in, vy1_in = x[:, 3:4], x[:, 4:5]
        vx2_in, vy2_in = x[:, 8:9], x[:, 9:10]
        px_in = m1 * vx1_in + m2 * vx2_in
        py_in = m1 * vy1_in + m2 * vy2_in

        # Parte cinematica (posizioni + velocità, 8 dim) su cui la rete impara un residuo
        kinematic_in = torch.cat([x[:, 1:5], x[:, 6:10]], dim=1)

        out = x
        for module in self.network_modules:
            out = module(out)

        kinematic_next = kinematic_in + out  # residuo appreso, non un delta*dt forzato

        x1n, y1n = kinematic_next[:, 0:1], kinematic_next[:, 1:2]
        vx1n, vy1n = kinematic_next[:, 2:3], kinematic_next[:, 3:4]
        x2n, y2n = kinematic_next[:, 4:5], kinematic_next[:, 5:6]
        vx2n, vy2n = kinematic_next[:, 6:7], kinematic_next[:, 7:8]

        # --- Correzione hard: conservazione ESATTA del momento lineare ---
        # Il momento totale non ha nessuna forza esterna che lo cambi (III legge di
        # Newton): m1*v1 + m2*v2 deve restare identico tra input e output di OGNI
        # step. Non lo affidiamo alla loss (soft constraint, puo' essere violato):
        # lo imponiamo esattamente, sottraendo la deriva in eccesso in proporzione
        # inversa alla massa, cosi' come farebbe una vera correzione d'impulso.
        M_tot = m1 + m2
        px_out = m1 * vx1n + m2 * vx2n
        py_out = m1 * vy1n + m2 * vy2n

        corr_x = (px_out - px_in) / M_tot
        corr_y = (py_out - py_in) / M_tot

        vx1n = vx1n - corr_x
        vx2n = vx2n - corr_x
        vy1n = vy1n - corr_y
        vy2n = vy2n - corr_y

        kinematic_next = torch.cat([x1n, y1n, vx1n, vy1n, x2n, y2n, vx2n, vy2n], dim=1)

        return torch.cat([m1, kinematic_next[:, :4], m2, kinematic_next[:, 4:]], dim=1)


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


class FullyEquivariant2BodyNet(nn.Module):
    """
    Rete per 2 corpi contemporaneamente EQUIVARIANTE PER ROTAZIONE SO(2)
    e PERMUTAZIONE S_2 dei due corpi.
    """
    def __init__(self, hidden_channels=32, num_blocks=4, dtype=torch.float, device=torch.device("cpu")):
        super().__init__()
        self.dtype = dtype
        self.device = device
        self.num_bodies = 2

        # 2 canali in ingresso per corpo: [p_i, v_i]
        in_channels = 2
        # 2 canali in uscita per corpo: [dp_i, dv_i]
        out_channels = 2

        layers = [PermutationVectorABBlock(in_channels, hidden_channels, dtype=dtype, device=device)]
        gates = [PermutationInvariantGate(hidden_channels, dtype=dtype, device=device)]

        for _ in range(num_blocks - 2):
            layers.append(PermutationVectorABBlock(hidden_channels, hidden_channels, dtype=dtype, device=device))
            gates.append(PermutationInvariantGate(hidden_channels, dtype=dtype, device=device))

        layers.append(PermutationVectorABBlock(hidden_channels, out_channels, dtype=dtype, device=device))

        self.layers = nn.ModuleList(layers)
        self.gates = nn.ModuleList(gates)

        # Inizializzazione piccola dell'ultimo blocco residuo
        with torch.no_grad():
            self.layers[-1].ab_layer.A.mul_(1e-2)
            self.layers[-1].ab_layer.B.mul_(1e-2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Input x: [B, 10] -> [m1, x1, y1, vx1, vy1,  m2, x2, y2, vx2, vy2]
        Output:  [B, 10] nello stesso formato.
        """
        B = x.shape[0]
        x_r = x.view(B, 2, 5)

        m = x_r[:, :, 0:1]         # [B, 2, 1]
        p = x_r[:, :, 1:3]         # [B, 2, 2]
        v = x_r[:, :, 3:5]         # [B, 2, 2]

        # --- Momento lineare di ingresso ---
        P_in = torch.sum(m * v, dim=1)            # [B, 2]
        M_tot = torch.sum(m, dim=1, keepdim=True)  # [B, 1, 1]

        # --- Canali Vettoriali Iniziali per i 2 corpi: [B, 2, 2, 2] ---
        # Per ciascun corpo i: canale 0 -> p_i, canale 1 -> v_i
        vecs = torch.stack([p, v], dim=2)

        # --- Forward Pass nei Layer Equivarianti ---
        out = vecs
        for layer, gate in zip(self.layers[:-1], self.gates):
            out = layer(out)
            out = gate(out, m)
        
        # Ultimo layer senza gate (residuo puro)
        out = self.layers[-1](out) # [B, 2, 2, 2]

        # Estrazione dei residui per posizione e velocità
        dp = out[:, :, 0, :]  # [B, 2, 2]
        dv = out[:, :, 1, :]  # [B, 2, 2]

        p_next = p + dp
        v_next = v + dv

        # --- Correzione Hard del Momento Lineare (Preserva S_2 e SO(2)) ---
        P_out = torch.sum(m * v_next, dim=1, keepdim=True) # [B, 1, 2]
        corr = (P_out - P_in.unsqueeze(1)) / M_tot          # [B, 1, 2]
        v_next = v_next - corr                              # Sottratto in modo identico a entrambi i corpi

        # --- Ricostruzione Formato Output Flat ---
        out_r = torch.cat([m, p_next, v_next], dim=-1) # [B, 2, 5]
        return out_r.view(B, 10)








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