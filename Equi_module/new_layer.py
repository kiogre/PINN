import torch
import torch.nn.functional as F
import torch.nn as nn

from typing import Optional

class AB_Block(torch.nn.Module):
    def __init__(self, in_feature, out_feature, bias=True, dtype=torch.float, device=None):
        super().__init__()
        if in_feature % 2 != 0 or out_feature % 2 != 0:
            raise Exception("in_feature e out_feature devono essere pari")

        half_in, half_out = in_feature // 2, out_feature // 2

        self.A = torch.nn.Parameter(torch.empty(half_out, half_in, dtype=dtype, device=device))
        self.B = torch.nn.Parameter(torch.empty(half_out, half_in, dtype=dtype, device=device))
        torch.nn.init.xavier_normal_(self.A)
        torch.nn.init.xavier_normal_(self.B)

        self.bias_half: Optional[torch.nn.Parameter] = None
        if bias:
            self.bias_half = torch.nn.Parameter(torch.empty(half_out, dtype=dtype, device=device))
            torch.nn.init.normal_(self.bias_half)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        top = torch.cat((self.A, self.B), dim=1)     # (out/2, in)
        bottom = torch.cat((self.B, self.A), dim=1)   # (out/2, in)
        matrix = torch.cat((top, bottom), dim=0)      # (out, in) — ricostruita ad ogni forward

        xp = torch.matmul(matrix, x.unsqueeze(-1)).squeeze(-1)

        if self.bias_half is not None:
            xp = xp + torch.cat((self.bias_half, self.bias_half), dim=0)

        return xp
    
class NBodyPermutationBlock(nn.Module):
    """
    Blocco lineare Equivariante per N corpi (Permutazione S_N).
    Input:  x di forma [Batch, N, In_Channels] o [Batch, N, Dim, In_Channels]
    Output: y di forma [Batch, N, Out_Channels] o [Batch, N, Dim, Out_Channels]
    
    Funziona con QUALSIASI valore di N senza dover ri-creare matrici!
    """
    def __init__(self, in_channels: int, out_channels: int, bias: bool = True, dtype=torch.float64, device=None):
        super().__init__()
        # Trasformazione per l'interazione con se stesso (A)
        self.lin_self = nn.Linear(in_channels, out_channels, bias=False, dtype=dtype, device=device)
        # Trasformazione per l'interazione con gli ALTRI corpi (B)
        self.lin_other = nn.Linear(in_channels, out_channels, bias=bias, dtype=dtype, device=device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x ha forma: [Batch, N, ..., In_Channels]
        # Sum lungo la dimensione dei corpi (dim 1)
        sum_all = torch.sum(x, dim=1, keepdim=True)  # [Batch, 1, ..., In_Channels]
        sum_others = sum_all - x                     # [Batch, N, ..., In_Channels] (S - x_i)

        # y_i = A(x_i) + B(sum_{j != i} x_j)
        out = self.lin_self(x) + self.lin_other(sum_others)
        return out