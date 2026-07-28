import torch
import torch.nn.functional as F

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
    
class multi_AB_Block(torch.nn.Module):
    """
    Linear layer parametrized as [A,B;B,A] such that if we permutate the input (instead of [x,y]
    we do [y,x]) then the output is permutated as well.
    This class is to do the permutation with n_obj elements, obviusly it requires that the size
    of the layer is a multiple of n_obj, in input and in output.
    """

    def __init__(
        self,
        in_feature: int,
        out_feature: int,
        n_obj: int,
        bias: bool = True,
        dtype: torch.dtype = torch.float,
        device = None
    ) -> None:
        super().__init__()

        if in_feature % n_obj != 0 or out_feature % n_obj != 0:
            raise Exception("Error, for preserving the switch we need an amount of lines divisible " \
            "with the number of objects.")

        # A & B
        self.A = torch.nn.Parameter(torch.empty(in_feature//n_obj, out_feature//n_obj, dtype=dtype, device=device))
        self.B = torch.nn.Parameter(torch.empty(in_feature//n_obj, out_feature//n_obj, dtype=dtype, device=device))

        torch.nn.init.xavier_normal_(self.A)
        torch.nn.init.xavier_normal_(self.B)

        for i in range(n_obj):
            if i == 0:
                self.AB = torch.cat((self.A, self.B), 0)
            elif i == 1:
                self.AB = torch.cat((self.B, self.A), 0)
            else:
                self.AB = torch.cat((self.B, self.B), 0)
            for j in range(2, n_obj):
                if i == j:
                    self.AB = torch.cat((self.AB, self.A), 0)
                else:
                    self.AB = torch.cat((self.AB, self.B), 0)
            
            if i == 0:
                self.matrix = self.AB.clone()
            else:
                self.matrix = torch.cat((self.matrix, self.AB), 1)

        # bias
        self.bias: Optional[torch.nn.Parameter] = None
        if bias:
            self.bias = torch.nn.Parameter(torch.empty(out_feature, dtype=dtype))
            torch.nn.init.normal_(self.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply equivariance weight matrix"""

        xp = torch.matmul(self.matrix, x.unsqueeze(-1)).squeeze(1)

        if self.bias is not None:
            xp = xp + self.bias

        return xp