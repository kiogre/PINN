import torch
import torch.nn.functional as F

from typing import Optional

class AB_Block(torch.nn.Module):
    """
    Linear layer parametrized as [A,B;B,A] such that if we permutate the input (instead of [x,y]
    we do [y,x]) then the output is permutated as well
    """

    def __init__(
        self,
        in_feature: int,
        out_feature: int,
        bias: bool = True,
        dtype: torch.dtype = torch.float,
        device = None
    ) -> None:
        super().__init__()

        if in_feature % 2 != 0 and out_feature % 2 != 0:
            raise Exception("Error, for preserving the switch we need an even amount of input feature" \
            "and output feature")

        # A & B
        self.A = torch.nn.Parameter(torch.empty(in_feature/2, out_feature/2, dtype=dtype, device=device))
        self.B = torch.nn.Parameter(torch.empty(in_feature/2, out_feature/2, dtype=dtype, device=device))

        self.AB = torch.cat((self.A, self.B), 0)
        self.BA = torch.cat((self.B, self.A), 0)

        self.matrix = torch.cat((self.AB, self.BA), 1)

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

        if in_feature % n_obj != 0 and out_feature % n_obj != 0:
            raise Exception("Error, for preserving the switch we need an amount of lines divisible " \
            "with the number of objects.")

        # A & B
        self.A = torch.nn.Parameter(torch.empty(in_feature/n_obj, out_feature/n_obj, dtype=dtype, device=device))
        self.B = torch.nn.Parameter(torch.empty(in_feature/n_obj, out_feature/n_obj, dtype=dtype, device=device))

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