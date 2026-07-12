import torch
import torch.nn.functional as F

from typing import Optional

class LUBlock(torch.nn.Module):
    """
    Linear layer parametrized as [A,B;B,A] such that if we exchange the input (instead of [x,y]
    we do [y,x]) then the output is exchanged as well
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