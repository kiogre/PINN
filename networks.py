import torch
import torch.nn as nn
import torch.nn.functional as F
from Equi_module.new_layer import AB_Block

class AB2Net(nn.Module):

    def __init__(
        self,
        in_features: int = 10, # just start with 2 dimension, 1 variable for m, 2 for p, 2 for q, for 2 bodies
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
            ls += [AB_Block(hidden_dim, hidden_dim, dtype=dtype, device = device)]
            ls += [nn.SiLU()]

        ls += [AB_Block(hidden_dim, out_features, dtype=dtype, device = device)]

        self.network_modules = nn.ModuleList(ls)


    def forward(self, x: torch.Tensor) -> torch.Tensor:

        m1 = x[:, 0:1]
        m2 = x[:, 5:6]

        for module in self.network_modules:
            x = module(x)

        x = torch.cat([m1, x[:, :4], m2, x[:, 4:]], dim=1)

        return x