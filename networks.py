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
        dtype: torch.dtype = torch.float
    ) -> None:

        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.num_blocks = num_blocks

        ls = [AB_Block(in_feature=self.in_features, out_feature=hidden_dim)]

        for _ in range(num_blocks - 2):
            ls += [AB_Block(hidden_dim, hidden_dim, dtype=dtype)]

        ls += [nn.Linear(hidden_dim, out_features, dtype=dtype)]

        self.network_modules = nn.ModuleList(ls)


    def forward(self, x: torch.Tensor) -> torch.Tensor:

        for module in self.network_modules:
            x = module(x)

        return x