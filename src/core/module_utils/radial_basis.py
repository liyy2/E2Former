# -*- coding: utf-8 -*-
"""
Radial Basis module for E2Former.
"""

import math
import torch
import numpy as np
from torch import nn

# @torch.jit.script
def gaussian(x, mean, std):
    pi = torch.pi
    a = (2 * pi) ** 0.5
    return torch.exp(-0.5 * (((x - mean) / std) ** 2)) / (a * std)


from fairchem.core.common.utils import (
    compute_neighbors,
    get_max_neighbors_mask,
    get_pbc_distances,
)




class RadialFunction(nn.Module):
    """
    Contruct a radial function (linear layers + layer normalization + SiLU) given a list of channels
    """

    def __init__(self, channels_list, use_layer_norm=True):
        super().__init__()
        modules = []
        input_channels = channels_list[0]
        for i in range(len(channels_list)):
            if i == 0:
                continue

            modules.append(nn.Linear(input_channels, channels_list[i], bias=True))
            input_channels = channels_list[i]

            if i == len(channels_list) - 1:
                break
            if use_layer_norm:
                modules.append(nn.LayerNorm(channels_list[i]))
            modules.append(torch.nn.SiLU())

        self.net = nn.Sequential(*modules)

    def forward(self, inputs):
        return self.net(inputs)


# From Graphormer
class GaussianRadialBasisLayer(torch.nn.Module):
    def __init__(self, num_basis, cutoff):
        super().__init__()
        self.num_basis = num_basis
        self.cutoff = cutoff + 0.0
        self.mean = torch.nn.Parameter(torch.zeros(1, self.num_basis))
        self.std = torch.nn.Parameter(torch.zeros(1, self.num_basis))
        self.weight = torch.nn.Parameter(torch.ones(1, 1))
        self.bias = torch.nn.Parameter(torch.zeros(1, 1))

        self.std_init_max = 1.0
        self.std_init_min = 1.0 / self.num_basis
        self.mean_init_max = 1.0
        self.mean_init_min = 0
        torch.nn.init.uniform_(self.mean, self.mean_init_min, self.mean_init_max)
        torch.nn.init.uniform_(self.std, self.std_init_min, self.std_init_max)
        torch.nn.init.constant_(self.weight, 1)
        torch.nn.init.constant_(self.bias, 0)

    def forward(self, dist, node_atom=None, edge_src=None, edge_dst=None):
        x = dist / self.cutoff
        x = x.unsqueeze(-1)
        x = self.weight * x + self.bias
        x = x.expand(-1, -1, self.num_basis) if x.dim() == 3 else x.expand(-1, self.num_basis)
        mean = self.mean
        std = self.std.abs() + 1e-5
        x = gaussian(x, mean, std)
        return x

    def extra_repr(self):
        return "mean_init_max={}, mean_init_min={}, std_init_max={}, std_init_min={}".format(
            self.mean_init_max, self.mean_init_min, self.std_init_max, self.std_init_min
        )




class GaussianSmearing(torch.nn.Module):
    def __init__(
        self,
        num_basis,
        cutoff: float = 5.0,
        basis_width_scalar: float = 2.0,
    ) -> None:
        super().__init__()
        offset = torch.linspace(0, cutoff, num_basis)
        self.coeff = -0.5 / (basis_width_scalar * (offset[1] - offset[0])).item() ** 2
        self.register_buffer("offset", offset)

    def forward(self, dist) -> torch.Tensor:
        shape = dist.shape
        dist = dist.view(-1, 1) - self.offset.view(1, -1)
        dist = torch.exp(self.coeff * torch.pow(dist, 2))
        return dist.reshape(*shape, -1)


# gaussian layer with edge type (i,j)


class GaussianLayer_Edgetype(nn.Module):
    def __init__(self, K=128, edge_types=512 * 3):
        super().__init__()
        self.K = K
        self.means = nn.Embedding(1, K)
        self.stds = nn.Embedding(1, K)
        self.mul = nn.Embedding(edge_types, 1, padding_idx=0)
        self.bias = nn.Embedding(edge_types, 1, padding_idx=0)
        nn.init.uniform_(self.means.weight, 0, 3)
        nn.init.uniform_(self.stds.weight, 0, 3)
        nn.init.constant_(self.bias.weight, 0)
        nn.init.constant_(self.mul.weight, 1)

    def forward(self, x, edge_types):
        '''
        x:*,*,*
        edge_types:*,*,*,2
        '''
        out_shape = x.shape
        x = x.view(-1)
        edge_types = edge_types.view(out_shape.numel(),2)
        mul = self.mul(edge_types).sum(dim=-2)
        bias = self.bias(edge_types).sum(dim=-2)
        x = mul * x.unsqueeze(-1) + bias
        # print(x.shape)
        # x = x.expand(-1, -1, -1, self.K)
        mean = self.means.weight.float().view(1,-1)
        std = self.stds.weight.float().view(1,-1).abs() + 1e-2
        x_rbf = gaussian(x.float(), mean, std).type_as(self.means.weight)
        return x_rbf.reshape(out_shape+(-1,))



# in farchem, the max_rep = [rep_a1.max(), rep_a2.max(), rep_a3.max()] is very big for some special case in oc20 dataset
# thus we use the max_rep clip to avoid this issue


def polynomial(dist: torch.Tensor, cutoff: float) -> torch.Tensor:
    """
    Polynomial cutoff function,ref: https://arxiv.org/abs/2204.13639
    Args:
        dist (tf.Tensor): distance tensor
        cutoff (float): cutoff distance
    Returns: polynomial cutoff functions
    """
    ratio = torch.div(dist, cutoff)
    result = (
        1
        - 6 * torch.pow(ratio, 5)
        + 15 * torch.pow(ratio, 4)
        - 10 * torch.pow(ratio, 3)
    )
    return torch.clamp(result, min=0.0)




class RadialProfile(nn.Module):
    def __init__(self, ch_list, use_layer_norm=True, use_offset=True):
        super().__init__()
        modules = []
        input_channels = ch_list[0]
        for i in range(len(ch_list)):
            if i == 0:
                continue
            modules.append(nn.Linear(input_channels, ch_list[i], bias=use_offset))
            input_channels = ch_list[i]

            if i == len(ch_list) - 1:
                break

            if use_layer_norm:
                modules.append(nn.LayerNorm(ch_list[i]))
            modules.append(torch.nn.SiLU())

        self.net = nn.Sequential(*modules)

    def forward(self, f_in):
        f_out = self.net(f_in)
        return f_out




