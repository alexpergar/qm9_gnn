"""
DimeNet++ model implementation for molecular property prediction.
"""


import torch
from torch_geometric.nn import DimeNetPlusPlus


class DimeNetPlusPlusModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.model = DimeNetPlusPlus(
        hidden_channels=128,
        out_channels=1,
        num_blocks=4,
        int_emb_size=64,
        basis_emb_size=8,
        out_emb_channels=256,
        num_spherical=7,
        num_radial=6,
        cutoff=5.0,
        envelope_exponent=5,
        num_before_skip=1,
        num_after_skip=2,
        num_output_layers=3
    )