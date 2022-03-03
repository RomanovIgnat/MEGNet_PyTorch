import torch
import torch.nn as nn
from .Layers import MegnetModule
from torch_geometric.nn import Set2Set


class MEGNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.m1 = MegnetModule(100, 1, 2)
        self.m2 = MegnetModule(32, 32, 32)
        self.m3 = MegnetModule(32, 32, 32)
        self.se = Set2Set(32, 1)
        self.sv = Set2Set(32, 1)
        self.hiddens = nn.Sequential(
            nn.Linear(160, 32),
            nn.Softplus(),
            nn.Linear(32, 16),
            nn.Softplus(),
            nn.Linear(16, 1)
        )

    def forward(self, x, edge_index, edge_attr, state, batch, bond_batch):
        x, edge_attr, state = self.m1(x, edge_index, edge_attr, state, batch, bond_batch)
        x, edge_attr, state = self.m2(x, edge_index, edge_attr, state, batch, bond_batch)
        x, edge_attr, state = self.m3(x, edge_index, edge_attr, state, batch, bond_batch)
        x = self.sv(x, batch)
        edge_attr = self.se(edge_attr, bond_batch)
        tmp = torch.cat((x, edge_attr, state), 1)
        out = self.hiddens(tmp)
        return out
