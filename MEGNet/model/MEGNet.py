import torch
import torch.nn as nn
import torch.nn.functional as F
from .Layers import MegnetModule, ShiftedSoftplus
from torch_geometric.nn import Set2Set


ATOMIC_NUMBERS = 95


class MEGNet(nn.Module):
    def __init__(self,
                 edge_input_shape,
                 node_input_shape,
                 state_input_shape,
                 node_embedding_size=16,
                 embedding_size=32,
                 n_blocks=3,
                 ):
        """
        Parameters
        ----------
        edge_input_shape: size of edge features'
        node_input_shape: size of node features'
        state_input_shape: size of global state features'
        node_embedding_size: if using embedding layer the size of result embeddings
        embedding_size: size of inner embeddings
        n_blocks: amount of MEGNet blocks
        """
        super().__init__()
        self.embedded = node_input_shape is None
        if self.embedded:
            node_input_shape = node_embedding_size
            self.emb = nn.Embedding(ATOMIC_NUMBERS, node_embedding_size)

        self.m1 = MegnetModule(
            edge_input_shape, node_input_shape, state_input_shape, inner_skip=True, embed_size=embedding_size
        )
        self.blocks = nn.ModuleList()
        for i in range(n_blocks - 1):
            self.blocks.append(MegnetModule(embedding_size, embedding_size, embedding_size))

        self.se = Set2Set(embedding_size, 1)
        self.sv = Set2Set(embedding_size, 1)
        self.hiddens = nn.Sequential(
            nn.Linear(5 * embedding_size, embedding_size),
            ShiftedSoftplus(),
            nn.Linear(embedding_size, embedding_size // 2),
            ShiftedSoftplus(),
            nn.Linear(embedding_size // 2, 1)
        )

    def forward(self, x, edge_index, edge_attr, state, batch, bond_batch):
        if self.embedded:
            x = self.emb(x).squeeze()
        else:
            x = x.float()

        x, edge_attr, state = self.m1(x, edge_index, edge_attr, state, batch, bond_batch)
        for block in self.blocks:
            x, edge_attr, state = block(x, edge_index, edge_attr, state, batch, bond_batch)
        x = self.sv(x, batch)
        edge_attr = self.se(edge_attr, bond_batch)

        tmp_shape = x.shape[0] - edge_attr.shape[0]
        edge_attr = F.pad(edge_attr, (0, 0, 0, tmp_shape), value=0.0)

        tmp = torch.cat((x, edge_attr, state), 1)
        out = self.hiddens(tmp)
        return out
