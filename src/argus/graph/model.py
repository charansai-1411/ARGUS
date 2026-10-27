"""Edge-classification GNN over the heterogeneous transaction graph.

Each layer passes messages along every relation (HeteroConv). Transfer relations use an
edge-feature-aware conv (GINE or GATv2) in both directions — direction matters in laundering
(fan-out vs fan-in). Ownership and bank relations use GraphSAGE. A transfer is then scored
from [h_src, h_dst, edge_features].
"""

import torch
from torch import nn
from torch_geometric.nn import GATv2Conv, GINEConv, HeteroConv, SAGEConv

from argus.graph.build import REV_TRANSFER, TRANSFER

CONVS = ("gine", "gat", "sage")


def _transfer_conv(kind: str, hidden: int, heads: int = 4):
    if kind == "gine":
        mlp = nn.Sequential(nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, hidden))
        return GINEConv(mlp, edge_dim=hidden)
    if kind == "gat":
        return GATv2Conv(hidden, hidden // heads, heads=heads, edge_dim=hidden, add_self_loops=False)
    if kind == "sage":
        return SAGEConv(hidden, hidden)
    raise ValueError(f"conv must be one of {CONVS}")


class EdgeGNN(nn.Module):
    def __init__(self, metadata, in_dims: dict, edge_dim: int, hidden: int = 64,
                 layers: int = 2, conv: str = "gine", dropout: float = 0.1):
        super().__init__()
        node_types, edge_types = metadata
        self.conv_kind = conv
        self.node_in = nn.ModuleDict({nt: nn.Linear(in_dims[nt], hidden) for nt in node_types})
        self.edge_in = nn.Sequential(nn.Linear(edge_dim, hidden), nn.ReLU(), nn.Linear(hidden, hidden))
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        for _ in range(layers):
            rel = {et: (_transfer_conv(conv, hidden) if et in (TRANSFER, REV_TRANSFER)
                        else SAGEConv((hidden, hidden), hidden))
                   for et in edge_types}
            self.convs.append(HeteroConv(rel, aggr="sum"))
            self.norms.append(nn.ModuleDict({nt: nn.LayerNorm(hidden) for nt in node_types}))
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Sequential(
            nn.Linear(3 * hidden, hidden), nn.ReLU(), nn.Dropout(dropout), nn.Linear(hidden, 1))

    def forward(self, x_dict, edge_index_dict, edge_attr_dict, label_index, label_attr):
        """label_index: [2, B] local (src, dst) account ids of the transfers to score;
        label_attr: [B, edge_dim] their raw edge features. Returns logits [B]."""
        h = {nt: torch.relu(self.node_in[nt](x)) for nt, x in x_dict.items()}
        e = {}
        if self.conv_kind != "sage":  # SAGE ignores edge features
            e = {et: self.edge_in(a) for et, a in edge_attr_dict.items() if et in (TRANSFER, REV_TRANSFER)}
        for conv, norm in zip(self.convs, self.norms):
            out = conv(h, edge_index_dict, edge_attr_dict=e) if e else conv(h, edge_index_dict)
            h = {nt: h[nt] + self.dropout(torch.relu(norm[nt](out[nt]))) if nt in out else h[nt]
                 for nt in h}
        acc = h["account"]
        z = torch.cat([acc[label_index[0]], acc[label_index[1]], self.edge_in(label_attr)], dim=-1)
        return self.head(z).squeeze(-1)
