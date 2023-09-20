import torch.nn.functional as F

from torch_geometric.nn.conv import GATv2Conv
from torch_geometric.nn.models.basic_gnn import BasicGNN

# 1-headed attention only
class NegationGATv2Conv(GATv2Conv):
    def __init__(self, in_channels, out_channels, **kwargs):
        super().__init__(in_channels, out_channels, **kwargs)

    def forward(self, x, edge_index, **kwargs):
        x = super().forward(x, edge_index, **kwargs)
        return -F.normalize(x)
    
class NegationGAT(BasicGNN):
    supports_edge_weight = False
    supports_edge_attr = True

    def init_conv(self, in_channels, out_channels, **kwargs):
        conv = NegationGATv2Conv(in_channels, out_channels,
                                 concat=True, dropout=self.dropout, **kwargs)
        return conv

    def forward(self, x, edge_index, **kwargs):
        return super().forward(x, edge_index)
