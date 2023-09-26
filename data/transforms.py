from typing import Any, Optional

import networkx
import numpy as np

import torch
from torch_geometric.data import Data
from torch_geometric.data.datapipes import functional_transform
from torch_geometric.transforms import BaseTransform
from torch_geometric.utils import (
    get_laplacian,
    get_self_loop_attr,
    scatter,
    to_dense_adj,
    to_torch_csr_tensor,
)
from utils.graph_utils import complement_graph

# cribbed from torch_geometric.transforms, modified to use dense eigenvalue functions
def add_node_attr(data: Data, value: Any,
                  attr_name: Optional[str] = None) -> Data:
    # TODO Move to `BaseTransform`.
    if attr_name is None:
        if 'x' in data:
            x = data.x.view(-1, 1) if data.x.dim() == 1 else data.x
            data.x = torch.cat([x, value.to(x.device, x.dtype)], dim=-1)
        else:
            data.x = value
    else:
        data[attr_name] = value

    return data

@functional_transform('add_laplacian_eigenvectors')
class AddLaplacianEigenvectorPE(BaseTransform):
    r"""Adds the Laplacian eigenvector positional encoding from the
    `"Benchmarking Graph Neural Networks" <https://arxiv.org/abs/2003.00982>`_
    paper to the given graph
    (functional name: :obj:`add_laplacian_eigenvector_pe`).

    Args:
        k (int): The number of non-trivial eigenvectors to consider.
        attr_name (str, optional): The attribute name of the data object to add
            positional encodings to. If set to :obj:`None`, will be
            concatenated to :obj:`data.x`.
            (default: :obj:`"laplacian_eigenvector_pe"`)
        is_undirected (bool, optional): If set to :obj:`True`, this transform
            expects undirected graphs as input, and can hence speed up the
            computation of eigenvectors. (default: :obj:`False`)
        **kwargs (optional): Additional arguments of
            :meth:`scipy.linalg.eig` (when :attr:`is_undirected` is
            :obj:`False`) or :meth:`scipy.linalg.eigh` (when
            :attr:`is_undirected` is :obj:`True`).
    """
    def __init__(
        self,
        k: int,
        attr_name: Optional[str] = 'laplacian_eigenvector_pe',
        is_undirected: bool = False,
        **kwargs,
    ):
        self.k = k
        self.attr_name = attr_name
        self.is_undirected = is_undirected
        self.kwargs = kwargs

    def __call__(self, data: Data) -> Data:
        from scipy.linalg import eig, eigh
        eig_fn = eig if not self.is_undirected else eigh

        num_nodes = data.num_nodes
        edge_index, edge_weight = get_laplacian(
            data.edge_index,
            data.edge_weight,
            normalization='sym',
            num_nodes=num_nodes,
        )

        L = to_dense_adj(edge_index, edge_attr=edge_weight, max_num_nodes=num_nodes)[0]
        #print("SHAPE", L.shape)

        eig_vals, eig_vecs = eig_fn(L, **self.kwargs)
        eig_vecs = np.real(eig_vecs[:, eig_vals.argsort()])
        #data = add_node_attr(data, eig_vecs, attr_name=self.attr_name)

        if num_nodes < self.k + 2:
            padding = self.k + 2 - num_nodes
            eig_vecs = np.concatenate((eig_vecs, np.zeros((num_nodes, padding), dtype=np.float32)), axis=1)
        pe = torch.from_numpy(eig_vecs[:, 1:self.k + 1])
        sign = -1 + 2 * torch.randint(0, 2, (self.k, ))
        pe *= sign

        data = add_node_attr(data, pe, attr_name=self.attr_name)
        return data

@functional_transform('to_complement')
class ToComplement(BaseTransform):
    r"""Complement of graph."""
    def __init__(self, **kwargs):
        self.kwargs = kwargs

    def __call__(self, data: Data) -> Data:
        return complement_graph(data)

