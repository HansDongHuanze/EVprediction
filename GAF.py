import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv

class GATWithFourier(nn.Module):
    def __init__(self, seq, n_fea, adj_dense, out_features=1, dropout=0.6, alpha=0.2):
        super(GATWithFourier, self).__init__()
        self.nodes = adj_dense.shape[0]
        self.dropout = dropout
        self.seq = seq
        self.n_fea = n_fea
        self.four_dim = seq // 2 + 1

        # Initialize GATConv layers
        self.gat1 = GATConv(n_fea, 10, dropout=dropout, heads=3)
        self.gat2 = GATConv(30, out_features, dropout=dropout, heads=1)
        self.decoder = nn.Linear(self.four_dim, 1)
        self.adj = adj_dense

    def forward(self, occ, prc):
        # Stack features and apply Fourier transform along the sequence dimension
        x = torch.stack([occ, prc], dim=-1)  # x.shape: (batch, node, seq, 2)
        
        # Perform FFT along the sequence dimension
        x = torch.fft.rfft(x, dim=2)  # Transform along the seq dimension (time)
        
        # Use the real part of the transformed features
        x = x.real
        
        # Flatten batch and node dimensions
        
        x = x.reshape(-1, x.size(-1))  # x.shape now: (batch*node, seq, features)
        # Convert adj_dense to edge_index format for GAT
        edge_index = self.adj.nonzero(as_tuple=False).t().contiguous()

        # Apply GAT layers
        x = self.gat1(x, edge_index)
        x = F.elu(x)
        x = self.gat2(x, edge_index)

        # Apply dropout
        x = F.dropout(x, self.dropout, training=self.training)

        # Reshape back to (batch, node, seq)
        x = x.view(-1, self.nodes, self.four_dim)
        x = self.decoder(x)
        return x.squeeze(-1)