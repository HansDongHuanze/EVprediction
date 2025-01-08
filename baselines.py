import torch
import torch.nn as nn
import models
import torch.nn.functional as F
import functions as fn
import copy

from torch.nn import Transformer, TransformerEncoder, TransformerEncoderLayer
from torch_geometric.nn import GraphConv

use_cuda = True
device = torch.device("cuda:0" if use_cuda and torch.cuda.is_available() else "cpu")
fn.set_seed(seed=2023, flag=True)


class VAR(nn.Module):
    def __init__(self, node=247, seq=12, feature=2):  # input_dim = seq_length
        super(VAR, self).__init__()
        self.linear = nn.Linear(node*seq*feature, node)

    def forward(self, occ, prc):
        x = torch.cat((occ, prc), dim=2)
        x = torch.flatten(x, 1, 2)
        x = self.linear(x)
        return x


class LSTM(nn.Module):
    def __init__(self, seq, n_fea, node=247):
        super(LSTM, self).__init__()
        self.nodes = node
        self.encoder = nn.Conv2d(self.nodes, self.nodes, (n_fea, n_fea))  # input.shape: [batch, channel, width, height]
        self.lstm = nn.LSTM(self.nodes, self.nodes, num_layers=2, batch_first=True)
        self.decoder = nn.Linear(seq-n_fea+1, 1)

    def forward(self, occ, prc):  # occ.shape = [batch, node, seq]
        x = torch.stack([occ, prc], dim=3)
        x = self.encoder(x)
        x = torch.transpose(x.squeeze(), 1, 2)  # shape [batch, seq-n_fea+1, node]
        x, _ = self.lstm(x)
        x = torch.transpose(x, 1, 2)  # shape [batch, node, seq-n_fea+1]
        x = self.decoder(x)
        x = torch.squeeze(x)
        return x

class TransformerModel(nn.Module):
    def __init__(self, input_dim, embedding_dim, hidden_dim, output_dim, n_layers, n_heads, pf_dim, dropout):
        super().__init__()

        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.pf_dim = pf_dim
        self.dropout = dropout

        self.input_linear = nn.Linear(24, embedding_dim)

        self.encoder_layer = nn.TransformerEncoderLayer(embedding_dim, n_heads, pf_dim, dropout)
        self.encoder = nn.TransformerEncoder(self.encoder_layer, n_layers)

        # Adjust the output layer to produce a sequence of the same length as the input
        self.fc = nn.Linear(embedding_dim, output_dim)  # output_dim should match the number of features you want per time step
        self.dropout = nn.Dropout(dropout)

    def forward(self, occ, prc):
        # Stack occ and prc together
        x = torch.stack([occ, prc], dim=-1)
        
        # Reshape to (batch_size, sequence_length, features) where features=12*2=24
        batch_size, seq_len, feature_size, _ = x.shape
        x = x.view(batch_size, seq_len, feature_size * 2)
        
        # Pass through the linear layer
        x = self.input_linear(x)

        # Use Transformer encoder
        embedded = self.dropout(x)
        embedded = self.encoder(embedded)
        
        # Apply output layer to each sequence element
        output = self.fc(embedded)
        return output[:,:,-1]


class GCN(nn.Module):
    def __init__(self, seq, n_fea, adj_dense):
        super(GCN, self).__init__()
        self.nodes = adj_dense.shape[0]
        self.encoder = nn.Conv2d(self.nodes, self.nodes, (n_fea, n_fea))
        self.gcn_l1 = nn.Linear(seq-n_fea+1, seq-n_fea+1)
        self.gcn_l2 = nn.Linear(seq-n_fea+1, seq-n_fea+1)
        self.A = adj_dense
        self.act = nn.ReLU()
        self.decoder = nn.Linear(seq-n_fea+1, 1)

        # calculate A_delta matrix
        deg = torch.sum(adj_dense, dim=0)
        deg = torch.diag(deg)
        deg_delta = torch.linalg.inv(torch.sqrt(deg))
        a_delta = torch.matmul(torch.matmul(deg_delta, adj_dense), deg_delta)
        self.A = a_delta

    def forward(self, occ, prc):  # occ.shape = [batch, node, seq]
        x = torch.stack([occ, prc], dim=3)
        x = self.encoder(x)
        x = x[:,:,:,-1]

        #  l1
        x = self.gcn_l1(x)
        x = torch.matmul(self.A, x)

        x = self.act(x)
        #  l2
        x = self.gcn_l2(x)
        x = torch.matmul(self.A, x)
        x = self.act(x)
        x = self.decoder(x)

        return x[:,:,-1]

class STGCN(nn.Module):
    def __init__(self, num_nodes, num_features, num_timesteps_input, num_timesteps_output, num_channels):
        super(STGCN, self).__init__()
        
        # Define the number of channels for each layer
        self.num_channels = num_channels
        
        # Define the ST-Conv blocks
        self.st_conv_blocks = nn.ModuleList([
            STConvBlock(num_features, num_channels[0], num_channels[1], num_channels[2]),
            STConvBlock(num_channels[1], num_channels[2], num_channels[3], num_channels[4])
        ])
        
        # Define the output layer
        self.output_layer = nn.Linear(num_channels[4], num_timesteps_output)
        
    def forward(self, x, edge_index):
        # Process the input through the ST-Conv blocks
        for block in self.st_conv_blocks:
            x = block(x, edge_index)
        
        # Apply the output layer
        x = self.output_layer(x)
        
        return x
class STConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels_spatial, out_channels_temporal1, out_channels_temporal2):
        super(STConvBlock, self).__init__()
        
        # Define the spatial graph convolution layer
        self.graph_conv = GraphConv(in_channels, out_channels_spatial)
        
        # Define the temporal gated convolution layers
        self.temporal_conv1 = nn.Conv1d(out_channels_spatial, out_channels_temporal1, kernel_size=3, padding=1)
        self.temporal_conv2 = nn.Conv1d(out_channels_temporal1, out_channels_temporal2, kernel_size=3, padding=1)
        
        # Define the residual connection
        self.residual_connection = nn.Conv1d(out_channels_temporal2, out_channels_spatial, kernel_size=1, bias=False)
        
    def forward(self, x, edge_index):
        # Apply the spatial graph convolution layer
        x_spatial = self.graph_conv(x, edge_index)
        
        # Apply the temporal gated convolution layers
        x_temporal1 = self.temporal_conv1(x_spatial)
        x_temporal2 = self.temporal_conv2(x_temporal1)
        
        # Apply the residual connection
        x_residual = self.residual_connection(x_temporal2)
        
        # Add the residual connection to the output of the temporal gated convolution layers
        x = x_temporal2 + x_residual
        
        # Apply the ReLU activation function
        x = F.relu(x)
        
        return x

class LstmGcn(nn.Module):
    def __init__(self, seq, n_fea, adj_dense):
        super(LstmGcn, self).__init__()
        self.A = adj_dense
        self.nodes = adj_dense.shape[0]
        self.encoder = nn.Conv2d(self.nodes, self.nodes, (n_fea, n_fea), device=device)
        self.gcn_l1 = nn.Linear(seq - n_fea + 1, seq - n_fea + 1, device=device)
        self.gcn_l2 = nn.Linear(seq - n_fea + 1, seq - n_fea + 1, device=device)
        self.lstm = nn.LSTM(self.nodes, self.nodes, num_layers=2, batch_first=True)
        self.act = nn.ReLU()
        self.decoder = nn.Linear(seq - n_fea + 1, 1, device=device)

        # calculate A_delta matrix
        deg = torch.sum(adj_dense, dim=0)
        deg = torch.diag(deg)
        deg_delta = torch.linalg.inv(torch.sqrt(deg))
        a_delta = torch.matmul(torch.matmul(deg_delta, adj_dense), deg_delta)
        self.A = a_delta

    def forward(self, occ, prc):  # occ.shape = [batch, node, seq]
        x = torch.stack([occ, prc], dim=3)
        x = self.encoder(x)
        x = torch.squeeze(x)
        #  l1
        x = self.gcn_l1(x)
        x = torch.matmul(self.A, x)
        x = self.act(x)
        #  l2
        x = self.gcn_l2(x)
        x = torch.matmul(self.A, x)
        x = self.act(x)
        # lstm
        x = x.transpose(1, 2)
        x, _ = self.lstm(x)
        x = x.transpose(1, 2)
        x = self.decoder(x)
        x = torch.squeeze(x)
        return x


class LstmGat(nn.Module):
    def __init__(self, seq, n_fea, adj_dense, adj_sparse):
        super(LstmGat, self).__init__()
        self.nodes = adj_dense.shape[0]
        self.gcn = nn.Linear(in_features=seq - n_fea + 1, out_features=seq - n_fea + 1, device=device)
        self.encoder = nn.Conv2d(self.nodes, self.nodes, (n_fea, n_fea), device=device)
        self.gat_l1 = models.MultiHeadsGATLayer(adj_sparse, seq - n_fea + 1, seq - n_fea + 1, 4, 0, 0.2)
        self.gat_l2 = models.MultiHeadsGATLayer(adj_sparse, seq - n_fea + 1, seq - n_fea + 1, 4, 0, 0.2)
        self.lstm = nn.LSTM(self.nodes, self.nodes, num_layers=2, batch_first=True)
        self.decoder = nn.Linear(seq - n_fea + 1, 1, device=device)

        # Activation
        self.dropout = nn.Dropout(p=0.5)
        self.LeakyReLU = nn.LeakyReLU()

    def forward(self, occ, prc):  # occ.shape = [batch, node, seq]
        x = torch.stack([occ, prc], dim=3)
        x = self.encoder(x)
        x = torch.squeeze(x)

        # first layer
        atts_mat = self.gat_l1(x)  # attention matrix, dense(nodes, nodes)
        occ_conv1 = torch.matmul(atts_mat, x)  # (b, n, s)
        occ_conv1 = self.dropout(self.LeakyReLU(self.gcn(occ_conv1)))

        # second layer
        atts_mat2 = self.gat_l2(occ_conv1)  # attention matrix, dense(nodes, nodes)
        occ_conv2 = torch.matmul(atts_mat2, occ_conv1)  # (b, n, s)
        occ_conv2 = self.dropout(self.LeakyReLU(self.gcn(occ_conv2)))

        # lstm
        x = occ_conv2.transpose(1, 2)
        x, _ = self.lstm(x)
        x = x.transpose(1, 2)

        # decode
        x = self.decoder(x)
        x = torch.squeeze(x)
        return x


class TPA(nn.Module):
    def __init__(self, seq, n_fea, nodes):
        super(TPA, self).__init__()
        self.nodes = nodes
        self.seq = seq
        self.n_fea = n_fea
        self.encoder = nn.Conv2d(self.nodes, self.nodes, (n_fea, n_fea), device=device)
        # TPA
        self.lstm = nn.LSTM(self.seq - 1, 2, num_layers=2, batch_first=True, device=device)
        self.fc1 = nn.Linear(in_features=self.seq - 1, out_features=2, device=device)
        self.fc2 = nn.Linear(in_features=2, out_features=2, device=device)
        self.fc3 = nn.Linear(in_features=2 + 2, out_features=1, device=device)
        self.decoder = nn.Linear(self.seq, 1, device=device)

    def forward(self, occ, prc):  # occ.shape = [batch, node, seq]
        x = torch.stack([occ, prc], dim=3)
        x = self.encoder(x)
        x = torch.squeeze(x)

        print("Shape of x:", x.shape)

        # TPA
        x = x.view(occ.shape[0] * occ.shape[1], occ.shape[2] - 1, self.n_fea)
        lstm_out, (_, _) = self.lstm(x)  # b*n, s, 2
        ht = lstm_out[:, -1, :]  # ht
        hw = lstm_out[:, :-1, :]  # from h(t-1) to h1
        hw = torch.transpose(hw, 1, 2)
        Hc = self.fc1(hw)
        Hn = self.fc2(Hc)
        ht = torch.unsqueeze(ht, dim=2)
        a = torch.bmm(Hn, ht)
        a = torch.sigmoid(a)
        a = torch.transpose(a, 1, 2)
        vt = torch.matmul(a, Hc)
        ht = torch.transpose(ht, 1, 2)
        hx = torch.cat((vt, ht), dim=2)
        y = self.fc3(hx)
        print(y.shape)
        return y


# https://doi.org/10.1016/j.trc.2023.104205
class HSTGCN(nn.Module):
    def __init__(self, seq, n_fea, adj_distance, adj_demand, alpha=0.5):
        super(HSTGCN, self).__init__()
        # hyper-params
        self.nodes = adj_distance.shape[0]
        self.alpha = alpha
        hidden = seq - n_fea + 1

        # network components
        self.encoder = nn.Conv2d(self.nodes, self.nodes, (n_fea, n_fea))
        self.linear = nn.Linear(hidden, hidden)
        self.distance_gcn_l1 = nn.Linear(hidden, hidden)
        self.distance_gcn_l2 = nn.Linear(hidden, hidden)
        self.gru1 = nn.GRU(self.nodes, self.nodes, num_layers=2, batch_first=True)
        self.demand_gcn_l1 = nn.Linear(hidden, hidden)
        self.demand_gcn_l2 = nn.Linear(hidden, hidden)
        self.gru2 = nn.GRU(self.nodes, self.nodes, num_layers=2, batch_first=True)
        self.decoder = nn.Sequential(nn.Linear(hidden, 16),
        nn.ReLU(),
        nn.Linear(16, 1)
        )
        
        self.act = nn.ReLU()
        self.dropout = nn.Dropout(p=0.5)

        # calculate A_delta matrix
        deg = torch.sum(adj_distance, dim=0)
        deg = torch.diag(deg)
        deg_delta = torch.linalg.inv(torch.sqrt(deg))
        a_delta = torch.matmul(torch.matmul(deg_delta, adj_distance), deg_delta)
        self.A_dis = a_delta

        deg = torch.sum(adj_demand, dim=0)
        deg = torch.diag(deg)
        deg_delta = torch.linalg.inv(torch.sqrt(deg))
        a_delta = torch.matmul(torch.matmul(deg_delta, adj_demand), deg_delta)
        self.A_dem = a_delta

    def forward(self, occ, prc):  # occ.shape = [batch, node, seq]
        x = torch.stack([occ, prc], dim=3)
        x = self.encoder(x)
        x = torch.squeeze(x)
        x = self.act(self.linear(x))

        # distance-based graph propagation
        #  l1
        x1 = self.distance_gcn_l1(x)
        x1 = torch.matmul(self.A_dis, x1)
        x1 = self.dropout(self.act(x1))
        #  l2
        x1 = self.distance_gcn_l2(x1)
        x1 = torch.matmul(self.A_dis, x1)
        x1 = self.dropout(self.act(x1))
        # gru
        x1 = x1.transpose(1, 2)
        x1, _ = self.gru1(x1)
        x1 = x1.transpose(1, 2)

        # demand-based graph propagation
        #  l1
        x2 = self.demand_gcn_l1(x)
        x2 = torch.matmul(self.A_dem, x2)
        x2 = self.dropout(self.act(x2))
        #  l2
        x2 = self.demand_gcn_l2(x2)
        x2 = torch.matmul(self.A_dem, x2)
        x2 = self.dropout(self.act(x2))
        # gru
        x2 = x2.transpose(1, 2)
        x2, _ = self.gru2(x2)
        x2 = x2.transpose(1, 2)

        # decode
        output = self.alpha * x1 + (1-self.alpha) * x2
        output = self.decoder(output)
        output = torch.squeeze(output)
        return output


# https://arxiv.org/abs/2311.06190
class FGN(nn.Module):
    def __init__(self, pre_length=1, embed_size=64,
                 feature_size=0, seq_length=12, hidden_size=32, hard_thresholding_fraction=1, hidden_size_factor=1, sparsity_threshold=0.01):
        super().__init__()
        self.embed_size = embed_size
        self.hidden_size = hidden_size
        self.number_frequency = 1
        self.pre_length = pre_length
        self.feature_size = feature_size
        self.seq_length = seq_length
        self.frequency_size = self.embed_size // self.number_frequency
        self.hidden_size_factor = hidden_size_factor
        self.sparsity_threshold = sparsity_threshold
        self.hard_thresholding_fraction = hard_thresholding_fraction
        self.scale = 0.02
        self.embeddings = nn.Parameter(torch.randn(1, self.embed_size))

        self.encoder = nn.Linear(2, 1)
        self.w1 = nn.Parameter(
            self.scale * torch.randn(2, self.frequency_size, self.frequency_size * self.hidden_size_factor))
        self.b1 = nn.Parameter(self.scale * torch.randn(2, self.frequency_size * self.hidden_size_factor))
        self.w2 = nn.Parameter(
            self.scale * torch.randn(2, self.frequency_size * self.hidden_size_factor, self.frequency_size))
        self.b2 = nn.Parameter(self.scale * torch.randn(2, self.frequency_size))
        self.w3 = nn.Parameter(
            self.scale * torch.randn(2, self.frequency_size,
                                     self.frequency_size * self.hidden_size_factor))
        self.b3 = nn.Parameter(
            self.scale * torch.randn(2, self.frequency_size * self.hidden_size_factor))
        self.embeddings_10 = nn.Parameter(torch.randn(self.seq_length, 8))
        self.fc = nn.Sequential(
            nn.Linear(self.embed_size * 8, 64),
            nn.LeakyReLU(),
            nn.Linear(64, self.hidden_size),
            nn.LeakyReLU(),
            nn.Linear(self.hidden_size, self.pre_length)
        )
        self.to('cuda:0')

    def tokenEmb(self, x):
        x = x.unsqueeze(2)
        y = self.embeddings
        return x * y

    # FourierGNN
    def fourierGC(self, x, B, N, L):
        o1_real = torch.zeros([B, (N*L)//2 + 1, self.frequency_size * self.hidden_size_factor],
                              device=x.device)
        o1_imag = torch.zeros([B, (N*L)//2 + 1, self.frequency_size * self.hidden_size_factor],
                              device=x.device)
        o2_real = torch.zeros(x.shape, device=x.device)
        o2_imag = torch.zeros(x.shape, device=x.device)

        o3_real = torch.zeros(x.shape, device=x.device)
        o3_imag = torch.zeros(x.shape, device=x.device)

        o1_real = F.relu(
            torch.einsum('bli,ii->bli', x.real, self.w1[0]) - \
            torch.einsum('bli,ii->bli', x.imag, self.w1[1]) + \
            self.b1[0]
        )

        o1_imag = F.relu(
            torch.einsum('bli,ii->bli', x.imag, self.w1[0]) + \
            torch.einsum('bli,ii->bli', x.real, self.w1[1]) + \
            self.b1[1]
        )

        # 1 layer
        y = torch.stack([o1_real, o1_imag], dim=-1)
        y = F.softshrink(y, lambd=self.sparsity_threshold)

        o2_real = F.relu(
            torch.einsum('bli,ii->bli', o1_real, self.w2[0]) - \
            torch.einsum('bli,ii->bli', o1_imag, self.w2[1]) + \
            self.b2[0]
        )

        o2_imag = F.relu(
            torch.einsum('bli,ii->bli', o1_imag, self.w2[0]) + \
            torch.einsum('bli,ii->bli', o1_real, self.w2[1]) + \
            self.b2[1]
        )

        # 2 layer
        x = torch.stack([o2_real, o2_imag], dim=-1)
        x = F.softshrink(x, lambd=self.sparsity_threshold)
        x = x + y

        o3_real = F.relu(
                torch.einsum('bli,ii->bli', o2_real, self.w3[0]) - \
                torch.einsum('bli,ii->bli', o2_imag, self.w3[1]) + \
                self.b3[0]
        )

        o3_imag = F.relu(
                torch.einsum('bli,ii->bli', o2_imag, self.w3[0]) + \
                torch.einsum('bli,ii->bli', o2_real, self.w3[1]) + \
                self.b3[1]
        )

        # 3 layer
        z = torch.stack([o3_real, o3_imag], dim=-1)
        z = F.softshrink(z, lambd=self.sparsity_threshold)
        z = z + x
        z = torch.view_as_complex(z)
        return z

    def forward(self, occ, prc):
        x = torch.stack([occ, prc], dim=3)
        x = self.encoder(x)
        x = torch.squeeze(x)

        B, N, L = x.shape
        # B*N*L ==> B*NL
        x = x.reshape(B, -1)
        # embedding B*NL ==> B*NL*D
        x = self.tokenEmb(x)

        # FFT B*NL*D ==> B*NT/2*D
        x = torch.fft.rfft(x, dim=1, norm='ortho')

        x = x.reshape(B, (N*L)//2+1, self.frequency_size)

        bias = x

        # FourierGNN
        x = self.fourierGC(x, B, N, L)

        x = x + bias

        x = x.reshape(B, (N*L)//2+1, self.embed_size)

        # ifft
        x = torch.fft.irfft(x, n=N*L, dim=1, norm="ortho")

        x = x.reshape(B, N, L, self.embed_size)
        x = x.permute(0, 1, 3, 2)  # B, N, D, L

        # projection
        x = torch.matmul(x, self.embeddings_10)
        x = x.reshape(B, N, -1)
        x = self.fc(x)
        x = torch.squeeze(x)
        return x

# Other baselines refer to its own original code.
