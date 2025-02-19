import torch
import torch.nn as nn
import models
import torch.nn.functional as F
import functions as fn
import copy

from torch.nn import Transformer, TransformerEncoder, TransformerEncoderLayer
from torch_geometric.nn import GATConv

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

class FCN(nn.Module):
    def __init__(self, node=247, seq=12, feature=2, hidden_dim=128, num_layers=2):
        super(FCN, self).__init__()
        input_dim = node * seq * feature
        
        # 定义一个简单的全连接层序列
        layers = []
        layers.append(nn.Linear(input_dim, hidden_dim))
        layers.append(nn.ReLU())  # 使用 ReLU 作为激活函数
        
        for _ in range(num_layers - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.ReLU())
        
        layers.append(nn.Linear(hidden_dim, node))
        
        self.network = nn.Sequential(*layers)

    def forward(self, occ, prc):
        # 合并和展平输入数据
        x = torch.cat((occ, prc), dim=2)
        x = torch.flatten(x, 1, 2)
        
        # 前向传播
        x = self.network(x)
        
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
    
class GAT(nn.Module):
    def __init__(self, seq, n_fea, adj_dense, out_features=1, dropout=0.6, alpha=0.2):
        super(GAT, self).__init__()
        self.nodes = adj_dense.shape[0]
        self.dropout = dropout
        self.seq = seq

        # 初始化 GATConv 层
        self.gat1 = GATConv(n_fea, 3, dropout=dropout, heads=3)
        self.gat2 = GATConv(9, out_features, dropout=dropout, heads=1)
        self.decoder = nn.Linear(seq, 1)
        self.adj = adj_dense

    def forward(self, occ, prc):
        x = torch.stack([occ, prc], dim=2)  # x.shape: (batch, node, seq, 2)
        x = x.view(-1, x.size(2))  # Flatten batch and node dimensions

        # 需要将 adj_dense 转换为 edge_index 格式
        edge_index = self.adj.nonzero(as_tuple=False).t().contiguous()

        # 调用 GATConv 层
        x = self.gat1(x, edge_index)
        x = F.elu(x)
        x = self.gat2(x, edge_index)

        x = F.dropout(x, self.dropout, training=self.training)

        x = x.view(-1, self.nodes, self.seq)  # Reshape back to (batch, node, seq)
        x = self.decoder(x)
        return x.squeeze(-1)

class TemporalGatedConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size):
        super(TemporalGatedConv, self).__init__()
        self.causal_conv = nn.Conv1d(in_channels, out_channels * 2, kernel_size, padding=1)  # GLU requires out_channels * 2
        self.glu = nn.GLU(dim=1)

    def forward(self, x):
        # x.shape = [batch, node, seq, features]
        batch_size, node, seq, features = x.size()
        x = x.reshape(batch_size * node, features, seq)  # Reshape to [batch_size * node, features, seq]
        x = self.causal_conv(x)  # Apply causal convolution
        x = self.glu(x)  # Apply GLU
        
        # Check output shape after causal_conv
        # Output shape will be [batch_size * node, out_channels, seq]
        out_channels = self.causal_conv.out_channels // 2  # Since we used GLU
        x = x.reshape(batch_size, node, out_channels, seq)  # Reshape back to [batch_size, node, out_channels, seq]
        return x

class SpatialGraphConv(nn.Module):
    def __init__(self, in_features, out_features, adj_dense):
        super(SpatialGraphConv, self).__init__()
        self.A = adj_dense
        self.linear = nn.Linear(in_features, out_features)

        # Calculate A_delta matrix
        deg = torch.sum(adj_dense, dim=0)
        deg = torch.diag(deg)
        deg_delta = torch.linalg.inv(torch.sqrt(deg))
        self.A = torch.matmul(torch.matmul(deg_delta, adj_dense), deg_delta)

    def forward(self, x):
        # x.shape = [batch, node, seq, features]
        batch_size, node, features, seq = x.size()
        x = x.reshape(batch_size * node, seq, features)
        x = self.linear(x)  # Apply linear transformation
        x = x.reshape(batch_size, -1, seq, node)  # Reshape back to [batch_size, node, seq, out_features]
        x = torch.matmul(x, self.A)  # Apply spatial graph conv
        return x

class STGCN(nn.Module):
    def __init__(self, seq, n_fea, adj_dense):
        super(STGCN, self).__init__()
        self.temporal_conv1 = TemporalGatedConv(n_fea, n_fea, kernel_size=3)
        self.spatial_conv = SpatialGraphConv(n_fea, n_fea, adj_dense)
        self.temporal_conv2 = TemporalGatedConv(n_fea, n_fea, kernel_size=3)
        self.decoder = nn.Linear(n_fea * seq, 1)

    def forward(self, occ, prc):  # occ.shape = [batch, node, seq]
        x = torch.stack([occ, prc], dim=1)  # Shape: [batch, 2, node, seq]
        x = x.permute(0, 2, 3, 1)  # Shape: [batch, node, seq, 2]
        x = x.view(x.size(0), x.size(1), -1, x.size(3))  # Shape: [batch, node, seq, features]

        # Temporal Convolution Layer 1
        residual = x  # Save input for residual connection
        x = self.temporal_conv1(x)  # Apply temporal gated conv
        dim1, dim2, dim3, dim4 = x.shape
        x = x + residual.reshape(dim1, dim2, dim3, -1)  # Residual connection

        # Spatial Graph Convolution Layer
        x = self.spatial_conv(x)  # Apply spatial graph conv

        # Temporal Convolution Layer 2
        x = x.reshape(dim1, dim2, dim4, dim3)
        residual = x  # Save input for residual connection
        x = self.temporal_conv2(x)  # Apply temporal gated conv
        x = x + residual.reshape(dim1, dim2, dim3, -1)  # Residual connection

        # Final output
        dim = x.shape
        x = self.decoder(x.reshape(dim[0], dim[1], -1))  # Get final output for the last time step
        return x[:, :, -1]

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
