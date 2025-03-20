import torch
import torch.nn as nn
import models
import torch.nn.functional as F
import functions as fn
import copy

import math
from torch.utils.checkpoint import checkpoint

from torch.nn import Transformer, TransformerEncoder, TransformerEncoderLayer

use_cuda = True
device = torch.device("cuda:0" if use_cuda and torch.cuda.is_available() else "cpu")
fn.set_seed(seed=2023, flag=True)

class GraphAttentionLayer(nn.Module):

    def __init__(self, in_features, out_features, dropout, alpha, 
                 concat=True, num_nodes=247, hidden_size_factor=1, embed_size=32, sparsity_threshold=0.01):
        super(GraphAttentionLayer, self).__init__()
        self.dropout = dropout
        self.in_features = in_features
        self.out_features = out_features
        self.alpha = alpha
        self.concat = concat
        self.num_nodes = num_nodes

        '''self.embed_size = embed_size
        self.number_frequency = 1
        self.frequency_size = self.embed_size // self.number_frequency
        self.embeddings = nn.Parameter(torch.randn(1, self.embed_size))
        self.decoder2 = nn.Linear(self.embed_size, 1)
        self.hidden_size_factor = hidden_size_factor
        self.scale = 0.02
        self.sparsity_threshold=sparsity_threshold'''

        self.W = nn.Parameter(torch.zeros(size=(in_features, out_features)))
        nn.init.kaiming_normal_(self.W, mode='fan_out', nonlinearity='leaky_relu')
        self.a = nn.Parameter(torch.empty(2*out_features, 1))
        nn.init.xavier_normal_(self.a, gain=nn.init.calculate_gain('leaky_relu', param=alpha))
        self.node_weights = nn.Parameter(torch.randn(num_nodes))
        self.node_bias = nn.Parameter(torch.zeros(num_nodes))
        # self.NodeWiseTransform = NodeWiseTransform(num_nodes)
        self.norm = nn.LayerNorm(out_features)

        '''self.w1 = nn.Parameter(
            self.scale * torch.randn(2, self.frequency_size, self.frequency_size * self.hidden_size_factor))
        self.b1 = nn.Parameter(self.scale * torch.randn(2, self.frequency_size * self.hidden_size_factor))
        self.w2 = nn.Parameter(
            self.scale * torch.randn(2, self.frequency_size * self.hidden_size_factor, self.frequency_size))
        self.b2 = nn.Parameter(self.scale * torch.randn(2, self.frequency_size))'''

        self.leakyrelu = nn.LeakyReLU(self.alpha)

        '''self.W_q = nn.ParameterList([
            nn.Parameter(torch.randn((num_nodes * in_features) // 2 + 1, (num_nodes * in_features) // 2 + 1)),
            nn.Parameter(torch.randn((num_nodes * in_features) // 2 + 1, (num_nodes * in_features) // 2 + 1))
        ])
        self.W_k = nn.ParameterList([
            nn.Parameter(torch.randn((num_nodes * in_features) // 2 + 1, (num_nodes * in_features) // 2 + 1)),
            nn.Parameter(torch.randn((num_nodes * in_features) // 2 + 1, (num_nodes * in_features) // 2 + 1))
        ])
        self.W_v = nn.ParameterList([
            nn.Parameter(torch.randn((num_nodes * in_features) // 2 + 1, (num_nodes * in_features) // 2 + 1)),
            nn.Parameter(torch.randn((num_nodes * in_features) // 2 + 1, (num_nodes * in_features) // 2 + 1))
        ])'''

    def forward(self, input, adj):
        # input = node_wise_operation(input)
        res = input
        input = F.dropout(input, self.dropout, training=self.training)
        # input = self.node_wise_matrix(input) + self.FGCN(input) + res
        input = self.node_wise_matrix(input) + res
        # input = self.NodeWiseTransform(input)
        batch_size, N, _ = input.size()
        if adj.dim() == 3:
            adj = adj[:,:,1].unsqueeze(2).repeat(1,1,adj.shape[1])  # 扩展为 [batch_size, N, N]
        elif adj.size(0) != batch_size:
            adj = adj[:,:].unsqueeze(0).repeat(batch_size, 1, 1)

        h = torch.matmul(input, self.W)  # [batch_size, N, out_features]
        h = self.norm(h)

        residential = h

        h_repeated1 = h.unsqueeze(2).expand(-1, -1, N, -1)  # [batch_size, N, N, out_features]
        h_repeated2 = h.unsqueeze(1).expand(-1, N, -1, -1)  # [batch_size, N, N, out_features]
        a_input = torch.cat([h_repeated1, h_repeated2], dim=-1)  # [batch_size, N, N, 2*out_features]

        e = self.leakyrelu(torch.matmul(a_input, self.a).squeeze(-1))  # [batch_size, N, N]

        if adj.dim() == 2:
            adj = adj.unsqueeze(0).expand(batch_size, -1, -1)  # 扩展为 [batch_size, N, N]

        zero_vec = -9e15 * torch.ones_like(e)
        attention = torch.where(adj > 0, e, zero_vec)  # [batch_size, N, N]

        attention = F.softmax(attention, dim=-1)
        attention = F.dropout(attention, self.dropout, training=self.training)

        h_prime = torch.matmul(attention, h) + residential  # [batch_size, N, out_features]

        if self.concat:
            return F.elu(h_prime)
        else:
            return h_prime
        
    def node_wise_matrix(self, x):
        return x * self.node_weights.view(1, -1, 1) + self.node_bias.view(1, -1, 1) 
    
    def tokenEmb(self, x):
        x = x.unsqueeze(2)
        y = self.embeddings
        return x * y
    
    def FGCN(self, x):
        B, N, L = x.shape
        res = x
        # B*N*L ==> B*NL
        x = x.reshape(B, -1)
        # embedding B*NL ==> B*NL*D
        x = self.tokenEmb(x)
        x = torch.fft.rfft(x, dim=1, norm='ortho')

        x = x.reshape(B, self.frequency_size, (N*L)//2+1)

        # FourierGNN
        # x = self.fourierGC(x, B, (N*L)//2 + 1, self.frequency_size)

        x = checkpoint(
            self.freq_attention,  # 要包装的函数
            x,                    # 第一个输入参数
            B,                    # 第二个参数batch_size
            (N*L)//2+1,                 # 频率维度大小
            self.frequency_size,                    # 特征维度大小
            preserve_rng_state=False,  # 不保存RNG状态以节省内存
            use_reentrant=False   # 推荐设置（适用于PyTorch 1.11+）
        )

        x = x.reshape(B, (N*L)//2+1, self.embed_size)

        # ifft
        x = torch.fft.irfft(x, n=N*L, dim=1, norm="ortho")
        x = x.reshape(B, N, L, self.embed_size)
        x = self.decoder2(x)
        x = x.view(B, N, L)
        return x + res
    
    # FourierGNN
    def fourierGC(self, x, B, N, L):
        o1_real = torch.zeros([B, (N*L)//2 + 1, self.frequency_size * self.hidden_size_factor],
                              device=x.device)
        o1_imag = torch.zeros([B, (N*L)//2 + 1, self.frequency_size * self.hidden_size_factor],
                              device=x.device)

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

        x = torch.view_as_complex(x)
        return x

    def freq_attention(self, x, B, N, L):
        x_real = x.real
        x_imag = x.imag

        Q_real = torch.einsum('bli,io->blo', x_real, self.W_q[0]) - torch.einsum('bli,io->blo', x_imag, self.W_q[1])
        Q_imag = torch.einsum('bli,io->blo', x_imag, self.W_q[0]) + torch.einsum('bli,io->blo', x_real, self.W_q[1])
        Q = torch.stack([Q_real, Q_imag], dim=-1)

        K_real = torch.einsum('bli,io->blo', x_real, self.W_k[0]) - torch.einsum('bli,io->blo', x_imag, self.W_k[1])
        K_imag = torch.einsum('bli,io->blo', x_imag, self.W_k[0]) + torch.einsum('bli,io->blo', x_real, self.W_k[1])
        K = torch.stack([K_real, K_imag], dim=-1)

        V_real = torch.einsum('bli,io->blo', x_real, self.W_v[0]) - torch.einsum('bli,io->blo', x_imag, self.W_v[1])
        V_imag = torch.einsum('bli,io->blo', x_imag, self.W_v[0]) + torch.einsum('bli,io->blo', x_real, self.W_v[1])
        V = torch.stack([V_real, V_imag], dim=-1)

        Q_complex = torch.view_as_complex(Q)
        K_complex = torch.view_as_complex(K)
        V_complex = torch.view_as_complex(V)
        
        scale = 1 / math.sqrt(N)

        scores = torch.einsum('bik,bjk->bij', Q_complex, K_complex) * scale

        mask = torch.triu(
            torch.ones(L, L, dtype=torch.bool, device=x.device), 
            diagonal=1
        ).transpose(0, 1)
        mask = mask.unsqueeze(0).expand(B, -1, -1)
        scores = scores.masked_fill(mask, -float('inf'))

        attention = torch.softmax(scores.real, dim=-1)
        output = attention @ V_complex.real

        return output

def node_wise_operation(x):
    mean = x.mean(dim=-1, keepdim=True)  # [batch, nodes, 1]
    std = x.std(dim=-1, keepdim=True)    # [batch, nodes, 1]
    return (x - mean) / (std + 1e-8)

class GateFusion(nn.Module):
    def __init__(self, in_dim):
        super().__init__()
        self.gate_block=nn.Sequential(
             nn.Linear(in_dim*2,in_dim),
             nn.Dropout(p=.5),
             nn.LayerNorm(in_dim),
             nn.GELU(),
             nn.Linear(in_dim,in_dim),
             nn.Sigmoid()
         )
         
        self._initialize_weights()

    def _initialize_weights(self):  
        """Kaiming initialization with fan-out mode"""
        for m in self.modules():
            if isinstance(m, nn.Linear):                      
                nn.init.xavier_normal_(m.weight)
    
    def forward(self, heads):
        avg_pool = torch.mean(heads, dim=0)
        max_pool = torch.max(heads, dim=0)[0]
        gate = self.gate_block(torch.cat([avg_pool,max_pool],dim=-1))
        return gate * avg_pool + (1-gate) * max_pool

class GAT_Multi(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout, alpha, nheads, adj, num_nodes = 247, hidden_size_factor=1, embed_size=32, sparsity_threshold=0.01):
        super(GAT_Multi, self).__init__()
        self.adj = adj
        self.dropout = dropout
        self.embed_size = embed_size
        self.number_frequency = 1
        self.frequency_size = self.embed_size // self.number_frequency
        self.embeddings = nn.Parameter(torch.randn(1, self.embed_size))
        self.decoder2 = nn.Linear(self.embed_size, 1)
        self.hidden_size_factor = hidden_size_factor
        self.scale = 0.02
        self.sparsity_threshold=sparsity_threshold

        self.attentions = nn.ModuleList([
            GraphAttentionLayer(nfeat, nhid//nheads, dropout, alpha) 
            # GraphAttentionLayer(nfeat, nhid, dropout, alpha) 
            for _ in range(nheads)
        ])
        for i, attention in enumerate(self.attentions):
            self.add_module('attention_{}'.format(i), attention)
        self.norm = nn.LayerNorm(nhid)
        self.out_att = GraphAttentionLayer(nhid * nheads, nclass, dropout=dropout, alpha=alpha, concat=False)
        self.encoder = nn.Linear(2, 1)
        self.activate = nn.LeakyReLU(0.01)
        self.decoder = nn.Linear(nfeat, 1)
        self.mapping = nn.Linear(nfeat, nfeat)

        self.gate_fusion = GateFusion(in_dim=nhid)

        self.W_q = nn.ParameterList([
            nn.Parameter(torch.randn((num_nodes * nfeat) // 2 + 1, (num_nodes * nfeat) // 2 + 1)),
            nn.Parameter(torch.randn((num_nodes * nfeat) // 2 + 1, (num_nodes * nfeat) // 2 + 1))
        ])
        self.W_k = nn.ParameterList([
            nn.Parameter(torch.randn((num_nodes * nfeat) // 2 + 1, (num_nodes * nfeat) // 2 + 1)),
            nn.Parameter(torch.randn((num_nodes * nfeat) // 2 + 1, (num_nodes * nfeat) // 2 + 1))
        ])
        self.W_v = nn.ParameterList([
            nn.Parameter(torch.randn((num_nodes * nfeat) // 2 + 1, (num_nodes * nfeat) // 2 + 1)),
            nn.Parameter(torch.randn((num_nodes * nfeat) // 2 + 1, (num_nodes * nfeat) // 2 + 1))
        ])

    def forward(self, x, prc):
        residual = x
        
        # x = F.dropout(x, self.dropout, training=self.training)
        multi_head_outputs = []
        for att in self.attentions:
            att_output = att(x, self.adj)  # [batch_size, N, nhid]qqq
            att_output = self.norm(att_output)
            multi_head_outputs.append(att_output)

        heads_stack = torch.stack(multi_head_outputs, dim=1)  # [B,num_heads,N,dim]
        fused_features = []

        for i in range(heads_stack.size(2)):  # 对每个节点独立做融合
            node_features = heads_stack[:, :, i]      # [B,num_heads,dim]
            fused_node_feature = self.gate_fusion(node_features)   # [B,dim]
            fused_features.append(fused_node_feature)
        x = torch.stack(fused_features, dim=1)
        
        x = F.dropout(x, self.dropout, training=self.training)
        x = F.elu(self.out_att(x, self.adj)) + self.FGCN(residual) + residual  # [batch_size, N, nclass]
        x = torch.stack([x, prc], dim=3)
        x = self.encoder(x)
        x = torch.squeeze(x) + residual
        x = self.activate(x)
        x = self.decoder(x)
        return x[:,:,-1]
    
    def tokenEmb(self, x):
        x = x.unsqueeze(2)
        y = self.embeddings
        return x * y
    
    def FGCN(self, x):
        B, N, L = x.shape
        res = x
        # B*N*L ==> B*NL
        x = x.reshape(B, -1)
        # embedding B*NL ==> B*NL*D
        x = self.tokenEmb(x)
        x = torch.fft.rfft(x, dim=1, norm='ortho')

        x = x.reshape(B, self.frequency_size, (N*L)//2+1)

        # FourierGNN
        # x = self.fourierGC(x, B, (N*L)//2 + 1, self.frequency_size)

        x = checkpoint(
            self.freq_attention,  # 要包装的函数
            x,                    # 第一个输入参数
            B,                    # 第二个参数batch_size
            (N*L)//2+1,                 # 频率维度大小
            self.frequency_size,                    # 特征维度大小
            preserve_rng_state=False,  # 不保存RNG状态以节省内存
            use_reentrant=False   # 推荐设置（适用于PyTorch 1.11+）
        )

        x = x.reshape(B, (N*L)//2+1, self.embed_size)

        # ifft
        x = torch.fft.irfft(x, n=N*L, dim=1, norm="ortho")
        x = x.reshape(B, N, L, self.embed_size)
        x = self.decoder2(x)
        x = x.view(B, N, L)
        return x + res
    
    # FourierGNN
    def fourierGC(self, x, B, N, L):
        o1_real = torch.zeros([B, (N*L)//2 + 1, self.frequency_size * self.hidden_size_factor],
                              device=x.device)
        o1_imag = torch.zeros([B, (N*L)//2 + 1, self.frequency_size * self.hidden_size_factor],
                              device=x.device)

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

        x = torch.view_as_complex(x)
        return x

    def freq_attention(self, x, B, N, L):
        x_real = x.real
        x_imag = x.imag

        Q_real = torch.einsum('bli,io->blo', x_real, self.W_q[0]) - torch.einsum('bli,io->blo', x_imag, self.W_q[1])
        Q_imag = torch.einsum('bli,io->blo', x_imag, self.W_q[0]) + torch.einsum('bli,io->blo', x_real, self.W_q[1])
        Q = torch.stack([Q_real, Q_imag], dim=-1)

        K_real = torch.einsum('bli,io->blo', x_real, self.W_k[0]) - torch.einsum('bli,io->blo', x_imag, self.W_k[1])
        K_imag = torch.einsum('bli,io->blo', x_imag, self.W_k[0]) + torch.einsum('bli,io->blo', x_real, self.W_k[1])
        K = torch.stack([K_real, K_imag], dim=-1)

        V_real = torch.einsum('bli,io->blo', x_real, self.W_v[0]) - torch.einsum('bli,io->blo', x_imag, self.W_v[1])
        V_imag = torch.einsum('bli,io->blo', x_imag, self.W_v[0]) + torch.einsum('bli,io->blo', x_real, self.W_v[1])
        V = torch.stack([V_real, V_imag], dim=-1)

        Q_complex = torch.view_as_complex(Q)
        K_complex = torch.view_as_complex(K)
        V_complex = torch.view_as_complex(V)
        
        scale = 1 / math.sqrt(N)

        scores = torch.einsum('bik,bjk->bij', Q_complex, K_complex) * scale

        mask = torch.triu(
            torch.ones(L, L, dtype=torch.bool, device=x.device), 
            diagonal=1
        ).transpose(0, 1)
        mask = mask.unsqueeze(0).expand(B, -1, -1)
        scores = scores.masked_fill(mask, -float('inf'))

        attention = torch.softmax(scores.real, dim=-1)
        output = attention @ V_complex.real

        return output

class NodeWiseTransform(nn.Module):
    def __init__(self, 
                 num_nodes: int, 
                 use_weights: bool = True,
                 use_bias: bool = True,
                 activation: str = None,
                 init_method: str = 'xavier'):
        """
        高级节点级变换层
        参数：
            num_nodes: 节点数量
            use_weights: 是否启用可学习权重
            use_bias: 是否启用可学习偏置
            activation: 激活函数类型（'relu','sigmoid','tanh'等）
            init_method: 参数初始化方法 ('xavier', 'kaiming', 'normal')
        """
        super().__init__()
        self.num_nodes = num_nodes
        self.use_weights = use_weights
        self.use_bias = use_bias

        # 权重参数初始化
        if use_weights:
            self.weights = nn.Parameter(torch.Tensor(num_nodes))
            self._init_parameter(self.weights, init_method)
        else:
            self.register_parameter('weights', None)

        # 偏置参数初始化
        if use_bias:
            self.bias = nn.Parameter(torch.Tensor(num_nodes))
            nn.init.zeros_(self.bias)
        else:
            self.register_parameter('bias', None)

        # 激活函数配置
        self.activation = None
        if activation:
            self.activation = getattr(nn, activation.capitalize() + '()', None)
            if not self.activation:
                raise ValueError(f"Unsupported activation: {activation}")

    def _init_parameter(self, tensor, method):
        """修正后的初始化方法"""
        if tensor.dim() == 1:
            if method == 'xavier':
                # 一维参数改用均匀分布初始化
                nn.init.uniform_(tensor, a=-0.1, b=0.1)
            elif method == 'kaiming':
                nn.init.normal_(tensor, mean=0, std=1.0/math.sqrt(tensor.size(0)))
            elif method == 'normal':
                nn.init.normal_(tensor, mean=0, std=0.01)
        else:
            # 保留原有二维初始化逻辑
            if method == 'xavier':
                nn.init.xavier_normal_(tensor)
            elif method == 'kaiming':
                nn.init.kaiming_normal_(tensor)
            elif method == 'normal':
                nn.init.normal_(tensor, mean=0, std=0.01)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        输入形状: [batch, nodes, time]
        输出形状: [batch, nodes, time]
        """
        assert x.size(1) == self.num_nodes, \
            f"节点数量不匹配，预期{self.num_nodes}，实际输入{x.size(1)}"

        # 应用权重
        if self.use_weights:
            weight_matrix = self.weights.view(1, -1, 1)  # 广播维度
            x = x * weight_matrix

        # 应用偏置
        if self.use_bias:
            bias_matrix = self.bias.view(1, -1, 1)
            x = x + bias_matrix

        # 应用激活函数
        if self.activation is not None:
            x = self.activation(x)

        return x

    def extra_repr(self) -> str:
        """打印配置信息"""
        return f"nodes={self.num_nodes}, weight={self.use_weights}, bias={self.use_bias}"


