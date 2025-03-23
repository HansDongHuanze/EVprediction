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

class CoupFourGAT(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout, alpha, nheads, adj, num_nodes = 247, hidden_size_factor=1, embed_size=32, sparsity_threshold=0.01, levels=3):
        super(CoupFourGAT, self).__init__()
        self.adj = adj
        self.nfeat = nfeat
        self.dropout = dropout
        self.nheads = nheads
        self.embed_size = embed_size
        self.number_frequency = 1
        self.frequency_size = self.embed_size // self.number_frequency
        self.embeddings = nn.Parameter(torch.randn(1, self.embed_size))
        self.decoder2 = nn.Linear(self.embed_size, 1)
        self.hidden_size_factor = hidden_size_factor
        self.scale = 0.02
        self.sparsity_threshold=sparsity_threshold

        self.attentions = nn.ModuleList([
            CFGATLayer(nfeat // 2, nfeat // 2, dropout, alpha) 
            # GraphAttentionLayer(nfeat, nhid, dropout, alpha) 
            for _ in range(nheads)
        ])
        for i, attention in enumerate(self.attentions):
            self.add_module('attention_{}'.format(i), attention)
        self.norm = nn.LayerNorm(nfeat // 2)
        # self.out_att = GraphAttentionLayer(nhid * nheads, nclass, dropout=dropout, alpha=alpha, concat=False)
        self.encoder = nn.Linear(2, 1)
        self.activate = nn.LeakyReLU(0.01)
        self.decoder = nn.Linear(nfeat, 1)
        self.mapping = nn.Linear(nfeat, nfeat)
        self.norm_2 = nn.LayerNorm(self.nfeat)
        self.freq_conv = nn.Conv2d(
            in_channels=2,   # 输入通道数
            out_channels=2,  # 输出通道数
            kernel_size=(3, 3),  # 卷积核大小
            stride=(1, 1),       # 步幅
            padding=(1, 1)       # 填充策略，使用1以确保输出尺寸不变
        )

        self.att_map = nn.Linear(num_nodes, self.nfeat // 2)

        self.gate_fusion = GateFusion(in_dim=nfeat // 2)

        self.W_q = nn.ParameterList([
            nn.Parameter(torch.randn(self.nfeat // 2, self.nfeat // 2)),
            nn.Parameter(torch.randn(self.nfeat // 2, self.nfeat // 2))
        ])
        self.W_k = nn.ParameterList([
            nn.Parameter(torch.randn(self.nfeat // 2, self.nfeat // 2)),
            nn.Parameter(torch.randn(self.nfeat // 2, self.nfeat // 2))
        ])
        self.W_v = nn.ParameterList([
            nn.Parameter(torch.randn(self.nfeat // 2, self.nfeat // 2)),
            nn.Parameter(torch.randn(self.nfeat // 2, self.nfeat // 2))
        ])
        self.attention_blocks = AttentionBlocks(
            self.W_q, self.W_k, self.W_v, self.att_map, num_nodes, nfeat
        )

        self.complexMapping = nn.Linear(self.nfeat // 2, self.nfeat // 2)
    
    def forward(self, x, prc):
        residual = x
        
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.norm_2(self.FGCN(x)) + residual
         
        # x = F.dropout(x, self.dropout, training=self.training)
        # x = F.elu(self.out_att(x, self.adj)) + self.FGCN(residual) + residual  # [batch_size, N, nclass]
        x = torch.stack([x, prc], dim=3)
        x = self.encoder(x)
        x = torch.squeeze(x)
        x = self.activate(x)
        x = self.decoder(x) + residual
        return x[:,:,-1]
    
    def tokenEmb(self, x):
        x = x.unsqueeze(2)
        y = self.embeddings
        return x * y
    
    def atten_com(self, x):
        multi_head_outputs = []
        for att in self.attentions:
            att_output = att(x, self.adj)  # [batch_size, N, nhid]qqq
            att_output = self.norm(att_output)
            multi_head_outputs.append(att_output)

        heads_stack = torch.stack(multi_head_outputs, dim=1)  # [B,num_heads,N,dim]

        fused_features = []

        for i in range(heads_stack.size(2)):
            node_features = heads_stack[:, :, i, :]  # [B=512, num_heads=1, in_dim=4]
            fused_node = self.gate_fusion(node_features.reshape(self.nheads,-1,self.nfeat // 2))  # [512,7]
            fused_features.append(fused_node + node_features.mean(dim=1))
        x = torch.stack(fused_features, dim=1)  # [512,247,7]
        return x

    def FGCN(self, x):
        B, N, L = x.shape
        res = x
        # B*N*L ==> B*NL
        # x = x.reshape(B, -1)
        # embedding B*NL ==> B*NL*D
        # x = self.tokenEmb(x)
        # print(x.shape)

        x = torch.fft.fft(x, dim=-1, norm="ortho")
        res_complex = x
        x = x[:,:,:(self.nfeat // 2)]
        half_res_complex = x

        x = checkpoint(
            self.freq_convolution,  # 要包装的函数
            x,                    # 第一个输入参数
            B,                    # 第二个参数batch_size
            N,                 # 频率维度大小
            self.nfeat // 2,                    # 特征维度大小
            preserve_rng_state=False,  # 不保存RNG状态以节省内存
            use_reentrant=False   # 推荐设置（适用于PyTorch 1.11+）
        )

        x = x + half_res_complex

        x = checkpoint(
            self.fourierGC,
            x,
            use_reentrant=False,
            preserve_rng_state=False
        )

        x = x + half_res_complex

        real_image = torch.stack([x.real, x.imag], dim = 0)
        half = self.complexMapping(real_image)
        half = self.activate(half)
        half_comp = torch.view_as_complex(torch.stack([half[0],half[1]],dim=-1))
        conj_half = torch.conj(half_comp)
        x = torch.cat((half_comp, conj_half), dim = 2)
        x = x + res_complex

        x = torch.fft.ifft(x, dim=-1, norm="ortho").real
        return x + res

    # FourierGNN
    def fourierGC(self, x):
        o1_real = self.atten_com(x.real)
        o1_imag = self.atten_com(x.imag)

        y = torch.stack([o1_real, o1_imag], dim=-1)
        y = F.softshrink(y, lambd=self.sparsity_threshold)

        x = torch.view_as_complex(y)
        return x

    def freq_convolution(self, x, B, N, L):
        x_real = x.real
        x_imag = x.imag
        vec = torch.stack([x_real, x_imag], dim = -1)
        res = vec
        vec = torch.reshape(vec, (B, 2, N, L))
        x = self.freq_conv(vec)
        x = torch.reshape(x, (B, N, L, 2))
        x = self.activate(x) + res
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

        # Make sure L is matching with the intended size of scores
        mask = torch.triu(torch.ones(scores.size(2), scores.size(2), dtype=torch.bool, device=x.device), diagonal=1)
        mask = mask.unsqueeze(0).expand(scores.size(0), -1, -1)  # 使用scores的batch_size维度
        scores = scores.masked_fill(mask, -float('inf'))

        real_softmax = torch.softmax(scores.real, dim=-1)
        imag_softmax = torch.softmax(scores.imag, dim=-1)

        real_temp = real_softmax @ V_complex.real
        imag_temp = imag_softmax @ V_complex.imag

        attention = torch.stack([real_temp, imag_temp], dim=-1)
        return torch.view_as_complex(attention)

class CFGATLayer(nn.Module):
    def __init__(self, in_features, out_features, dropout, alpha, 
                 concat=True, num_nodes=247):
        super(CFGATLayer, self).__init__()
        self.dropout = dropout
        self.in_features = in_features
        self.out_features = out_features
        self.alpha = alpha
        self.concat = concat
        self.num_nodes = num_nodes

        self.W = nn.Parameter(torch.zeros(size=(in_features, out_features)))
        nn.init.kaiming_normal_(self.W, mode='fan_out', nonlinearity='leaky_relu')
        self.a = nn.Parameter(torch.empty(2*out_features, 1))
        nn.init.xavier_normal_(self.a, gain=nn.init.calculate_gain('leaky_relu', param=alpha))
        self.node_weights = nn.Parameter(torch.randn(num_nodes))
        self.node_bias = nn.Parameter(torch.zeros(num_nodes))
        # self.NodeWiseTransform = NodeWiseTransform(num_nodes)
        self.norm = nn.LayerNorm(out_features)

        self.leakyrelu = nn.LeakyReLU(self.alpha)

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

        e = self.leakyrelu(
            (h @ self.a[:(self.in_features)]).unsqueeze(2) +  # [B,N,1,1]
            (h @ self.a[self.in_features:]).unsqueeze(1)    # [B,1,N,1]
        ).squeeze(-1)

        if adj.dim() == 2:
            adj = adj.unsqueeze(0).expand(batch_size, -1, -1)  # 扩展为 [batch_size, N, N]

        zero_vec = -9e15 * torch.ones_like(e)
        attention = torch.where(adj > 0, e, zero_vec)  # [batch_size, N, N]

        attention = self.leakyrelu(attention)
        attention = F.dropout(attention, self.dropout, training=self.training)

        h_prime = torch.matmul(attention, h) + residential  # [batch_size, N, out_features]

        if self.concat:
            return F.elu(h_prime)
        else:
            return h_prime
        
    def node_wise_matrix(self, x):
        return x * self.node_weights.view(1, -1, 1) + self.node_bias.view(1, -1, 1) 
    
# 1. 定义注意力计算的子模块序列
class AttentionBlocks(nn.ModuleList):
    def __init__(self, W_q, W_k, W_v, att_map, num_nodes, nfeat):
        super().__init__()
        # 将原有freq_attention分解为4个计算阶段
        self.add_module("stage1_Q", QComputation(W_q))
        self.add_module("stage2_K", KComputation(W_k))
        self.add_module("stage3_V", VComputation(W_v))
        self.add_module("stage4_Output", AttentionOutput(att_map, num_nodes, nfeat))

# 2. 定义各个子计算模块
class QComputation(nn.Module):
    def __init__(self, W_q):
        super().__init__()
        self.W_q = W_q
    
    def forward(self, x_real, x_imag):
        Q_real = torch.einsum('bli,io->blo', x_real, self.W_q[0]) - torch.einsum('bli,io->blo', x_imag, self.W_q[1])
        Q_imag = torch.einsum('bli,io->blo', x_imag, self.W_q[0]) + torch.einsum('bli,io->blo', x_real, self.W_q[1])
        return torch.stack([Q_real, Q_imag], dim=-1), x_real, x_imag  # 保留中间结果

class KComputation(nn.Module):
    def __init__(self, W_k):
        super().__init__()
        self.W_k = W_k
    
    def forward(self, inputs):
        Q, x_real, x_imag = inputs
        K_real = torch.einsum('bli,io->blo', x_real, self.W_k[0]) - torch.einsum('bli,io->blo', x_imag, self.W_k[1])
        K_imag = torch.einsum('bli,io->blo', x_imag, self.W_k[0]) + torch.einsum('bli,io->blo', x_real, self.W_k[1])
        return Q, torch.stack([K_real, K_imag], dim=-1), x_real, x_imag

class VComputation(nn.Module):
    def __init__(self, W_v):
        super().__init__()
        self.W_v = W_v
    
    def forward(self, inputs):
        Q, K, x_real, x_imag = inputs
        V_real = torch.einsum('bli,io->blo', x_real, self.W_v[0]) - torch.einsum('bli,io->blo', x_imag, self.W_v[1])
        V_imag = torch.einsum('bli,io->blo', x_imag, self.W_v[0]) + torch.einsum('bli,io->blo', x_real, self.W_v[1])
        return Q, K, torch.stack([V_real, V_imag], dim=-1)

class AttentionOutput(nn.Module):
    def __init__(self, att_map, num_nodes, nfeat):
        super().__init__()
        self.att_map = att_map
        self.num_nodes = num_nodes
        self.nfeat = nfeat
        self.leakyrelu = nn.LeakyReLU(negative_slope=0.01)
        
    def forward(self, inputs):
        Q, K, V = inputs
        Q_complex = torch.view_as_complex(Q)
        K_complex = torch.view_as_complex(K)
        V_complex = torch.view_as_complex(V)
        
        scale = 1 / math.sqrt(self.num_nodes)
        scores = torch.einsum('bik,bjk->bij', Q_complex, K_complex) * scale
        
        # 优化mask生成
        B, N, _ = scores.shape
        mask = torch.triu(torch.ones(N, N, dtype=torch.bool, device=scores.device), diagonal=1)
        mask = mask.unsqueeze(0).expand(B, -1, -1)
        scores = scores.masked_fill(mask, -float('inf'))
        
        real_softmax = self.leakyrelu(scores.real, dim=-1)
        imag_softmax = self.leakyrelu(scores.imag, dim=-1)
        
        real_output = self.att_map(real_softmax @ V_complex.real)
        imag_output = self.att_map(imag_softmax @ V_complex.imag)
        
        return torch.stack([real_output, imag_output], dim=-1)    

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