import torch
import torch.nn as nn
import models
import torch.nn.functional as F
import functions as fn
import copy
import pytorch_wavelets as pw
from pytorch_wavelets import DWT1D
import math
from torch.utils.checkpoint import checkpoint

from torch.nn import Transformer, TransformerEncoder, TransformerEncoderLayer

use_cuda = True
device = torch.device("cuda:0" if use_cuda and torch.cuda.is_available() else "cpu")
fn.set_seed(seed=2023, flag=True)

class WaveSTFTGAT(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout, alpha, nheads, adj, 
                 num_nodes = 247, hidden_size_factor=1, embed_size=32, sparsity_threshold=0.01, wavelet='db4', level=3,):
        super(WaveSTFTGAT, self).__init__()
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
        self.num_nodes = num_nodes

        self.wavelet_type = wavelet
        self.decomp_level = level

        self.dwt_layer = DWT1D(wave=self.wavelet_type, J=self.decomp_level)

        # Calculate input dimensions for both branches
        dummy_input = torch.randn(1, num_nodes, nfeat)
        
        # Wavelet branch dimension calculation
        yL, yH = self.dwt_layer(dummy_input)
        coeffs_list = [yL] + [h.unsqueeze(2) for h in yH]
        wavelet_dim = sum(c.reshape(1, num_nodes, -1).shape[-1] for c in coeffs_list)

        self.wavelet_attention = nn.TransformerEncoderLayer(wavelet_dim, nhead=1)
        self.wavelet_linear = nn.Linear(wavelet_dim, nfeat)

        self.attentions_spart = nn.ModuleList([
            CFGATLayer(nfeat, nfeat, dropout, alpha) 
            # GraphAttentionLayer(nfeat, nhid, dropout, alpha) 
            for _ in range(nheads)
        ])

        for i, attention in enumerate(self.attentions_spart):
            self.add_module('attention_{}'.format(i), attention)

        self.norm = nn.LayerNorm(nfeat)
        # self.out_att = GraphAttentionLayer(nhid * nheads, nclass, dropout=dropout, alpha=alpha, concat=False)
        self.encoder = nn.Linear(2, 1)
        self.activate = nn.LeakyReLU(0.01)
        self.sigmoid = nn.Sigmoid()
        self.decoder = nn.Linear(nfeat, 1)
        self.mapping = nn.Linear(129, num_nodes)
        self.norm_2 = nn.LayerNorm(self.nfeat)
        self.freq_conv = nn.Conv2d(
            in_channels=2,   # 输入通道数
            out_channels=2,  # 输出通道数
            kernel_size=(3, 3),  # 卷积核大小
            stride=(1, 1),       # 步幅
            padding=(1, 1)       # 填充策略，使用1以确保输出尺寸不变
        )

        self.freq_wave_conv = nn.Conv2d(
            in_channels=4,   # 输入通道数
            out_channels=1,  # 输出通道数
            kernel_size=(3, 3),  # 卷积核大小
            stride=(1, 1),       # 步幅
            padding=(1, 1)       # 填充策略，使用1以确保输出尺寸不变
        )

        self.freq_wave_linear = nn.Linear(4, 1)

        self.freq_time_conv1 = nn.Conv2d(
            in_channels=2,   # 输入通道数
            out_channels=1,  # 输出通道数
            kernel_size=(3, 3),  # 卷积核大小
            stride=(1, 1),       # 步幅
            padding=(1, 1)       # 填充策略，使用1以确保输出尺寸不变
        )
        self.freq_time_conv2 = nn.Conv2d(
            in_channels=2,   # 输入通道数
            out_channels=1,  # 输出通道数
            kernel_size=(3, 3),  # 卷积核大小
            stride=(1, 1),       # 步幅
            padding=(1, 1)       # 填充策略，使用1以确保输出尺寸不变
        )

        self.att_map = nn.Linear(num_nodes, self.nfeat)

        self.gate_fusion = GateFusion(in_dim=nfeat)

        self.W_q = nn.ParameterList([
            nn.Parameter(torch.randn(self.nfeat, self.nfeat)),
            nn.Parameter(torch.randn(self.nfeat, self.nfeat))
        ])
        self.W_k = nn.ParameterList([
            nn.Parameter(torch.randn(self.nfeat, self.nfeat)),
            nn.Parameter(torch.randn(self.nfeat, self.nfeat))
        ])
        self.W_v = nn.ParameterList([
            nn.Parameter(torch.randn(self.nfeat, self.nfeat)),
            nn.Parameter(torch.randn(self.nfeat, self.nfeat))
        ])
        self.attention_blocks = AttentionBlocks(
            self.W_q, self.W_k, self.W_v, self.att_map, num_nodes, nfeat
        )

        self.complexMapping = nn.Linear(self.nfeat, self.nfeat)
    
    def forward(self, x, prc):
        residual = x
        
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.FGCN(x) + residual
         
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
    
    def atten_com_spart(self, x):
        res = x
        multi_head_outputs = []
        for att in self.attentions_spart:
            att_output = att(x, self.adj)  # [batch_size, N, nhid]qqq
            att_output = self.norm(att_output)
            multi_head_outputs.append(att_output)

        heads_stack = torch.stack(multi_head_outputs, dim=1)  # [B,num_heads,N,dim]

        fused_features = []

        for i in range(heads_stack.size(2)):
            node_features = heads_stack[:, :, i, :]  # [B=512, num_heads=1, in_dim=4]
            fused_node = self.gate_fusion(node_features.reshape(self.nheads,-1,self.nfeat))  # [512,7]
            fused_features.append(fused_node + node_features.mean(dim=1))
        x = torch.stack(fused_features, dim=1)  # [512,247,7]
        return x
    
    def wavelet_transform(self, x):
        B, N, L = x.shape  # [512, 247, 12]

        yL, yH = self.dwt_layer(x)
        coeffs_list = [yL] + [h.unsqueeze(2) for h in yH]
        coeffs_flat = torch.cat([
            c.reshape(B,N, -1)
            for c in coeffs_list
        ], dim=-1)
        return coeffs_flat
    
    def atten_com_temp(self, x):
        res = x
        multi_head_outputs = []
        for att in self.attentions_temp:
            att_output = att(x, self.adj)  # [batch_size, N, nhid]qqq
            att_output = self.norm(att_output)
            multi_head_outputs.append(att_output)

        heads_stack = torch.stack(multi_head_outputs, dim=1)  # [B,num_heads,N,dim]

        fused_features = []

        for i in range(heads_stack.size(2)):
            node_features = heads_stack[:, :, i, :]  # [B=512, num_heads=1, in_dim=4]
            fused_node = self.gate_fusion(node_features.reshape(self.nheads,-1,self.nfeat))  # [512,7]
            fused_features.append(fused_node + node_features.mean(dim=1))
        x = torch.stack(fused_features, dim=1)  # [512,247,7]
        return x

    def FGCN(self, x):
        res = x
        B, N, L = x.shape  # [512, 247, 12]
        x_reshaped = x.reshape(B, -1)  # [512, 2964]

        n_fft = 256
        target_frames = 12
        seq_len = 2964
        hop_length = (seq_len - n_fft) // (target_frames - 1)  # = (2964-256)//11 = 246

        calculated_frames = (seq_len - n_fft) // hop_length + 1
        if calculated_frames != target_frames:
            hop_length = (seq_len - n_fft + 1) // target_frames

        x_stft = torch.stft(x_reshaped, n_fft=n_fft, hop_length=246,
                        win_length=n_fft, return_complex=True)

        x_stft = x_stft[..., :12]  # [512, 129, 12]

        x_stft = x_stft.reshape(B, L, -1)  # [512, 12, 129]
        x_stft_real = self.mapping(x_stft.real)
        x_stft_imag = self.mapping(x_stft.imag)
        x_stft = torch.stack([x_stft_real, x_stft_imag], dim=-1)
        x_stft = torch.view_as_complex(x_stft)
        x_stft = x_stft.reshape(B, N, -1)  # [512, 247, 12]

        x_real_wave = self.wavelet_transform(x_stft.real)
        x_real_wave = self.wavelet_linear(x_real_wave)
        x_imag_wave = self.wavelet_transform(x_stft.imag)
        x_imag_wave = self.wavelet_linear(x_imag_wave)
        x_stft = torch.stack([x_real_wave, x_imag_wave], dim=-1)
        x_stft = torch.view_as_complex(x_stft)

        x = checkpoint(
            self.freq_time_fusion,
            x,
            x_stft,
            B,
            N,
            self.nfeat,
            preserve_rng_state=False,
            use_reentrant=False
        )

        x = torch.stack([x[:,:,:,0], x[:,:,:,1]], dim=-1)
        x = torch.view_as_complex(x)

        x = checkpoint(
            self.fourierGC,
            x,
            use_reentrant=False,
            preserve_rng_state=False
        )

        x_vec = torch.stack([x.real, x.imag], dim = -1)
        x = self.encoder(x_vec).squeeze()

        return x
    
    # FourierGNN
    def fourierGC(self, x):
        res = x
        o1_real = self.atten_com_spart(x.real)
        o1_imag = self.atten_com_spart(x.imag)

        y = torch.stack([o1_real, o1_imag], dim=-1)
        y = F.softshrink(y, lambd=self.sparsity_threshold)

        x = torch.view_as_complex(y)
        return x
    
    # WaveletGNN
    def waveletGC(self, x):
        res = x
        x = self.atten_com_spart(x)
        return x

    def freq_time_fusion(self, x, stft, B, N, L):
        # Assuming stft has shape (B, N, L, 2) due to complex number representation
        input_1 = torch.stack([x, stft.real], dim=-1).squeeze()
        input_1 = torch.reshape(input_1, (B, 2, N, L))
        input_1 = self.freq_time_conv1(input_1)
        input = self.sigmoid(input_1)

        input_2 = torch.stack([x, stft.imag], dim=-1).squeeze()
        input_2 = torch.reshape(input_2, (B, 2, N, L))
        input_2 = self.freq_time_conv2(input_2)
        input = self.sigmoid(input_2)

        input = torch.stack([input_1, input_2], dim=-1).squeeze()
        input = torch.reshape(input, (B, 2, N, L))
        input = self.freq_conv(input)
        input = torch.reshape(input, (B, N, L, -1))
        input = self.activate(input)
        return input
    
    def wavelet_time_fusion(self, x, stft, wave, B, N, L):
        stft_time = self.freq_time_fusion(x, stft, B, N, L)
        input = torch.stack([x, wave, stft_time[:,:,:,0], stft_time[:,:,:,-1]], dim=-1).squeeze()
        # input = torch.stack([x, wave, stft.real, stft.imag], dim=-1).squeeze()
        input = torch.reshape(input, (B * 4, N, L))
        input = self.atten_com_temp(input)
        input = torch.reshape(input, (B, N, L, -1))
        input = self.freq_wave_linear(input)
        input = torch.reshape(input, (B, N, L))
        input = self.activate(input)
        return input + x

    def freq_time_GAT(self, x):
        res = x
        x = self.atten_com(x)
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
    
    def wavelet_transform_along_last_dim_gpu(
        self,
        tensor: torch.Tensor, 
        wavelet: str = 'db1', 
        level: int = 1
    ) -> torch.Tensor:
        # 定义小波滤波器（以db1为例）
        wavelet_dict = {
            'db1': {
                'dec_lo': [0.7071, 0.7071],  # 低通滤波器
                'dec_hi': [-0.7071, 0.7071] # 高通滤波器
            }
        }
        dec_lo = torch.tensor(wavelet_dict[wavelet]['dec_lo'], 
                            dtype=tensor.dtype, device=device).view(1, 1, -1)
        dec_hi = torch.tensor(wavelet_dict[wavelet]['dec_hi'],
                            dtype=tensor.dtype, device=device).view(1, 1, -1)
        
        # dec_lo_expanded = dec_lo.repeat(self.num_nodes, self.num_nodes, 1)
        # dec_hi_expanded = dec_hi.repeat(self.num_nodes, self.num_nodes, 1)
        
        # 多级分解
        coeffs = []
        current = tensor
        for _ in range(level):
            pad = len(dec_lo) - 1
            batch_size, num_nodes, seq_len = current.shape
            current_merged = current.view(batch_size * num_nodes, 1, seq_len)  # 合并批次和节点维度
            current_padded = F.pad(current_merged, (pad//2, pad - pad//2))     # 统一填充
            cA = F.conv1d(current_padded, dec_lo, stride=2)
            cD = F.conv1d(current_padded, dec_hi, stride=2)
            cA = cA.view(batch_size, num_nodes, -1)  # 恢复原始维度
            cD = cD.view(batch_size, num_nodes, -1)
            coeffs.append(cD)
            current = cA
        coeffs.append(current)
        adjusted_coeffs = [coeffs[-1]] + list(reversed(coeffs[:-1]))
        # 拼接系数
        return torch.cat(adjusted_coeffs, dim=-1)

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

        h_prime = torch.matmul(attention, h)  # [batch_size, N, out_features]

        if self.concat:
            return F.elu(h_prime) + residential
        else:
            return h_prime + residential
        
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